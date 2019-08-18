using Flux
using Flux.Optimise

# We define our model and load it onto the GPU
training_model = Chain(Conv((8, 8), 1=>32, relu; stride=(4, 4)),
                       Conv((4, 4), 32=>64, relu; stride=(2, 2)),
                       Conv((3, 3), 64=>64, relu; stride=(1, 1)),
                       x -> reshape(x, :, size(x, 4)),
                       Dense(3136, 512, relu),
                       Dense(512, ACTION_SIZE)) |> gpu

# Same goes for our target_model
target_model = deepcopy(training_model) |> gpu

opt = Optimise.ADAM(η)

# We have to define our own loss here, because using the Base.:^ operator
# gets us into trouble during gradient calculation, when using a GPU.
function mse_loss(x, y)
    diff = x .- y
    return sum(diff .* diff)/length(diff)
end


function huber_loss(x, y)
    diff = x .- y
    return sum(sqrt.(1 .+ diff .* diff) .- 1)/length(diff)
end

# Vectorizing the TD target calculation step.
function calculate_target_Q(reward_batch, terminal_batch, next_state_batch)
    training_model_actions = Flux.onecold(cpu(training_model(next_state_batch)))
    target_model_values = target_model(next_state_batch)[CartesianIndex.(training_model_actions, eachindex(training_model_actions))]
    reward_batch .+ γ .* target_model_values .* (1 .- terminal_batch)
end

function trainnetwork!()
    current_state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = sample(memory, BATCH_SIZE) |> gpu
    action_batch = Int.(action_batch)

    # Calculate the TD target values...
    target_Q_values_batch = calculate_target_Q(reward_batch, terminal_batch, next_state_batch)

    # Calculate the training network's Q values...
    model_output = training_model(current_state_batch)
    training_Q_values_batch = model_output[CartesianIndex.(action_batch, eachindex(action_batch))]

    # Calculate loss. Notice that we are detaching the `target_Q_values_batch`
    # from the graph when calculating the loss during TD loss caluculation. We
    # are only interested in gradients obtained from the training Q values batch.
    loss = huber_loss(training_Q_values_batch, target_Q_values_batch.data)

    # Get the gradients of the parameters of the training network wrt to the
    # calculated loss
    grads = Tracker.gradient(() -> loss, params(training_model))

    # Learning from our mistakes...
    Flux.Optimise.update!(opt, params(training_model), grads)

    # Fitted Q Iteration and saving model...
    if frame_idx % UPDATE_TIME == 0
        target_model = deepcopy(training_model)
        save_model(training_model, "")
    end

    return loss.data
end
