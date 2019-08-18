using CuArrays
using Flux
using Flux.Optimise
using Gym
using Images
using BSON: @save, @load
using Logging
using Gadfly
# ------------------------ Load game environment -------------------------------
env = nothing
# This is something we do to make it run on headless servers.
# Best not talk about it...
try
    global env
    env = make("Pong-greyNoFrameskip-v4")
catch
    global env
    env = make("Pong-greyNoFrameskip-v4")
end

#-------------------------Some utis-––––---------------------------------------

mutable struct MemoryArray
    current_state_arr::Array{Float32, 4}
    action_arr
    reward_arr::Array{Float32, 1}
    next_state_arr::Array{Float32, 4}
    terminal_arr
    size::Int64
    current_ptr::Int64
    filled_once::Bool
end

function MemoryArray(size::Integer)
    current_state_arr = zeros(Float32, 84, 84, 4, size)
    action_arr = zeros(Int64, size)
    reward_arr = zeros(Float32, size)
    next_state_arr = zeros(Float32, 84, 84, 4, size)
    terminal_arr = zeros(Bool, size)

    MemoryArray(current_state_arr, action_arr, reward_arr, next_state_arr, terminal_arr, size, 1, false)
end

function push!(memarr::MemoryArray, current_state, action, reward, next_state, terminal)
    idx = memarr.current_ptr
    memarr.current_state_arr[:, :, :, idx] .= current_state[:, :, :, 1]
    memarr.next_state_arr[:, :, :, idx]    .= next_state[:, :, :, 1]
    memarr.action_arr[idx]   = action
    memarr.reward_arr[idx]   = reward
    memarr.terminal_arr[idx] = terminal
    memarr.current_ptr  += 1
    if memarr.current_ptr > memarr.size
        memarr.current_ptr = 1
        memarr.filled_once = true
    end
end

function sample(memarr::MemoryArray, batch_size::Integer)

    final_elem = memarr.filled_once ? memarr.size : memarr.current_ptr - 1

    # Ideally, we want all entries to  be different
    idx = zeros(Int, batch_size)
    while true
        idx .= rand(1:final_elem, batch_size)
        length(unique(idx)) == batch_size && break
    end

    current_state_batch = memarr.current_state_arr[:, :, :, idx]
    action_batch = memarr.action_arr[idx]
    reward_batch = memarr.reward_arr[idx]
    next_state_batch = memarr.next_state_arr[:, :, :, idx]
    terminal_batch = memarr.terminal_arr[idx]

    return current_state_batch, action_batch, reward_batch, next_state_batch,
            terminal_batch
end

function save_model(model, info)
    wts = cpu.(Tracker.data(params(model)))

    @save "duel_dqn_$info.bson" wts
    @info("Model saved: duel_dqn_$info.bson")
end

function plot_losses(losses)
    p = plot(x=collect(1:length(losses)), y=losses, Geom.line,
     Guide.xlabel("#episodes"), Guide.ylabel("Episodic Loss"))
    draw(SVG("losses.svg", 8inch, 4inch), p)
end

function plot_rewards(rewards)
    p = plot(x=collect(1:length(rewards)), y=rewards, Geom.line,
     Guide.xlabel("#episodes"), Guide.ylabel("Episodic Rewards"))
    draw(SVG("rewards.svg", 8inch, 4inch), p)
end


# ---------------------------- Parameters --------------------------------------
const ACTION_SIZE = length(env.action_space)
const FRAMESKIP = 1
const γ = 0.98f0 # Discount


const FINAL_ϵ = 2f-2
const INITIAL_ϵ = 1f0
const EXPLORE = 100000
const OBSERVE = 10000

const BATCH_SIZE = 32
const UPDATE_TIME = 1000

const MEMORY_SIZE = 100000

# ADAM parameters
const η = 0.0001f0

current_state = Array{Float32, 4}(undef, 84, 84, 4, 1)
memory = MemoryArray(MEMORY_SIZE)
ϵ = INITIAL_ϵ

# ------------------------- Model Architecture ---------------------------------
# We define our model and load it onto the GPU
training_model = Chain(Conv((8, 8), 4=>32, relu; stride=(4, 4)),
                       Conv((4, 4), 32=>64, relu; stride=(2, 2)),
                       Conv((3, 3), 64=>64, relu; stride=(1, 1)),
                       x -> reshape(x, :, size(x, 4)),
                       Dense(3136, 512, relu),
                       Dense(512, ACTION_SIZE)) |> gpu

# Same goes for our target_model
target_model = deepcopy(training_model) |> gpu

# We have to define our own loss here, because using the Base.:^ operator
# gets us into trouble during the gradient step...
function mse_loss(x, y)
    diff = x .- y
    return sum(diff .* diff)/length(diff)
end


opt = Optimise.ADAM(η)

# Vectorize that shit, bro...
function calculate_target_Q(reward_batch, terminal_batch, next_state_batch)
    training_model_actions = Flux.onecold(cpu(training_model(next_state_batch)))
    target_model_values = target_model(next_state_batch)[CartesianIndex.(training_model_actions, eachindex(training_model_actions))]
    reward_batch .+ γ .* target_model_values .* (1 .- terminal_batch)
end
# ------------------------- Helper functions -----------------------------------

function preprocess(observation)
    observation = imresize(observation, (110, 84))
    observation = observation[19:102, :]
    observation[observation .== 0.41960785f0] .= 0
    observation[observation .== 0.34117648f0] .= 0
    observation[observation .!= 0] .= 1
    return reshape(observation, size(observation)..., 1, 1)
end

function update_ϵ!()
    global ϵ
    if ϵ > FINAL_ϵ && env.total_steps > OBSERVE
        ϵ -= (INITIAL_ϵ - FINAL_ϵ)/EXPLORE
    end
end

function getaction!(state)
    # We take action after some frames to simulate human reaction time.
    # Following ϵ-greedy policy, we see if a randomly generated number is greater
    # than ϵ. If greater, we act greedily. Otherwise, we pick a random action.
    update_ϵ!()
    return env.total_steps % FRAMESKIP == 0 ? rand(Float32) ≤ ϵ ? rand(1:ACTION_SIZE) : Flux.onecold(cpu(training_model(state |> gpu)))[1] : 1
end

function init!()
    global current_state
    observation, reward, terminal, _ = step!(env, 1) # Do nothing
    current_state = observation |> preprocess |> (obs) -> cat(obs, obs, obs, obs, dims=3)
    return
end

function episode!(env; debug=false)
    global current_state
    reset!(env)
    loss = 0f0
    while !Gym.game_over(env)
        action = getaction!(current_state)
        next_state, reward, terminal, _ = step!(env, action)
        new_state = cat(current_state[:, :, 2:end, :], preprocess(next_state), dims=3)

        # Push the data objects into memory for exp replay...
        push!(memory, current_state, action, reward, new_state, terminal)

        # First we populate the memory with experiences, then we sample from
        # those experiences to train our model...
        if env.total_steps > OBSERVE && !debug
            loss += trainnetwork!()
        end
        current_state .= new_state
    end
    return env.total_reward, loss
end

function trainnetwork!()
    current_state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = sample(memory, BATCH_SIZE) |> gpu

    # Calculate the target network's Q values...
    target_Q_values_batch = calculate_target_Q(reward_batch, terminal_batch, next_state_batch)

    # Calculate the training network's Q values...
    model_output = training_model(current_state_batch)
    action_batch = Int.(action_batch)
    training_Q_values_batch = model_output[CartesianIndex.(action_batch, eachindex(action_batch))]

    # Calculate loss
    loss = mse_loss(training_Q_values_batch, target_Q_values_batch)

    # Get the gradients of the parameters of the training network wrt to the
    # calculated loss
    grads = Tracker.gradient(() -> loss, params(training_model))

    # Work it harder
    # Make it better
    # Do it faster
    # Makes us stronger
    Flux.Optimise.update!(opt, params(training_model), grads)

    # Fitted Q Iteration and saving model...
    if env.total_steps % UPDATE_TIME == 0
        target_model = deepcopy(training_model)
        save_model(training_model, "")
    end

    return loss.data
end

function mainloop(num_episodes)
    io = open("log.txt", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)

    losses = Float32[]
    rewards = Float32[]
    reset!(env);
    init!()

    @info("Starting training for $num_episodes...")
    println("Starting training for $num_episodes...")
    for ctr=1:num_episodes
        reward, loss = episode!(env)

        @info("episode $ctr: Loss = $loss  |  Reward = $reward")
        println("episode $ctr: Loss = $loss  |  Reward = $reward")
        Base.push!(rewards, reward)
        Base.push!(losses, loss)

        if ctr % 20 == 0
            save_model(training_model, "episode #$ctr")
        end

        plot_losses(losses)
        plot_rewards(rewards)
        flush(io)
    end
    @info("Training ended successfully after $num_episodes")
    println("Training ended successfully after $num_episodes")
    close(io)
end

mainloop(50)

#=
function dummy()
    reset!(env); init!();
    episode!(env, debug=true);
    episode!(env; debug=true);
    episode!(env; debug=true);

    #=
    for ctr=1:7
        total_reward, loss = episode!(env)
        println("$ctr : $loss & $total_reward")

    end=#
end
dummy()
=#

#A = [ -1.6111217f0, 0.36582416f0, -0.38699928f0, 1.5262374f0, 0.9370651f0,-1.1155703f0]
#A = reshape(cu(A), length(A), 1)
