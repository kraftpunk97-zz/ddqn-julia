import Base: push!
using Images


frame_idx = 0
current_state = Array{Float32, 4}(undef, 84, 84, 1, 1)
memory = MemoryArray(MEMORY_SIZE)


function preprocess(observation)
    observation = imresize(observation, (84, 84))
    #observation = observation[19:102, :]
    observation[observation .== 0.41960785f0] .= 0
    observation[observation .== 0.34117648f0] .= 0
    #observation[observation .!= 0] .= 1
    return reshape(observation, size(observation)..., 1, 1)
end

function init!()
    global current_state
    memory.current_ptr = 1
    observation, reward, terminal, _ = step!(env, 1) # Do nothing
    current_state = observation |> preprocess# |> (obs) -> cat(obs, obs, obs, obs, dims=3)
    return
end

@inline get_ϵ() = INITIAL_ϵ - (INITIAL_ϵ - FINAL_ϵ) * frame_idx / ϵ_DECAY

function getaction!(state)
    # Following ϵ-greedy policy, we see if a randomly generated number is greater
    # than ϵ. If greater, we act greedily. Otherwise, we pick a random action.
    # If wondering why we use offload the model from the GPU when using `Flux.onecold`,
    # refer to https://github.com/JuliaGPU/CuArrays.jl/issues/304
    ϵ = get_ϵ()
    return rand(Float32) ≤ ϵ ? rand(1:ACTION_SIZE) : Flux.onecold(cpu(training_model(state |> gpu)))[1]
end

function episode!(env; debug=false)
    global current_state, frame_idx
    reset!(env)
    episode_reward = 0
    loss = 0
    while !Gym.game_over(env)
        action = getaction!(current_state)
        next_state, reward, terminal, _ = step!(env, action)
        #new_state = cat(current_state[:, :, 2:end, :], preprocess(next_state), dims=3)
        new_state = preprocess(next_state)

        # Push the data objects into memory for exp replay...
        push!(memory, current_state, action, reward, new_state, terminal)

        # First we populate the memory with experiences, then we sample from
        # those experiences to train our model.
        if !debug && memory.current_ptr > OBSERVE
            loss += trainnetwork!()
        end

        current_state .= new_state
        frame_idx += 1
    end
    return env.total_reward, loss
end
