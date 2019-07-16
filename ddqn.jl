#=TODO: Go thru each function to test if it works as intended. Try to learn about
        the internals of `Flux.train!`. We may have to write our own training loop.
=#

using CuArrays
using Flux
using Flux.Optimise
using Gym
using Images
using DataStructures: CircularBuffer
using Distributions: sample
using Statistics: mean
ENV["CUDA_VISIBLE_DEVICES"] = 4


# ------------------------ Load game environment -------------------------------
env = make("Pong-greyNoFrameskip-v0")

# ---------------------------- Parameters --------------------------------------
const ACTION_SIZE = length(env.action_space)
const FRAMESKIP = 1
const γ = 0.95f0 # Discount


const FINAL_ϵ = 1f-1
const INITIAL_ϵ = 1f0
const OBSERVE = 10000
const EXPLORE = 1000000
const BATCH_SIZE = 32
const UPDATE_TIME = 1000

const MEMORY_SIZE = 100000

# RMSProp parameters
const η = 25f-5
const ρ = 99f-2

current_state = Array{Float32, 4}(undef, 84, 84, 4, 1)
memory = CircularBuffer{Any}(MEMORY_SIZE)
ϵ = INITIAL_ϵ

# ------------------------- Model Architecture ---------------------------------
training_model = Chain(Conv((8, 8), 4=>32, relu; stride=(4, 4)),
                       Conv((4, 4), 32=>64, relu; stride=(2, 2)),
                       Conv((3, 3), 64=>64, relu; stride=(1, 1)),
                       x -> reshape(x, :, size(x, 4)),
                       Dense(3136, 512, relu),
                       Dense(512, ACTION_SIZE)) |> gpu

target_model = deepcopy(training_model) |> gpu
huber_loss(x, y) = mean(sqrt.(1 .+ (x.-y).^2) .- 1)

opt = Optimise.RMSProp(η, ρ)

# ------------------------- Helper functions -----------------------------------

function preprocess(observation)
    observation = imresize(observation, (110, 84))
    observation = observation[17:100, :]
    observation[observation .== 0.41960785f0] .= 0
    observation[observation .== 0.34117648f0] .= 0
    observation[observation .!= 0] .= 1
    return reshape(observation, size(observation)..., 1, 1)
end

function update_ϵ!()
    global ϵ, INITIAL_ϵ, FINAL_ϵ, OBSERVE, EXPLORE
    ϵ > FINAL_ϵ && env.total_steps > OBSERVE &&
        (ϵ -= (INITIAL_ϵ - FINAL_ϵ)/EXPLORE)
end

function getaction!(state)
    # We take action after some frames to simulate human reaction time.
    # Following ϵ-greedy policy, we see if a randomly generated number is greater
    # than ϵ. If greater, we act greedily. Otherwise, we pick a random action.
    global FRAMESKIP
    update_ϵ!()
    return env.total_steps % FRAMESKIP == 0 ? rand(Float32) <= ϵ ? rand(1:ACTION_SIZE) : Flux.argmax(training_model(gpu(state)))[1] : 1
end

function init!()
    global current_state
    observation, reward, terminal, _ = step!(env, 1) # Do nothing
    current_state = observation |> preprocess |> (obs) -> cat(obs, obs, obs, obs, dims=3) |> gpu
end

function episode!()
    global current_state
    reset!(env)
    while !Gym.game_over(env)
        action = getaction!(current_state)
        next_state, reward, terminal, _ = step!(env, action)
        new_state = cat(current_state[:, :, 2:end, :], preprocess(next_state), dims=3) |> gpu
        push!(memory, (current_state, action, reward, new_state, terminal))

        env.total_steps > OBSERVE && trainnetwork!()
        current_state = new_state
    end
    return env.total_reward
end

#current_state_batch = nothing
#y_batch = nothing
function trainnetwork!()
    global UPDATE_TIME, BATCH_SIZE, γ, UPDATE_TIME, target_model, training_model
    #global current_state_batch, y_batch
    minibatch = sample(memory, BATCH_SIZE, replace = false)
    #=
    current_state_batch = [current_state for (current_state, _, _, _, _) ∈ minibatch]
    action_batch        = [action        for (_, action, _, _, _) ∈ minibatch]
    reward_batch        = [reward        for (_, _, reward, _, _) ∈ minibatch]
    next_state_batch    = [next_state    for (_, _, _, next_state, _) ∈ minibatch]
    terminal_batch      = [terminal      for (_, _, _, _, terminal) ∈ minibatch]
    =#
    current_Q_value_batch = []
    y_batch = []

    for (current_state, action, reward, next_state, terminal) ∈ minibatch
        push!(y_batch, terminal ? reward : reward + γ * maximum(target_model(gpu(next_state))))
        push!(current_Q_value_batch, training_model(current_state |> gpu)[action])
    end
    #=
    for (t, r, Q) ∈ collect(zip(terminal_batch, reward_batch, prev_Q_value_batch))
        push!(y_batch, t ? r : r + γ * maximum(Q))
    end
    =#
    #current_Q_value_batch = hcat(current_Q_value_batch...)
    #y_batch = hcat(y_batch...)


    Flux.train!(huber_loss, params(training_model), [(current_Q_value_batch, y_batch)], opt)

    if env.total_steps % UPDATE_TIME == 0
        target_model = deepcopy(training_model)
        println("Hit that swap button!")
    end
end

function mainloop()
    global ϵ, current_state, env
    scores = zeros(20)
    idx = 1
    reset!(env)
    init!()
    while true
        episode!()
        scores[((idx-1) % 20)+1] = env.total_reward
        avg_score = mean(scores)
        println("Episode: $idx | Score: $(env.total_reward) | steps: $(env.total_steps) | Avg Score: $avg_score")
        idx += 1
    end
end
