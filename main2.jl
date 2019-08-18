# Uncomment the following line if you have access to a GPU
#using CuArrays
using Gym
using Logging

# ---------------------------- Hyperparameters ---------------------------------

const ACTION_SIZE = 6
const γ = 0.99f0 # Discount


const FINAL_ϵ = 0.01f0
const INITIAL_ϵ = 1f0
const ϵ_DECAY = 100000
const OBSERVE = 10000

const BATCH_SIZE = 32
const UPDATE_TIME = 1000

const MEMORY_SIZE = 100000

# ADAM parameters
const η = 0.0001f0


const NUM_EPISODES = 80

#-------------------------------------------------------------------------------

# Bringing everything together...
include("env_wrappers.jl")
include("memory_arr.jl")
include("utils.jl")
include("model_funcs.jl")
include("episode_funcs.jl")

#-------------------------- Env setup ------------------------------------------

env = nothing
# This is something we do to make an environment run on servers.
try
    global env
    env = make("Pong-greyNoFrameskip-v4")
catch
    global env
    env = make("Pong-greyNoFrameskip-v4")
end
env = FireResetEnv(MaxAndSkipEnv(NoopResetEnv(env, 30), 4))


# ------------------------------------------------------------------------------
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


mainloop(NUM_EPISODES)
