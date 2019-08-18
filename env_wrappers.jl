import Gym: reset!, step!, game_over

struct NoopResetEnv
    env
    noop_max
    noop_action

    """
    Sample initial states by taking a random number on no-ops on reset.
    """
    NoopResetEnv(env, noop_max=30) = new(env, noop_max, 1)
end

step!(env::NoopResetEnv, action) = step!(env.env, action)

function reset!(env::NoopResetEnv)
    reset!(env.env)
    noops = rand(1:env.noop_max)
    obs = nothing
    for _=1:noops
        obs, _, done, _  = step!(env.env, env.noop_action)
        if done
            obs = reset!(env)
        end
    end
    return obs
end


struct FireResetEnv
    env

    """
    Take action on reset for environments that are fixed until firing.
    """
    FireResetEnv(env) = new(env)
end

function reset!(env::FireResetEnv)
    reset!(env.env)
    obs, _, done, _ = step!(env.env, 2)
    if done
        reset!(env.env)
    end
    obs, _, done, _ = step!(env.env, 3)
    if done
        reset!(env.env)
    end
    return obs
end

step!(env::FireResetEnv, action) = step!(env.env, action)


struct MaxAndSkipEnv
    env
    obs_buffer::Array{Float32, 3}
    skip

    """
    Return only every `skip`-th frame
    """
    MaxAndSkipEnv(env, skip=4) = new(env, zeros(Float32, 210, 160, 2), skip)
end

reset!(env::MaxAndSkipEnv) = reset!(env.env)

function step!(env::MaxAndSkipEnv, action)
    total_reward = 0f0
    done = nothing
    info = nothing
    for i=1:env.skip
        obs, reward, done, info = step!(env.env, action)
        if i == env.skip - 1
            env.obs_buffer[:, :, 1:1] .= obs
        end
        if i == env.skip
            env.obs_buffer[:, :, 2:2] .= obs
        end
        total_reward += reward
        done && break
    end
    max_frame = maximum(env.obs_buffer, dims=3)[:, :, 1]
    return max_frame, total_reward, done, info
end

function Base.getproperty(env::Union{FireResetEnv, MaxAndSkipEnv, NoopResetEnv}, sym::Symbol)
    if sym == :done
        return env.env.done
    elseif sym == :total_steps
        return env.total_steps
    elseif sym == :total_reward
        return env.env.total_reward
    else
        return Base.getfield(env, sym)
    end
end

Gym.game_over(env::FireResetEnv) = env.env.done
