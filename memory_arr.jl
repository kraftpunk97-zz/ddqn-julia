# Defining our own memory buffer because it may help in vectorizing and thus
# speeding up the training process.
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
    current_state_arr = zeros(Float32, 84, 84, 1, size)
    action_arr = zeros(Int64, size)
    reward_arr = zeros(Float32, size)
    next_state_arr = zeros(Float32, 84, 84, 1, size)
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
        println("Memory full now! Overwriting...")
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
