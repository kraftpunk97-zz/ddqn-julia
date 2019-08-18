using BSON: @save
using Gadfly

function save_model(model, info)
    wts = cpu.(Tracker.data(params(model)))

    @save "ddqn_$info.bson" wts
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
