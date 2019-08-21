# DDQN implementation in Julia [WIP]

To be honest, this doesn't work yet. But I'm trying to make that happen.

Trying it out on `Pong-greyNoFrameskip-v4`, which is just a greyscale version of `PongNoFrameskip-v4`.
I should also let you know that unlike the DeepMind paper, in which a state comprises of a stack of the 
4 latest frames, my implementation uses only 1 frame, which is every 4th frame. It gave me similar results 
and sped up training so I decided to stick with it.

To train run `julia main2.jl`

**UPDATE** - Ok, so this still doesn't work, but it doesn't \***not**\* work as much as
it used to. The model seems to learn for upto 20-30 episodes and you can see it go
from scoring -21 to then -20 and -19, and even -18 if you're really lucky. But then it,
just forgets everything it has learned and you see mile long streaks of -21 rewards.
It ends up performing worse than a random agent. Now the bug can be in three places...

1) Environment code
2) RL agent code
3) Hyperparameters

I'm ruling out 1 and 2, because A) I've double and triple checked my code numerous times
and B) if there was a bug in the implementation of either the environment code or
RL agent code, we shouldn't have been able to see the initial progress that we do. But then again,
never say never...

The Hyperparameter case is also tricky. Almost all implementations use similar hyperparameters
with the only differences in `η` used in ADAM optimizer, `γ` and the way `ϵ` is calculated.
Those metrics are also very similar.

I have included the results of a test run of 80 episodes in the `extra` folder.

Everything is ok, if you always have a Plan B up your sleeve, incase Plan A doesn't pan out. So here's
my Plan B... I will take my understanding of the working of DDQN algorithm, and reimplement it in PyTorch.
If I still get similar results, well then, I'll have to take closer look at my agent code. If not, then I will take a closer
look at my enviroment code and investigate if the problem lies with Flux.
