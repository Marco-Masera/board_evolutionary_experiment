I want to apply some evolutionary algorithm to a neural network. So I will have a population of networks, and perform some sort of crossover and mutation.
Consider that:
* I cannot afford to keep a large population, for memory limitations. And of course the network has many parameters.
* I want the system to be sligthly caotic. Ideally it should improve over time but never reach stability, keep changing forever. Maybe this can be reached by always keeping at least part of the population random, or applying a high mutation rate to part of the population.

Which evolutionary algorithm would you apply?
ChatGPT ha detto:

Given your constraints—tiny population due to memory limits, very high-dimensional individuals (neural network weights), desire for perpetual instability / chaos rather than strict convergence—I’d suggest looking at approaches that don’t rely on standard genetic crossover of full weight vectors, because traditional GA on raw weights behaves poorly in high-dim space and requires large populations to work well.

Instead, consider one of these two families:

✅ Best Fit: Evolution Strategies (ES), specifically (μ + λ) / (1 + λ) strategies or CMA-ES variants with very small μ

Instead of crossover on full weights, ES treats mutation as the main operator and often works surprisingly well even with small populations.

You can run with μ = 1 (one parent) and λ = 4–8 mutated children.

Always keep the best, or do pure 1→λ where the parent is replaced no matter what.

To enforce chaos / non-convergence, occasionally inject random individuals or reset mutation variance.

If memory is tight, don’t store full networks—store only RNG seeds and reconstruct weights on the fly (this is how OpenAI ES works).

This gives you continuous drift with gradual improvement, especially if you occasionally sabotage progress with deliberate noise bursts.

🔁 Alternative: Neuroevolution with Weight Perturbation only (No crossover at all)

Genetic crossover doesn't work well on neural networks unless you're neuroevolving architectures or modular/hypernetwork weights. For fixed topology, just do mutation-only evolution:

population = [best_network, ...some random ones...]

for generation:
    offspring = mutate(best_network)   # add Gaussian noise to weights
    maybe inject a random network to maintain entropy
    select best few


This is extremely memory-light. If you want to stay in "chaotic-but-improving" territory:

Set mutation variance to oscillate (e.g. sine wave over time).

Occasionally flip optimization direction (reward hacking / anti-objective pass).

Maintain one or two completely random individuals forever to ensure fresh blood.

🚫 What I wouldn't recommend
Method	Why Not
Standard Genetic Algorithm / Crossover on raw weights	Assumes meaningful recombination. Weight-space crossover usually destroys learned structure.
NEAT / topology-evolving algorithms	Too memory-heavy and mostly useful for small NNs only.
Full CMA-ES	Needs covariance matrix = O(N²) memory, bad for large nets.
🧪 If You Want True Eternal Chaos

Use a "Never Converge ES" loop:

Track two elites: best performer and most novel (maximally different) performer.

Alternate between optimizing for reward and novelty.

Occasionally delete the best network on purpose and let the ecosystem regrow from noise.

This essentially makes your training process more like artificial life than optimization.

Final Recommendation

Start with:

(1 + λ) Evolution Strategy with Gaussian mutation on weights, λ = 4–8, and forced entropy via periodic random resets or novelty pressure.

Simple, memory-cheap, chaotic, and surprisingly powerful.