# Genetic Algorithm for the 0/1 Knapsack Problem

This project implements a genetic algorithm to solve the classic 0/1 knapsack problem in Rust. The algorithm uses evolutionary techniques to find near-optimal solutions for item selection with weight constraints.

## Problem Description

The knapsack problem is a classic optimization challenge:

- You have a set of items, each with a weight and a value
- Your knapsack has a maximum weight capacity
- The goal is to select items to maximize the total value while staying within the weight limit
- In the 0/1 variant, you either take an entire item (1) or leave it (0)

## Implementation Details

### Individual Representation

- Each potential solution (individual) is represented as a binary vector
- Each bit represents whether a specific item is included (1) or excluded (0)
- The length of each individual equals the number of items available

### Algorithm Components

#### Initialization

The algorithm begins with randomly generated individuals where each item has a 50% chance of being selected.

```rust
fn initial_population(
    individual_size: usize,
    population_size: usize,
    rng: &mut ThreadRng,
) -> Population {
    (0..population_size)
        .map(|_| (0..individual_size).map(|_| rng.random_bool(0.5)).collect())
        .collect()
}
```

#### Fitness Function

The fitness function evaluates how good a solution is:
- Calculate the total weight and value of selected items
- If the weight exceeds the knapsack capacity, the fitness is 0
- Otherwise, fitness equals the total value

```rust
fn fitness(items: &[Item], knapsack_max_capacity: f64, individual: &Individual) -> f64 {
    let mut total_weight = 0.0;
    let mut total_value = 0.0;

    for (i, &included) in individual.iter().enumerate() {
        if included {
            total_weight += items[i].weight;
            total_value += items[i].value;
        }
    }

    if total_weight > knapsack_max_capacity {
        return 0.0;
    }

    total_value
}
```

#### Selection (Parent Selection)

The implementation uses roulette wheel selection (fitness-proportionate selection):
- Parents are selected with probability proportional to their fitness
- This gives better solutions a higher chance of reproducing
- Includes a fallback for cases where all individuals have zero fitness

```rust
// Select parents with weighted probability based on fitness
let mut parents = Vec::with_capacity(n_selection);
for _ in 0..n_selection {
    // Select an individual with probability proportional to fitness
    let mut r = rng.random_range(0.0..sum_fitness);
    for (i, fitness_value) in fitnesses.iter().enumerate() {
        r -= fitness_value;
        if r <= 0.0 {
            parents.push(population[i].clone());
            break;
        }
        // Fallback in case of floating point errors
        if i == fitnesses.len() - 1 {
            parents.push(population[i].clone());
        }
    }
}
```

#### Elitism

To preserve good solutions, the algorithm implements elitism:
- The top `n_elite` individuals are preserved intact for the next generation
- This ensures the best solutions are not lost during evolution

```rust
// Find elites (top n_elite individuals)
let mut population_with_fitness: Vec<(usize, f64)> =
    fitnesses.iter().enumerate().map(|(i, &f)| (i, f)).collect();

population_with_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

let elites: Population = population_with_fitness
    .iter()
    .take(n_elite)
    .map(|&(i, _)| population[i].clone())
    .collect();
```

#### Crossover

The implementation uses single-point crossover at the midpoint:
- Parent pairs produce two children by exchanging genetic material
- The first half of parent 1 combines with the second half of parent 2 (and vice versa)

```rust
fn generate_children(parents: &Population, item_count: usize) -> Population {
    let mut children = Vec::new();
    let half_parents = parents.len() / 2;
    let item_midpoint = item_count / 2;

    for i in 0..half_parents {
        let parent1 = &parents[i];
        let parent2 = &parents[i + half_parents];

        // First child: first half from parent1, second half from parent2
        let mut child1 = Vec::with_capacity(item_count);
        child1.extend_from_slice(&parent1[0..item_midpoint]);
        child1.extend_from_slice(&parent2[item_midpoint..]);
        children.push(child1);

        // Second child: first half from parent2, second half from parent1
        let mut child2 = Vec::with_capacity(item_count);
        child2.extend_from_slice(&parent2[0..item_midpoint]);
        child2.extend_from_slice(&parent1[item_midpoint..]);
        children.push(child2);
    }

    children
}
```

#### Mutation

To maintain genetic diversity, the algorithm implements single-point mutation:
- For each individual, randomly flip one bit
- This allows exploring new combinations that might not be reached through crossover alone

```rust
fn mutate(population: &mut Population, rng: &mut ThreadRng) {
    for individual in population.iter_mut() {
        let mutation_index = rng.random_range(0..individual.len());
        individual[mutation_index] = !individual[mutation_index];
    }
}
```

### Evolution Process

1. Create a random initial population
2. For each generation:
   - Evaluate fitness of all individuals
   - Select parents using roulette wheel selection
   - Identify elite individuals to preserve
   - Create offspring through crossover
   - Apply mutations to offspring
   - Form the new population from offspring and elites
3. Track the best solution found across all generations

## Visualization

The project includes functionality to plot the evolution of the best fitness value over generations, helping visualize the algorithm's convergence.

## Possible Tweaks and Enhancements

The genetic algorithm implementation can be customized and improved in various ways to better address the knapsack problem. Here are several modifications worth exploring:

### Parameter Tuning

Fine-tuning these parameters can significantly impact algorithm performance:

- **Population Size**: Larger populations increase diversity but require more computational resources. Try values between 50-500.
- **Number of Generations**: More generations allow for further evolution but increase runtime. Experiment with 100-1000 generations.
- **Selection Size**: Adjusting how many parents are selected affects selection pressure. Start with 20-40% of population size.
- **Elite Count**: Higher values preserve more good solutions but may reduce diversity. Try keeping 10-20% of the population as elites.
- **Mutation Rate**: Currently fixed at one mutation per individual. Consider making this a probability per bit (e.g., 1-5%).

### Alternative Selection Methods

- **Tournament Selection**: Randomly select k individuals (typically 2-7) and choose the best one as a parent. This approach offers controllable selection pressure through tournament size and works well even when fitness values vary widely.

- **Rank Selection**: Assign selection probability based on the rank of individuals rather than their actual fitness values. This reduces selection pressure when fitness differences are extreme and helps maintain diversity.

- **Elitism Selection**: Select only the best inidivuals in the population. With this approach we can often fall into a local best solution and never be diverse enough to find global bests.

### Alternative Crossover Methods

- **Two-Point Crossover**: Select two random points and exchange the segment between them. This preserves more building blocks than single-point crossover.

- **Uniform Crossover**: For each bit position, randomly select which parent contributes its bit to the child. This allows for more thorough mixing of genetic material.

- **Ordered Crossover**: Particularly useful if the order of items becomes important in more complex variants of the knapsack problem.

- **Adaptive Crossover**: Dynamically adjust crossover points or probabilities based on the fitness of solutions.

- **Arithmetic Crossover**: Apply some kind of mathematical operations on parents, the results will be new children.

### Alternative Mutation Methods

- **Bit-Flip Probability**: Instead of always flipping one bit, assign a probability (e.g., 1/n where n is the number of items) of flipping each bit.

- **Adaptive Mutation**: Increase mutation rates when the population converges to avoid premature convergence, decrease when diversity is high.

- **Swap Mutation**: Exchange the values of two randomly selected positions, maintaining the same number of selected items.

- **Inversion Mutation**: Select two points and reverse the sequence of bits between them.

## Usage

To run the algorithm:

```bash
cargo run
```

This will execute the genetic algorithm on the predefined set of items and display the best solution found, including selected items, total weight, and total value.