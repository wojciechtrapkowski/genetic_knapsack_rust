use plotters::prelude::*;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Item {
    name: String,
    weight: f64,
    value: f64,
}

type Individual = Vec<bool>;
type Population = Vec<Individual>;

fn main() {
    // Define items
    let items = vec![
        Item {
            name: "Item 1".to_string(),
            weight: 15.0,
            value: 30.0,
        },
        Item {
            name: "Item 2".to_string(),
            weight: 10.0,
            value: 25.0,
        },
        Item {
            name: "Item 3".to_string(),
            weight: 20.0,
            value: 40.0,
        },
        Item {
            name: "Item 4".to_string(),
            weight: 25.0,
            value: 50.0,
        },
        Item {
            name: "Item 5".to_string(),
            weight: 30.0,
            value: 60.0,
        },
        Item {
            name: "Item 6".to_string(),
            weight: 18.0,
            value: 36.0,
        },
        Item {
            name: "Item 7".to_string(),
            weight: 12.0,
            value: 24.0,
        },
        Item {
            name: "Item 8".to_string(),
            weight: 22.0,
            value: 45.0,
        },
        Item {
            name: "Item 9".to_string(),
            weight: 8.0,
            value: 16.0,
        },
        Item {
            name: "Item 10".to_string(),
            weight: 11.0,
            value: 22.0,
        },
        Item {
            name: "Item 11".to_string(),
            weight: 16.0,
            value: 32.0,
        },
        Item {
            name: "Item 12".to_string(),
            weight: 21.0,
            value: 42.0,
        },
        Item {
            name: "Item 13".to_string(),
            weight: 23.0,
            value: 46.0,
        },
        Item {
            name: "Item 14".to_string(),
            weight: 33.0,
            value: 67.0,
        },
        Item {
            name: "Item 15".to_string(),
            weight: 13.0,
            value: 26.0,
        },
        Item {
            name: "Item 16".to_string(),
            weight: 19.0,
            value: 38.0,
        },
        Item {
            name: "Item 17".to_string(),
            weight: 14.0,
            value: 28.0,
        },
        Item {
            name: "Item 18".to_string(),
            weight: 7.0,
            value: 14.0,
        },
        Item {
            name: "Item 19".to_string(),
            weight: 26.0,
            value: 52.0,
        },
        Item {
            name: "Item 20".to_string(),
            weight: 29.0,
            value: 58.0,
        },
        Item {
            name: "Item 21".to_string(),
            weight: 17.0,
            value: 35.0,
        },
        Item {
            name: "Item 22".to_string(),
            weight: 27.0,
            value: 54.0,
        },
        Item {
            name: "Item 23".to_string(),
            weight: 9.0,
            value: 19.0,
        },
        Item {
            name: "Item 24".to_string(),
            weight: 24.0,
            value: 48.0,
        },
        Item {
            name: "Item 25".to_string(),
            weight: 31.0,
            value: 62.0,
        },
        Item {
            name: "Item 26".to_string(),
            weight: 5.0,
            value: 10.0,
        },
        Item {
            name: "Item 27".to_string(),
            weight: 6.0,
            value: 13.0,
        },
        Item {
            name: "Item 28".to_string(),
            weight: 28.0,
            value: 57.0,
        },
        Item {
            name: "Item 29".to_string(),
            weight: 32.0,
            value: 64.0,
        },
        Item {
            name: "Item 30".to_string(),
            weight: 39.0,
            value: 78.0,
        },
    ];

    let knapsack_max_capacity = 1000.0;
    let population_size = 100;
    let generations = 100;
    let n_selection = 30; // How many parents are chosen
    let n_elite = 20;

    println!("Items: {:?}", items);
    println!("Knapsack capacity: {}", knapsack_max_capacity);

    let start_time = Instant::now();
    let (best_solution, best_fitness, history) = genetic_algorithm(
        &items,
        knapsack_max_capacity,
        population_size,
        generations,
        n_selection,
        n_elite,
    );
    let duration = start_time.elapsed();

    print_solution(&items, &best_solution, best_fitness);
    println!("Time elapsed: {:?}", duration);

    plot_history(&history, "fitness_history.png").expect("Plotting failed");
}

fn genetic_algorithm(
    items: &[Item],
    knapsack_max_capacity: f64,
    population_size: usize,
    generations: usize,
    n_selection: usize,
    n_elite: usize,
) -> (Individual, f64, Vec<f64>) {
    let mut rng = rand::rng();
    let mut population = initial_population(items.len(), population_size, &mut rng);

    let mut best_solution = population[0].clone();
    let mut best_fitness = fitness(items, knapsack_max_capacity, &best_solution);
    let mut best_history = Vec::with_capacity(generations);

    for _ in 0..generations {
        let (current_best, current_fit) =
            population_best(items, knapsack_max_capacity, &population);

        if current_fit > best_fitness {
            best_solution = current_best.clone();
            best_fitness = current_fit;
        }
        best_history.push(best_fitness);

        let (parents, elites) = select_parents_and_elites(
            items,
            knapsack_max_capacity,
            &population,
            n_selection,
            n_elite,
        );

        let mut children = generate_children(&parents, items.len());
        mutate(&mut children, &mut rng);
        population = [children, elites].concat();
    }

    (best_solution, best_fitness, best_history)
}

fn plot_history(history: &[f64], file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file, (800, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Best Fitness per Generation", ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..history.len(), 0f64..max_val)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Best Fitness")
        .draw()?;

    chart.draw_series(LineSeries::new(
        history.iter().enumerate().map(|(g, v)| (g, *v)),
        &BLUE,
    ))?;

    Ok(())
}

fn initial_population(
    individual_size: usize,
    population_size: usize,
    rng: &mut ThreadRng,
) -> Population {
    (0..population_size)
        .map(|_| (0..individual_size).map(|_| rng.random_bool(0.5)).collect())
        .collect()
}

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

fn population_best(
    items: &[Item],
    knapsack_max_capacity: f64,
    population: &Population,
) -> (Individual, f64) {
    let mut best_individual = population[0].clone();
    let mut best_individual_fitness = fitness(items, knapsack_max_capacity, &best_individual);

    for individual in population {
        let individual_fitness = fitness(items, knapsack_max_capacity, individual);
        if individual_fitness > best_individual_fitness {
            best_individual = individual.clone();
            best_individual_fitness = individual_fitness;
        }
    }

    (best_individual, best_individual_fitness)
}

fn select_parents_and_elites(
    items: &[Item],
    knapsack_max_capacity: f64,
    population: &Population,
    n_selection: usize,
    n_elite: usize,
) -> (Population, Population) {
    let mut rng = rand::rng();

    // Calculate fitness for all individuals
    let fitnesses: Vec<f64> = population
        .iter()
        .map(|individual| fitness(items, knapsack_max_capacity, individual))
        .collect();

    // Calculate probabilities for selection
    let sum_fitness: f64 = fitnesses.iter().sum();

    // Handle edge case where all individuals have zero fitness
    if sum_fitness == 0.0 {
        // Randomly select parents and elites
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.shuffle(&mut rng);

        let parents: Population = indices
            .iter()
            .take(n_selection)
            .map(|&i| population[i].clone())
            .collect();

        let elites: Population = indices
            .iter()
            .take(n_elite)
            .map(|&i| population[i].clone())
            .collect();

        return (parents, elites);
    }

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

    // Find elites (top n_elite individuals)
    let mut population_with_fitness: Vec<(usize, f64)> =
        fitnesses.iter().enumerate().map(|(i, &f)| (i, f)).collect();

    population_with_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let elites: Population = population_with_fitness
        .iter()
        .take(n_elite)
        .map(|&(i, _)| population[i].clone())
        .collect();

    (parents, elites)
}

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

fn mutate(population: &mut Population, rng: &mut ThreadRng) {
    for individual in population.iter_mut() {
        let mutation_index = rng.random_range(0..individual.len());
        individual[mutation_index] = !individual[mutation_index];
    }
}

fn print_solution(items: &[Item], solution: &Individual, fitness_value: f64) {
    println!("Best solution value: {}", fitness_value);
    println!("Selected items:");

    let mut total_weight = 0.0;
    let mut total_value = 0.0;

    for (i, &included) in solution.iter().enumerate() {
        if included {
            println!(
                "  {} (Weight: {}, Value: {})",
                items[i].name, items[i].weight, items[i].value
            );
            total_weight += items[i].weight;
            total_value += items[i].value;
        }
    }

    println!("Total weight: {}", total_weight);
    println!("Total value: {}", total_value);
}
