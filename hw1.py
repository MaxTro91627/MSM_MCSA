import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

# Test functions to test algorithm performance
# Sphere function: a simple quadratic function with a minimum in (0, 0,...,0)
def sphere_function(x):
    return sum(xi**2 for xi in x)

# Rastrigin function: a multimodal function with multiple local minima
def rastrigin_function(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Pressure vessel design: an engineering optimization challenge
def pressure_vessel_design(x):
    R, T = x[0], x[1] # R is the radius, T is the thickness
    return (0.6224 * R * T * (1 + T))  # Simplified version for testing

# Himmelblau function: a two-dimensional function with multiple local minima
def himmelblau_function(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Other test functions for performance analysis
# 23 benchmark functions from the paper
def f1(x):
    return sum(xi**2 for xi in x)

def f2(x):
    return sum(abs(xi) for xi in x) + np.prod([abs(xi) for xi in x])

def f3(x):
    return sum([(sum(x[:i+1]))**2 for i in range(len(x))])

def f4(x):
    return max(abs(xi) for xi in x)

def f5(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

def f6(x):
    return sum(xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x)

def f7(x):
    return -20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / len(x))) - np.exp(sum(np.cos(2 * np.pi * xi) for xi in x) / len(x)) + 20 + np.e

def f8(x):
    return sum(-xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def f9(x):
    return sum((xi**2) / 4000 - np.cos(xi / np.sqrt(i + 1)) + 1 for i, xi in enumerate(x))

def f10(x):
    return sum((xi - 1)**2 for xi in x)

def f11(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

def f12(x):
    return sum(abs(xi) + np.prod([abs(xi) for xi in x]) for xi in x)

def f13(x):
    return sum(abs(xi**3 - 2 * xi + 1) for xi in x)

def f14(x):
    return sum(np.sin(xi)**2 - 0.5 / ((1 + 0.001 * sum(xi**2 for xi in x))**2) for xi in x)

def f15(x):
    return sum(abs(xi) + np.prod([abs(xj) for xj in x]) for xi in x)

def f16(x):
    return sum((xi - 5)**2 for xi in x)

def f17(x):
    return sum(xi**4 for xi in x)

def f18(x):
    return max(abs(xi) for xi in x)

def f19(x):
    return sum(abs(xi)**0.5 for xi in x)

def f20(x):
    return sum(np.log(abs(xi) + 1) for xi in x)

def f21(x):
    return sum(np.sin(10 * xi) + xi**2 for xi in x)

def f22(x):
    return sum(np.abs(np.cos(xi)) for xi in x)

def f23(x):
    return sum(xi**2 - np.cos(2 * np.pi * xi) + 1 for xi in x)

# A list of test functions with names for ease of analysis
benchmark_functions = [
    (f1, "F1 (Sphere)"),
    (f2, "F2 (Schwefel 2.22)"),
    (f3, "F3 (Schwefel 1.2)"),
    (f4, "F4 (Schwefel 2.21)"),
    (f5, "F5 (Rosenbrock)"),
    (f6, "F6 (Rastrigin)"),
    (f7, "F7 (Ackley)"),
    (f8, "F8 (Schwefel)"),
    (f9, "F9 (Griewank)"),
    (f10, "F10 (Shifted Sphere)"),
    (f11, "F11 (Shifted Rosenbrock)"),
    (f12, "F12 (Schwefel Variant)"),
    (f13, "F13 (Custom Polynomial)"),
    (f14, "F14 (Levy)"),
    (f15, "F15 (Modified Schwefel)"),
    (f16, "F16 (Sphere Variant)"),
    (f17, "F17 (Quartic)"),
    (f18, "F18 (Maximum Sphere)"),
    (f19, "F19 (Root Sphere)"),
    (f20, "F20 (Logarithmic Sphere)"),
    (f21, "F21 (Sinusoidal Sphere)"),
    (f22, "F22 (Cosine Sphere)"),
    (f23, "F23 (Custom Rastrigin Variant)")
]

# MCSA Hybrid Algorithm Class
class MCSA:
    # Step 1: Set algorithm parameters such as population size (N), maximum iterations (iterMax), and dimension (d) inside the class constructor.
    def __init__(self, fitness_func, d, iterMax, N, lower_bound, upper_bound,
                 alpha=0.5, lambda_c=0.5, freq=0.1, c1=1.5, c2=1.5):
        # the main parameters of the algorithm

        """
        fitness_func: target function
        d: problem dimension
        iterMax: maximum number of iterations
        N: population size
        lb: lower limit of the search
        ub: upper limit of the search
        alpha: fractional derivative coefficient
        lambda_c: crossover probability
        freq: frequency of sinusoidal adjustment
        c1: weight of a personal best position
        c2: weight of the global best position
        """

        self.fitness_func = fitness_func 
        self.d = d
        self.iterMax = iterMax
        self.N = N 
        self.lb = np.array(lower_bound)
        self.ub = np.array(upper_bound)
        self.alpha = alpha
        self.lambda_c = lambda_c
        self.freq = freq
        self.c1 = c1
        self.c2 = c2

        # Step 2: Initialization of chameleon positions based on Eq. (2.1).
        self.positions = np.random.uniform(self.lb, self.ub, (N, d))
        # Step 2: Initialize velocities of the chameleons' tongues.
        self.velocities = np.zeros((N, d))
        self.personal_best_positions = np.copy(self.positions)
        # Step 3: Calculate the fitness values of each chameleon.
        self.personal_best_scores = np.array([fitness_func(pos) for pos in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

        self.best_fitness_history = []  # history of best positions

    # Step 6: For t > 4, update the velocity using Eq. (3.7).
    def fractional_order_update(self, velocity_history):
        t1, t2, t3, t4 = velocity_history
        return (self.alpha * t1 +
                0.5 * self.alpha * (1 - self.alpha) * t2 +
                (1 / 6) * self.alpha * (1 - self.alpha) * (2 - self.alpha) * t3 +
                (1 / 24) * self.alpha * (1 - self.alpha) * (2 - self.alpha) * (3 - self.alpha) * t4)

    # Step 4: Calculate the parameter p1 using Eq. (3.9).
    def sinusoidal_adjustment(self, iteration):
        return 0.5 * (np.sin(2 * np.pi * self.freq * iteration / self.iterMax) + 1)

    # Step 5: Pursuing prey with chameleon's eyes using crossover strategies to produce new positions based on Eq. (2.4) and Eq. (2.5).
    # Crossover-based learning function
    def crossover_based_learning(self, personal_best, global_best):
        new_position = np.copy(personal_best)
        for j in range(self.d):
            r1, r2 = np.random.rand(), np.random.rand()
            if np.random.rand() < self.lambda_c:
                c = random.uniform(-1, 1)
                new_position[j] = r1 * personal_best[j] + (1 - r1) * global_best[j] + c * (global_best[j] - personal_best[j])
            else:
                new_position[j] = r2 * personal_best[j] + (1 - r2) * global_best[j]
        return new_position

    def run(self):
        velocity_history = [np.copy(self.velocities) for _ in range(4)]
        # Step 9: Iterate until the termination condition (t < iterMax) is met.
        for t in range(self.iterMax):
            # Step 7: Recalculate the fitness values to determine individual best and global best positions.
            # Assessment of current positions
            current_scores = np.array([self.fitness_func(pos) for pos in self.positions])
            # Updating personal and global best positions
            for i in range(self.N):
                if current_scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_scores[i]
                    self.personal_best_positions[i] = np.copy(self.positions[i])
            if np.min(self.personal_best_scores) < self.global_best_score:
                self.global_best_score = np.min(self.personal_best_scores)
                self.global_best_position = np.copy(self.personal_best_positions[np.argmin(self.personal_best_scores)])

            # Saving the best value for analysis
            self.best_fitness_history.append(self.global_best_score)

            # Step 4: Update the position of the chameleon based on Eq. (2.2).
            # Sinusoidal parameter setting
            p1 = self.sinusoidal_adjustment(t)
            inertia_weight = (1 - t / self.iterMax) * np.sqrt(t / self.iterMax)

            # Updating particle positions and velocities
            for i in range(self.N):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                r_i = np.random.rand()

                if r_i >= 0.5:
                    self.positions[i] += p1 * (self.personal_best_positions[i] - self.global_best_position) * r2 + \
                                         (1 - p1) * (self.global_best_position - self.positions[i]) * r1
                else:
                    mu = 0.5 * np.exp(-0.1 * (t / self.iterMax))
                    self.positions[i] += mu * ((self.ub - self.lb) * r3 + self.lb) * np.sign(np.random.rand() - 0.5)

                # Step 6: For t ≤ 4, update the velocity of chameleons' tongues using Eq. (2.9).
                if t < 4:
                    self.velocities[i] = (inertia_weight * self.velocities[i] +
                                          self.c1 * (self.global_best_position - self.positions[i]) * r1 +
                                          self.c2 * (self.personal_best_positions[i] - self.positions[i]) * r2)
                else:
                    self.velocities[i] = self.fractional_order_update([vh[i] for vh in velocity_history]) + \
                                         self.c1 * (self.global_best_position - self.positions[i]) * r1 + \
                                         self.c2 * (self.personal_best_positions[i] - self.positions[i]) * r2

                # Limiting the movement of particles within an acceptable range
                self.positions[i] += (self.velocities[i] ** 2 - velocity_history[0][i] ** 2) / (2 * 2590 * (1 - np.exp(-np.log(t + 1))))
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            velocity_history = [np.copy(self.velocities)] + velocity_history[:-1]

        return self.global_best_position, self.global_best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCSA algorithm on various benchmark functions.")

    parser.add_argument("--dims", type=int, default=10, help="Number of dimensions.")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument("--population", type=int, default=30, help="Population size.")
    parser.add_argument("--lower_bound", type=float, default=-10.0, help="Lower bound of the search space.")
    parser.add_argument("--upper_bound", type=float, default=10.0, help="Upper bound of the search space.")

    args = parser.parse_args()

    # Test specific functions
    specific_functions = [
        (sphere_function, "Sphere Function"),
        (rastrigin_function, "Rastrigin Function"),
        (pressure_vessel_design, "Pressure Vessel Design"),
        (himmelblau_function, "Himmelblau Function")
    ]

    plt.figure(figsize=(10, 6))
    for func, name in specific_functions:
        lb = [args.lower_bound] * args.dims
        ub = [args.upper_bound] * args.dims
        dims = 2 if func == pressure_vessel_design or func == himmelblau_function else args.dims

        mcsa = MCSA(
            fitness_func=func,
            d=dims,
            iterMax=args.iterations,
            N=args.population,
            lower_bound=lb[:dims],
            upper_bound=ub[:dims]
        )

        best_position, best_score = mcsa.run()
        print(f"{name}: Best Score = {best_score}")

        plt.plot(mcsa.best_fitness_history, label=name)

    plt.title("Convergence Curves for Specific Functions")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.show()

    # Test all benchmark functions
    results = []
    plt.figure(figsize=(10, 6))
    for func, name in benchmark_functions:
        lb = [args.lower_bound] * args.dims
        ub = [args.upper_bound] * args.dims

        mcsa = MCSA(
            fitness_func=func,
            d=args.dims,
            iterMax=args.iterations,
            N=args.population,
            lower_bound=lb,
            upper_bound=ub
        )

        best_position, best_score = mcsa.run()
        results.append((name, best_score))
        print(f"{name}: Best Score = {best_score}")

        plt.plot(mcsa.best_fitness_history, label=name)

    plt.title("Convergence Curves for Benchmark Functions")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.show()

    # Radar chart for benchmark functions
    labels = [name.split()[0] for _, name in benchmark_functions]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores = [score for _, score in results]
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='black', alpha=0.25)
    ax.plot(angles, scores, color='black', linewidth=2, label="MCSA")
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    plt.title("Performance on F1-F23 Benchmark Functions")
    plt.show()
