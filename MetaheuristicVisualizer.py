import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import seaborn as sns
import pandas as pd
from tqdm import tqdm

plt.style.use('seaborn-darkgrid')

class MetaheuristicVisualizer:
    def __init__(self):
        self.results = {
            'PSO': {'fitness': [], 'time': [], 'params': {}},
            'GA': {'fitness': [], 'time': [], 'params': {}},
            'ACO': {'fitness': [], 'time': [], 'params': {}}
        }
        self.functions = {
            'Sphere': lambda x: np.sum(x**2),
            'Rastrigin': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)),
            'Rosenbrock': lambda x: np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        }
        
    def run_algorithm(self, algorithm, dimensions=2, iterations=100):
        """Simulate algorithm run (placeholder for actual implementations)"""
        fitness_history = []
        time_history = []
        
        # Simulate performance patterns
        if algorithm == 'PSO':
            # PSO typically shows fast initial convergence
            base = np.logspace(3, 0, iterations)
            noise = np.random.normal(0, 0.1, iterations)
            fitness_history = base + noise
            
        elif algorithm == 'GA':
            # GA shows more step-wise improvement
            steps = np.linspace(0, iterations, 5)
            for i in range(1, len(steps)):
                fitness_history.extend(np.linspace(10/(i**2), 10/((i+1)**2), int(steps[i]-steps[i-1])))
            fitness_history += np.random.normal(0, 0.05, iterations)
            
        elif algorithm == 'ACO':
            # ACO shows slower but steady improvement
            fitness_history = 5/(1 + np.exp(0.1*(np.arange(iterations)-50))) + np.random.normal(0, 0.02, iterations)
        
        time_history = np.cumsum(np.random.exponential(0.1, iterations))
        
        self.results[algorithm]['fitness'] = fitness_history
        self.results[algorithm]['time'] = time_history
        self.results[algorithm]['params'] = {
            'dimensions': dimensions,
            'iterations': iterations
        }
    
    def plot_convergence(self, ax=None):
        """Plot fitness convergence comparison"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in self.results:
            if self.results[algo]['fitness']:
                ax.plot(self.results[algo]['fitness'], 
                       label=f"{algo} (D={self.results[algo]['params']['dimensions']})")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness (Lower = Better)')
        ax.set_title('Algorithm Convergence Comparison')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True)
        plt.tight_layout()
        
    def plot_runtime(self, ax=None):
        """Plot computational time comparison"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in self.results:
            if self.results[algo]['time']:
                ax.plot(self.results[algo]['time'], self.results[algo]['fitness'],
                       label=f"{algo} (D={self.results[algo]['params']['dimensions']})")
        
        ax.set_xlabel('Computation Time (s)')
        ax.set_ylabel('Fitness (Lower = Better)')
        ax.set_title('Runtime Performance Comparison')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True)
        plt.tight_layout()
    
    def plot_parallel_coordinates(self):
        """Parallel coordinates plot for multi-run analysis"""
        data = []
        for algo in self.results:
            if self.results[algo]['fitness']:
                data.append({
                    'Algorithm': algo,
                    'Dimensions': self.results[algo]['params']['dimensions'],
                    'Final Fitness': np.min(self.results[algo]['fitness']),
                    'Convergence Speed': 1/np.argmin(self.results[algo]['fitness']),
                    'Runtime': self.results[algo]['time'][-1]
                })
        
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(12, 6))
        pd.plotting.parallel_coordinates(
            df, 'Algorithm', 
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            ax=ax
        )
        ax.set_title('Algorithm Performance Characteristics')
        ax.grid(True)
        plt.tight_layout()
    
    def plot_boxplot(self, runs=5):
        """Boxplot comparison after multiple runs"""
        data = {algo: [] for algo in self.results}
        
        for _ in tqdm(range(runs), desc="Running multiple trials"):
            for algo in self.results:
                self.run_algorithm(algo)
                data[algo].append(np.min(self.results[algo]['fitness']))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data.values(), labels=data.keys(),
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='red'))
        ax.set_ylabel('Best Fitness Found')
        ax.set_title(f'Algorithm Performance over {runs} Runs')
        ax.grid(True)
        plt.tight_layout()
    
    def interactive_dashboard(self):
        """Interactive dashboard for performance analysis"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create axes
        ax_convergence = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax_runtime = plt.subplot2grid((3, 2), (1, 0))
        ax_parallel = plt.subplot2grid((3, 2), (1, 1))
        ax_boxplot = plt.subplot2grid((3, 2), (2, 0), colspan=2))
        
        # Adjust layout for controls
        plt.subplots_adjust(bottom=0.25)
        
        # Control axes
        ax_color = 'lightgoldenrodyellow'
        ax_dim = plt.axes([0.2, 0.15, 0.2, 0.03], facecolor=ax_color)
        ax_iter = plt.axes([0.2, 0.10, 0.2, 0.03], facecolor=ax_color)
        ax_func = plt.axes([0.6, 0.15, 0.2, 0.03], facecolor=ax_color)
        ax_runs = plt.axes([0.6, 0.10, 0.2, 0.03], facecolor=ax_color)
        
        # Create controls
        s_dim = Slider(ax_dim, 'Dimensions', 2, 10, valinit=2, valstep=1)
        s_iter = Slider(ax_iter, 'Iterations', 50, 500, valinit=100, valstep=50)
        s_func = RadioButtons(ax_func, ('Sphere', 'Rastrigin', 'Rosenbrock'), active=0)
        s_runs = Slider(ax_runs, 'Boxplot Runs', 3, 20, valinit=5, valstep=1)
        
        # Update function
        def update(val):
            dim = int(s_dim.val)
            iterations = int(s_iter.val)
            func_name = s_func.value_selected
            
            # Update all algorithms
            for algo in self.results:
                self.run_algorithm(algo, dimensions=dim, iterations=iterations)
            
            # Update plots
            ax_convergence.clear()
            self.plot_convergence(ax_convergence)
            
            ax_runtime.clear()
            self.plot_runtime(ax_runtime)
            
            ax_parallel.clear()
            self.plot_parallel_coordinates()
            fig.delaxes(fig.axes[3])  # Remove duplicate axis
            
            ax_boxplot.clear()
            self.plot_boxplot(runs=int(s_runs.val))
            
            fig.suptitle(f'Performance Analysis - {func_name} Function', y=0.98)
            fig.canvas.draw_idle()
        
        # Register update functions
        s_dim.on_changed(update)
        s_iter.on_changed(update)
        s_func.on_clicked(update)
        s_runs.on_changed(update)
        
        # Initial update
        update(None)
        
        plt.show()

if __name__ == "__main__":
    visualizer = MetaheuristicVisualizer()
    
    # Quick static comparison
    print("Running quick comparison...")
    visualizer.run_algorithm('PSO')
    visualizer.run_algorithm('GA')
    visualizer.run_algorithm('ACO')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    visualizer.plot_convergence()
    
    plt.subplot(2, 2, 2)
    visualizer.plot_runtime()
    
    plt.subplot(2, 2, 3)
    visualizer.plot_parallel_coordinates()
    
    plt.subplot(2, 2, 4)
    visualizer.plot_boxplot(runs=5)
    
    plt.suptitle('Metaheuristic Algorithm Performance Comparison', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Uncomment for interactive dashboard
    # print("\nLaunching interactive dashboard...")
    # visualizer.interactive_dashboard()
