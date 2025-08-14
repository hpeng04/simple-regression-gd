import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Linear Regression implemented with Batch Gradient Descent

INITIAL_WEIGHT = 0
INITIAL_BIAS = 0
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
FPS = 40
BATCH_SIZE = 2

# Global list to store MSE values for animation
mse_history = []

"""
Batch Gradient Descent for Linear Regression
This function performs batch gradient descent to find the optimal parameters for a linear regression model.
parameters: data (pd.DataFrame): DataFrame containing the input features and target variable.
yields: Generator yielding the updated parameters (weight, bias) and the title for each epoch.
"""
def gradient_descent(data):
    global BATCH_SIZE, mse_history
    mse_history = []  # Reset MSE history
    data_array = data.to_numpy()
    w = INITIAL_WEIGHT
    b = INITIAL_BIAS
    n = len(data_array)
    BATCH_SIZE = min(BATCH_SIZE, n)  # Ensure batch size does not exceed data length
    for i in range(1, NUM_EPOCHS+1):
        np.random.shuffle(data_array)  # Shuffle data for each epoch
        for j in range(0, n, BATCH_SIZE):
            batch = data_array[j:j+BATCH_SIZE]
            dw = 0
            db = 0
            loss_array = []
            for x, y in batch:
                y_pred = w * x + b
                error = y - y_pred
                loss_array.append(error)
                dw += -2 * x * error
                db += -2 * error
            dw /= BATCH_SIZE
            db /= BATCH_SIZE
            w -= LEARNING_RATE * dw
            b -= LEARNING_RATE * db
            mse = np.sum(np.array(loss_array)**2)/len(loss_array)
            mae = np.sum(np.abs(np.array(loss_array)))/len(loss_array)

        mse_history.append(mse)  # Store MSE for animation
        title = f"EPOCH {i}: MSE={mse:.2f} MAE={mae:.2f}"
        print(title)
        yield w, b, title
    print(f"Trained model parameters: w={w}, b={b}")

def create_regression_animation(data):
    fig, ax = plt.subplots()
    ax.set_xlim(data['x'].min(), data['x'].max())
    ax.set_ylim(data['y'].min(), data['y'].max())
    line, = ax.plot([], [], lw=2)
    scatter = ax.scatter(data['x'], data['y'])
    title_text = ax.set_title("Linear Regression with Batch Gradient Descent")

    def init():
        line.set_data([], [])
        return line, title_text

    def update(frame):
        w, b, title = frame
        x_vals = np.array(ax.get_xlim())
        y_vals = w * x_vals + b
        line.set_data(x_vals, y_vals)
        title_text.set_text(title)
        return line, title_text

    ani = FuncAnimation(fig, update, frames=gradient_descent(data),
                        init_func=init, blit=False, repeat=False, interval=1000/FPS)
    plt.show()

def create_mse_animation(data):
    """
    Create an animation showing the MSE trend over epochs.
    This function must be called after gradient_descent has been run to populate mse_history.
    """
    global mse_history
    
    # First, run gradient descent to get MSE values
    list(gradient_descent(data))  # Consume the generator to populate mse_history
    
    if not mse_history:
        print("No MSE data available. Run gradient descent first.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(mse_history))
    ax.set_ylim(0, max(mse_history) * 1.1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('MSE Trend During Training')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'b-', linewidth=2, label='MSE')
    points = ax.scatter([], [], c='red', s=30, zorder=5)
    ax.legend()
    
    # Text box for current MSE value
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                      verticalalignment='top', fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        line.set_data([], [])
        points.set_offsets(np.empty((0, 2)))
        text_box.set_text('')
        return line, points, text_box
    
    def animate(frame):
        epochs = range(1, frame + 2)
        mse_values = mse_history[:frame + 1]
        
        # Update line plot
        line.set_data(epochs, mse_values)
        
        # Update scatter plot (current point)
        if frame < len(mse_history):
            current_epoch = frame + 1
            current_mse = mse_history[frame]
            points.set_offsets([[current_epoch, current_mse]])
            
            # Update text box
            text_box.set_text(f'Epoch: {current_epoch}\nMSE: {current_mse:.4f}')
        
        return line, points, text_box
    
    ani = FuncAnimation(fig, animate, frames=len(mse_history),
                       init_func=init, blit=False, repeat=False, 
                       interval=100)  # Faster animation for MSE trend
    plt.tight_layout()
    plt.show()
    
    return ani

def load_data(data_path):
    with open(data_path, "r") as data:
        df = pd.read_csv(data)
        print("Data loaded successfully.")
        print(df.head())
    return df

def show_mse_animation():
    """Convenience function to show only the MSE animation"""
    data = load_data("data.csv")
    if data is not None:
        create_mse_animation(data)
    else:
        print("Failed to load data.")

def plot_mse_static(data):
    """
    Create a static plot showing the complete MSE trend.
    Useful for seeing the overall trend without animation.
    """
    global mse_history
    
    # Run gradient descent to get MSE values
    list(gradient_descent(data))  # Consume the generator to populate mse_history
    
    if not mse_history:
        print("No MSE data available. Run gradient descent first.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create subplot for MSE trend
    plt.subplot(1, 2, 1)
    epochs = range(1, len(mse_history) + 1)
    plt.plot(epochs, mse_history, 'b-', linewidth=2, label='MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE Trend During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create subplot for log-scale MSE (better for seeing convergence)
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs, mse_history, 'r-', linewidth=2, label='MSE (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) - Log Scale')
    plt.title('MSE Trend (Log Scale) - Better View of Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nMSE Statistics:")
    print(f"Initial MSE: {mse_history[0]:.4f}")
    print(f"Final MSE: {mse_history[-1]:.4f}")
    print(f"Reduction: {((mse_history[0] - mse_history[-1]) / mse_history[0] * 100):.2f}%")
    print(f"Minimum MSE: {min(mse_history):.4f} at epoch {mse_history.index(min(mse_history)) + 1}")

def show_mse_static():
    """Convenience function to show only the static MSE plot"""
    data = load_data("data.csv")
    if data is not None:
        plot_mse_static(data)
    else:
        print("Failed to load data.")

def main():
    data = load_data("data.csv")
    if data is not None:
        print("\nChoose visualization type:")
        print("1. Regression line animation (shows how the line fits the data)")
        print("2. MSE trend animation (shows how MSE decreases over epochs)")
        print("3. MSE trend static plot (complete view with statistics)")
        print("4. All visualizations")
        
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()
        
        if choice == "1":
            create_regression_animation(data)
        elif choice == "2":
            create_mse_animation(data)
        elif choice == "3":
            plot_mse_static(data)
        elif choice == "4":
            create_regression_animation(data)
            create_mse_animation(data)
            plot_mse_static(data)
        else:
            print("Invalid choice. Showing MSE trend animation by default.")
            create_mse_animation(data)
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()