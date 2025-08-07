import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Linear Regression implemented with Batch Gradient Descent

INITIAL_WEIGHT = 0
INITIAL_BIAS = 0
NUM_ITERATIONS = 50
LEARNING_RATE = 0.01
FPS = 10

"""
Batch Gradient Descent for Linear Regression
This function performs batch gradient descent to find the optimal parameters for a linear regression model.
parameters: data (pd.DataFrame): DataFrame containing the input features and target variable.
yields: Generator yielding the updated parameters (weight, bias) and the title for each epoch.
"""
def gradient_descent(data):
    data_array = data.to_numpy()
    w = INITIAL_WEIGHT
    b = INITIAL_BIAS
    n = len(data_array)
    for i in range(1, NUM_ITERATIONS+1):
        dw = 0
        db = 0
        loss_array = []
        for x, y in data_array:
            y_pred = w * x + b
            error = y - y_pred
            loss_array.append(error)
            dw += -2 * x * error
            db += -2 * error
        dw /= n
        db /= n
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db
        mse = np.sum(np.array(loss_array)**2)/len(loss_array)
        mae = np.sum(np.abs(np.array(loss_array)))/len(loss_array)

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

def load_data(data_path):
    with open(data_path, "r") as data:
        df = pd.read_csv(data)
        print("Data loaded successfully.")
        print(df.head())
    return df

def main():
    data = load_data("data.csv")
    if data is not None:
        create_regression_animation(data)
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()