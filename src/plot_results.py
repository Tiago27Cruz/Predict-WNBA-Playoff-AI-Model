from matplotlib import pyplot as plt
import numpy as np


def read_result(model):
    with open(f"results_{model}.txt", "r") as file:
        return [float(line.rstrip()) for line in file]

def plot_results(models):
    x = np.linspace(3, 10, 8)
    for model in models:
        series = read_result(model)
        print(series)
        plt.plot(x, series, label=model)
    
    plt.legend()
    plt.savefig("results.png", dpi=200)

plot_results(["svc", "knn", "randomforest", "gradientboost", "adaboost", "mlp", "decisiontree", "gradientboost_noPCA"])