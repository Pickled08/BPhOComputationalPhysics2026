import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

from randomwalk import NRandomWalks


def run_plot():
    try:
        n_walks = int(n_walks_var.get())
        kappa = float(kappa_var.get())
        res = float(res_var.get())
        maxSteps = int(maxSteps_var.get())
        stepSize = float(stepSize_var.get())

        plt.clf()
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("2D Random Walk")

        results = NRandomWalks(n_walks, kappa, res, maxSteps, stepSize)

        for x, y, label in results:
            RGB = np.random.uniform(0, 1, 3)
            plt.plot(x, y, color=RGB, linewidth=0.5)

        plt.draw()

    except Exception as e:
        print("Error:", e)


# --- GUI SETUP ---
root = tk.Tk()
root.title("Random Walk GUI")

frame = ttk.Frame(root, padding=10)
frame.grid()

# Variables
n_walks_var = tk.StringVar(value="10")
kappa_var = tk.StringVar(value="0")
res_var = tk.StringVar(value="0.001")
maxSteps_var = tk.StringVar(value="1000000")
stepSize_var = tk.StringVar(value="1.0")

# Inputs
ttk.Label(frame, text="Number of walks").grid(row=0, column=0)
ttk.Entry(frame, textvariable=n_walks_var).grid(row=0, column=1)

ttk.Label(frame, text="Kappa").grid(row=1, column=0)
ttk.Entry(frame, textvariable=kappa_var).grid(row=1, column=1)

ttk.Label(frame, text="Resolution").grid(row=2, column=0)
ttk.Entry(frame, textvariable=res_var).grid(row=2, column=1)

ttk.Label(frame, text="Max Steps").grid(row=3, column=0)
ttk.Entry(frame, textvariable=maxSteps_var).grid(row=3, column=1)

ttk.Label(frame, text="Step Size").grid(row=4, column=0)
ttk.Entry(frame, textvariable=stepSize_var).grid(row=4, column=1)

# Button
ttk.Button(frame, text="Run Plot", command=run_plot).grid(row=5, column=0, columnspan=2)

# Matplotlib interactive mode
plt.ion()
plt.figure()

root.mainloop()