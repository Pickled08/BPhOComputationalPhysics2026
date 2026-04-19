import tkinter as tk
from tkinter import ttk
import os
import subprocess
import sys
import threading
import json

# Load program configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "programs.json")

with open(json_path) as f:
    programs = json.load(f)

#Commands
def launch_program(program, *args):
    def run():
        full_path = os.path.join(base_dir, program)
        subprocess.run([sys.executable, full_path, *args])

    threading.Thread(target=run, daemon=True).start()
    
def open_file(filepath):
    full_filepath = os.path.join(base_dir, filepath)
    if sys.platform.startswith("win"):
        os.startfile(full_filepath)
    elif sys.platform.startswith("darwin"):  # macOS
        subprocess.run(["open", full_filepath])
    else:  # Linux
        subprocess.run(["xdg-open", full_filepath])

#GUI interface to launch and confingure each program from

root = tk.Tk()

# Setting some window properties
root.title("Quantum Manager")
root.configure(background="white")
root.minsize(500, 500)

#Style
style = ttk.Style()
style.configure("TButton",
                font=("Arial", 12),
                padding=6)

ttk.Label(root, text="Quantum Manager").pack()
ttk.Label(root, text="Tools of the Quantum World").pack()

#Buttons to launch each program
for prog in programs:
    ttk.Button(
        root,
        text=prog["name"],
        command=lambda p=prog: launch_program(p["path"], *p["args"])
    ).pack(pady=10)

ttk.Button(
    root,
    text="Add/Edit",
    command=lambda: open_file("programs.json")
).pack(pady=20)

root.mainloop()