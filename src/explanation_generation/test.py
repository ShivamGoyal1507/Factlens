import os
os.environ["USE_TF"] = "0"

from explainer import VerdictExplainer
import time

explainer_model = VerdictExplainer()

claim = "Python is an interpreted programming language."

verdict = "Supported"

evidences = [
    "Python is commonly described as a high-level, interpreted programming language.",
"The standard Python implementation, CPython, compiles source code into bytecode before executing it using a virtual machine.",
"Python supports object-oriented and functional programming."
]

start = time.perf_counter()

print(explainer_model.explain(claim, verdict, evidences))

print(f"Time taken: {time.perf_counter() - start:.2f} seconds")