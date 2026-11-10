import os
os.environ["USE_TF"] = "0"

from explainer import VerdictExplainer

explainer_model = VerdictExplainer()

claim = "The Eiffel Tower is located in Berlin."

verdict = "Refuted"

evidences = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Berlin is the capital and largest city of Germany.",
    "The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair."
]

print(explainer_model.explain(claim, verdict, evidences))