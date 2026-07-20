# LinkedIn Drafts

## Short

I built a PyTorch research project to study federated learning under controlled label-poisoning and backdoor attacks on MNIST. I compared FedAvg, coordinate-wise median, trimmed mean, and an experimental fuzzy client-reliability rule inspired by my mathematics research in uncertainty modelling. The project is educational, reproducible, and focused on defensive ML security. [GitHub link]

## Medium

How robust is federated learning when some clients cannot be trusted?

I developed a local PyTorch simulation with multiple MNIST clients and evaluated controlled label-poisoning and backdoor attacks. The project compares FedAvg with coordinate-wise median, trimmed mean, and a fuzzy client-reliability aggregation prototype based on update norm, distance, and cosine alignment.

This connects my M.Phil. work in fuzzy uncertainty and matrix games with my current direction in machine learning security. I report clean accuracy and attack success separately, save reproducible configurations, and state the method's limitations clearly—this is an experimental idea, not a security guarantee.

Code and methodology: [GitHub link]

## Technical

I completed a reproducible PyTorch study of robust federated learning on MNIST. Ten simulated clients train a CNN through synchronous communication rounds. Configurable malicious clients perform source-to-target label flipping or train on a white-square trigger with a target label. Evaluation separates clean accuracy from backdoor ASR and label source-to-target behavior.

I implemented FedAvg, coordinate-wise median, trimmed mean, and an experimental fuzzy weighting rule. The latter robustly scales update-norm deviation and distance from the mean, combines them with cosine alignment as membership/non-membership evidence, and applies softmax to reliability scores. Per-round client diagnostics make the heuristic inspectable.

The next step is evaluating multiple seeds, Dirichlet non-IID splits, partial participation, adaptive attacks, and stronger published baselines. [GitHub link]

## Non-technical

Imagine several people jointly teaching one model without exchanging their private notebooks. What happens if some participants submit misleading lessons? This project creates a safe miniature version using handwritten digits and compares ways to reduce suspicious contributions.

## Hashtags

`#FederatedLearning #MachineLearningSecurity #PyTorch #RobustML #AdversarialML #DataScience #Research`

