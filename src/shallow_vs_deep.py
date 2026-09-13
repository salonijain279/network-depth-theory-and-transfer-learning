"""Replicates the core empirical finding of Mhaskar, Liao & Poggio (2017),
"When and Why Are Deep Networks Better Than Shallow Ones?" (AAAI):
for a compositional target function, deeper networks reach lower test error
than shallow networks at an equivalent neuron budget.

Target function: f(x) = 2*(2*cos^2(x) - 1)^2 - 1, x ~ Uniform[-2*pi, 2*pi].
This is a composition of simple functions (cos, square, affine), which is
exactly the structure deep networks are theoretically suited to exploit
efficiently -- each layer can compute one stage of the composition, whereas
a shallow network must approximate the whole composed function with a single
hidden layer of basis functions.

Three architectures are compared at matched neuron budgets:
  - 1 hidden layer:  Linear(n) -> ReLU -> Output
  - 2 hidden layers: Linear(n) -> BN -> ReLU -> Linear(n) -> BN -> ReLU -> Output
  - 3 hidden layers: three such blocks

Each configuration is trained multiple times (different random init) and the
best test RMSE is reported, following the paper's own evaluation protocol.
"""
import numpy as np
import torch
import torch.nn as nn

SEED = 42
N_SAMPLES = 120_000
N_TRAIN = 60_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def target_function(x: np.ndarray) -> np.ndarray:
    return 2 * (2 * np.cos(x) ** 2 - 1) ** 2 - 1


def generate_data(seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2 * np.pi, 2 * np.pi, size=N_SAMPLES).astype(np.float32)
    y = target_function(x).astype(np.float32)
    idx = rng.permutation(N_SAMPLES)
    train_idx, test_idx = idx[:N_TRAIN], idx[N_TRAIN:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


class MLP(nn.Module):
    def __init__(self, n_hidden_layers: int, neurons: int):
        super().__init__()
        layers = []
        in_dim = 1
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, neurons))
            if n_hidden_layers > 1:
                layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_evaluate(n_hidden_layers: int, neurons: int, x_train, y_train, x_test, y_test,
                        repeats: int = 3, epochs: int = 150, batch_size: int = 3000, seed: int = SEED):
    x_train_t = torch.from_numpy(x_train).unsqueeze(1).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1).to(DEVICE)
    y_test_t = torch.from_numpy(y_test).to(DEVICE)
    n = len(x_train_t)

    best_rmse = float("inf")
    for r in range(repeats):
        torch.manual_seed(seed + r)
        model = MLP(n_hidden_layers, neurons).to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = model(x_train_t[idx])
                loss = loss_fn(pred, y_train_t[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test_t)
            test_mse = loss_fn(test_pred, y_test_t).item()
        rmse = float(np.sqrt(test_mse))
        best_rmse = min(best_rmse, rmse)
    return best_rmse


NEURON_CONFIGS = {
    1: [8, 16, 24, 32, 48],
    2: [8, 16, 24],
    3: [6, 12, 18],
}


def run_experiment(repeats: int = 2, epochs: int = 150):
    x_train, y_train, x_test, y_test = generate_data()
    results = {}
    for n_layers, neuron_list in NEURON_CONFIGS.items():
        rmses = []
        for n in neuron_list:
            rmse = train_and_evaluate(n_layers, n, x_train, y_train, x_test, y_test,
                                       repeats=repeats, epochs=epochs)
            rmses.append(rmse)
            print(f"{n_layers} hidden layer(s), {n} neurons/layer -> test RMSE {rmse:.4f}")
        results[n_layers] = {"neurons": neuron_list, "rmse": rmses}
    return results


if __name__ == "__main__":
    run_experiment()
