from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class Regressor(Protocol):
    def fit(self, features: np.ndarray, target: np.ndarray) -> Any: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...


class ScaledRegressor:
    """Standardize features before fitting a regular sklearn regressor."""

    def __init__(self, model: Regressor) -> None:
        from sklearn.preprocessing import StandardScaler

        self.model = model
        self.scaler = StandardScaler()

    def fit(self, features: np.ndarray, target: np.ndarray) -> "ScaledRegressor":
        self.model.fit(self.scaler.fit_transform(features), target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(features)))


class TorchLSTMRegressor:
    """Small two-layer LSTM with a sklearn-compatible interface.

    This model is only selected when PyTorch is installed. Padding keeps one
    output per input row, which makes time-ordered validation straightforward.
    """

    def __init__(self, sequence_length: int = 48, epochs: int = 10, seed: int = 42) -> None:
        import torch
        from sklearn.preprocessing import StandardScaler

        self.torch = torch
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.seed = seed
        self.scaler = StandardScaler()
        self.network: Any = None
        self.target_scale = 1.0

    def _sequences(self, features: np.ndarray):
        values = self.scaler.transform(features)
        padding = np.repeat(values[:1], self.sequence_length - 1, axis=0)
        values = np.vstack([padding, values]).astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(values, self.sequence_length, axis=0)
        return self.torch.from_numpy(np.ascontiguousarray(windows.transpose(0, 2, 1)))

    def fit(self, features: np.ndarray, target: np.ndarray) -> "TorchLSTMRegressor":
        torch = self.torch
        torch.manual_seed(self.seed)
        self.scaler.fit(features)
        self.target_scale = max(float(np.std(target)), 1e-8)

        class Network(torch.nn.Module):
            def __init__(inner_self, feature_count: int) -> None:
                super().__init__()
                inner_self.rnn = torch.nn.LSTM(feature_count, 48, num_layers=2, batch_first=True, dropout=0.15)
                inner_self.output = torch.nn.Linear(48, 1)

            def forward(inner_self, values):
                output, _ = inner_self.rnn(values)
                return inner_self.output(output[:, -1]).squeeze(-1)

        self.network = Network(features.shape[1])
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        loss_function = torch.nn.HuberLoss()
        sequences = self._sequences(features)
        targets = torch.tensor(target / self.target_scale, dtype=torch.float32)
        for _ in range(self.epochs):
            self.network.train()
            for indexes in torch.randperm(len(sequences)).split(256):
                optimizer.zero_grad()
                loss = loss_function(self.network(sequences[indexes]), targets[indexes])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.network.eval()
        with self.torch.no_grad():
            return self.network(self._sequences(features)).cpu().numpy() * self.target_scale


def build_models(sequence_length: int = 48, random_state: int = 42) -> dict[str, Regressor]:
    """Build three diverse regressors, preferring XGBoost and a real LSTM.

    Portable sklearn fallbacks keep the core operational when optional native or
    deep-learning dependencies are unavailable. The API reports the actual model
    names, so a fallback is never presented as XGBoost or LSTM.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor

    models: dict[str, Regressor] = {}
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        models["Gradient Boosting"] = HistGradientBoostingRegressor(
            max_iter=250, max_leaf_nodes=20, learning_rate=0.04, l2_regularization=0.2,
            random_state=random_state,
        )

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=16,
        n_jobs=-1,
        random_state=random_state,
    )
    try:
        models["LSTM"] = TorchLSTMRegressor(sequence_length, epochs=10, seed=random_state)
    except ImportError:
        models["Neural Network"] = ScaledRegressor(MLPRegressor(
            hidden_layer_sizes=(48, 24),
            activation="tanh",
            alpha=0.001,
            early_stopping=True,
            max_iter=250,
            random_state=random_state,
        ))
    return models


ModelFactory = Callable[[], dict[str, Regressor]]
