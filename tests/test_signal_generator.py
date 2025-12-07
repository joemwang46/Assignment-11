import numpy as np
import pandas as pd
from signal_generator import generate_signals

class DummyModel:
    def predict(self, X):
        return np.array([1, 0, 1])

def test_generate_signals():
    df = pd.DataFrame({"f": [1, 2, 3]})
    signals = generate_signals(df, DummyModel())

    assert len(signals) == 3
    assert list(signals) == [1, 0, 1]
