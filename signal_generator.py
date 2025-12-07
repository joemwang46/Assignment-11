import pandas as pd

def generate_signals(x_test: pd.DataFrame, model) -> list:
    signals = []
    predictions = model.predict(x_test)

    for pred in predictions:
        if pred == 1:
            signals.append("BUY")
        else:
            signals.append("HOLD")
    return predictions

