from feature_engineering import load_tickers, load_market_data, add_features
from train_model import read_features, read_model_options, read_model_params, initialize_model, evaluate_model, evaluate_feature_importance
from sklearn.model_selection import train_test_split
import numpy as np
from backtest import run_model_backtest, run_benchmark_backtest, plot_backtest

def main():
    tickers = load_tickers('tickers-1.csv')
    feature_cols = read_features('features_config.json')
    model_options = read_model_options('model_params.json')

    market_data = load_market_data('market_data_ml.csv')
    enhanced_data = add_features(market_data)

    for model in model_options:

        model_params = read_model_params('model_params.json', model)

        y_test_list = []
        clf_pred_list = []

        model_storage = {}

        for symbol in tickers:
            symbol_data = enhanced_data[enhanced_data['ticker'] == symbol]
            if symbol_data.empty:
                print(f"No data found for ticker: {symbol}")
                continue

            X = symbol_data[feature_cols]
            y = symbol_data['direction']
            X = X[:-1]
            y = y[:-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            clf = initialize_model(model, model_params)
            clf.fit(X_train, y_train)
            clf_pred = clf.predict(X_test)

            y_test_list.append(y_test)
            clf_pred_list.append(clf_pred)
            model_storage[symbol] = clf

            # model feature importance
            print(f'Feature importance for {symbol} using {model}:')
            evaluate_feature_importance(clf, model, feature_cols)
            print("\n" + "="*40 + "\n")
        
        # model evaluation
        y_test = np.concatenate(y_test_list)
        clf_pred = np.concatenate(clf_pred_list)
        evaluate_model(model, y_test, clf_pred)

        # model backtesting
        model_portfolio = run_model_backtest(model_storage, enhanced_data, tickers, feature_cols, 100000)
        plot_backtest(model_portfolio, title="Model Strategy")

        bench = run_benchmark_backtest(enhanced_data, tickers, 100000)
        plot_backtest(bench, title="Benchmark Buy & Hold")

if __name__ == "__main__":
    main()