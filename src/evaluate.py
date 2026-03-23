"""
src/evaluate.py
===============
Evaluation metrics, residual plots, and SHAP explainability utilities.

Usage
-----
    from src.evaluate import evaluate_model, plot_training_history, plot_residuals

    metrics = evaluate_model('DNN', model, val_inputs, y_val, test_inputs, y_test)
    plot_training_history(history)
    plot_residuals(y_val, pred_val, title='DNN Residuals')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : np.ndarray — ground truth target values
    y_pred : np.ndarray — model predictions (clipped to [0, 1])

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return {
        'mae'  : float(mean_absolute_error(y_true, y_pred)),
        'rmse' : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2'   : float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    name         : str,
    model,
    val_inputs,
    y_val        : np.ndarray,
    test_inputs  = None,
    y_test       : np.ndarray = None,
    batch_size   : int = 1024,
    verbose      : bool = True,
) -> dict:
    """
    Evaluate a trained model on validation (and optionally test) data.

    Parameters
    ----------
    name        : str — display name for the model
    model       : trained keras.Model or sklearn estimator
    val_inputs  : model inputs for validation (dict for Keras, array for sklearn)
    y_val       : validation targets
    test_inputs : model inputs for test (optional)
    y_test      : test targets (optional)
    batch_size  : batch size for Keras predict (ignored for sklearn)
    verbose     : print the results table

    Returns
    -------
    dict with keys: name, val_mae, val_rmse, val_r2, [test_mae, test_rmse, test_r2]
    """
    # ── Predict ────────────────────────────────────────────────────
    # Handle both Keras models (predict method) and sklearn estimators
    if hasattr(model, 'predict'):
        try:
            pred_val = model.predict(val_inputs, batch_size=batch_size,
                                     verbose=0).ravel()
        except TypeError:
            pred_val = model.predict(val_inputs).ravel()
    else:
        pred_val = model.predict(val_inputs).ravel()

    results = {'name': name, **{f'val_{k}': v
                                for k, v in compute_metrics(y_val, pred_val).items()}}

    if test_inputs is not None and y_test is not None:
        try:
            pred_test = model.predict(test_inputs, batch_size=batch_size,
                                      verbose=0).ravel()
        except TypeError:
            pred_test = model.predict(test_inputs).ravel()
        results.update({f'test_{k}': v
                        for k, v in compute_metrics(y_test, pred_test).items()})

    if verbose:
        print(f"  {name:<28}  "
              f"Val  MAE={results['val_mae']:.4f}  "
              f"RMSE={results['val_rmse']:.4f}  "
              f"R²={results['val_r2']:.4f}", end='')
        if 'test_mae' in results:
            print(f"  |  Test MAE={results['test_mae']:.4f}  "
                  f"RMSE={results['test_rmse']:.4f}  "
                  f"R²={results['test_r2']:.4f}", end='')
        print()

    return results


def dummy_baseline(y_train: np.ndarray, y_val: np.ndarray,
                   y_test: np.ndarray = None) -> dict:
    """
    Compute metrics for a dummy model that always predicts the training mean.

    Parameters
    ----------
    y_train : training targets (used to compute the mean)
    y_val   : validation targets
    y_test  : test targets (optional)

    Returns
    -------
    dict with same structure as evaluate_model output
    """
    mean_pred = y_train.mean()
    results = {
        'name'     : 'Dummy (mean)',
        **{f'val_{k}': v for k, v in
           compute_metrics(y_val, np.full_like(y_val, mean_pred)).items()}
    }
    if y_test is not None:
        results.update({f'test_{k}': v for k, v in
                        compute_metrics(y_test, np.full_like(y_test, mean_pred)).items()})
    return results


# ── Training curves ───────────────────────────────────────────────────────

def plot_training_history(
    history,
    title     : str = 'Training History',
    save_path : str = None,
) -> None:
    """
    Plot training and validation loss + RMSE curves side by side.

    Parameters
    ----------
    history   : keras.callbacks.History from model.fit()
    title     : plot title
    save_path : full file path to save the figure, or None
    """
    h      = history.history
    epochs = range(1, len(h['loss']) + 1)
    best   = int(np.argmin(h['val_loss']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (tr_key, vl_key, ylabel) in zip(axes, [
        ('loss',     'val_loss',    'MAE'),
        ('rmse',     'val_rmse',    'RMSE'),
    ]):
        ax.plot(epochs, h[tr_key], 'steelblue', lw=2, label=f'Train {ylabel}')
        ax.plot(epochs, h[vl_key], 'tomato',    lw=2, label=f'Val {ylabel}')
        ax.axvline(best + 1, color='gold', ls='--', lw=1.5,
                   label=f'Best epoch: {best + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Train vs Val {ylabel}')
        ax.legend()

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    plt.show()

    print(f"Best epoch    : {best + 1}")
    print(f"Best val MAE  : {min(h['val_loss']):.4f}")
    print(f"Best val RMSE : {h['val_rmse'][best]:.4f}")


# ── Residual plots ────────────────────────────────────────────────────────

def plot_residuals(
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    title     : str = 'Residual Analysis',
    save_path : str = None,
) -> None:
    """
    Plot three residual diagnostics: predicted vs actual, residuals vs
    predicted, and residual distribution.

    Parameters
    ----------
    y_true    : ground truth values
    y_pred    : model predictions
    title     : plot title
    save_path : full file path to save the figure, or None
    """
    y_pred = np.clip(y_pred, 0.0, 1.0)
    resids = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Predicted vs actual
    axes[0].scatter(y_pred, y_true, alpha=0.06, s=5,
                    color='steelblue', rasterized=True)
    axes[0].plot([0, 1], [0, 1], 'r--', lw=1.5, label='Perfect fit')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Predicted delay_rate')
    axes[0].set_ylabel('Actual delay_rate')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()

    # Residuals vs predicted
    axes[1].scatter(y_pred, resids, alpha=0.06, s=5,
                    color='darkorange', rasterized=True)
    axes[1].axhline(0, color='red', ls='--', lw=1.5)
    axes[1].axhline(resids.mean(), color='blue', ls=':', lw=1.2,
                    label=f'Mean residual = {resids.mean():.4f}')
    axes[1].set_xlabel('Predicted delay_rate')
    axes[1].set_ylabel('Residual (actual − predicted)')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].legend(fontsize=9)

    # Residual distribution
    axes[2].hist(resids, bins=60, color='steelblue',
                 edgecolor='white', alpha=0.85, density=True)
    axes[2].axvline(0, color='red', ls='--', lw=1.5)
    axes[2].set_xlabel('Residual')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Residual Distribution')

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    plt.show()


# ── Results comparison ────────────────────────────────────────────────────

def make_results_table(results_list: list) -> pd.DataFrame:
    """
    Build a sorted comparison DataFrame from a list of result dicts.

    Parameters
    ----------
    results_list : list of dicts — each from evaluate_model() or dummy_baseline()

    Returns
    -------
    pd.DataFrame sorted by val_mae ascending (lower = better)
    """
    df = pd.DataFrame(results_list)
    metric_cols = [c for c in df.columns if c != 'name']
    df = df[['name'] + metric_cols].sort_values('val_mae').reset_index(drop=True)
    df.index += 1
    return df
