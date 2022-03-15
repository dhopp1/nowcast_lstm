# Change Log

## 0.2.0

### Added

* ability to obtain feature contributions to the model via `model.feature_contribution()` function
* automatic variable selection given a set of hyperparameters via `variable_selection()` function in `LSTM.model_selection`
* automatic hyperparameter tuning given a set of variables via `hyperparameter_tuning()` function in `LSTM.model_selection`
* automatic variable and hyperparameter tuning via `select_model()` function in `LSTM.model_selection`

### Changed

* hide printing of `Training model n` when `quiet=True`