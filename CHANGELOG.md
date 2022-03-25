# Change Log

## 0.2.1

### Added

* `initial_ordering` parameter to `variable_selection()` and `select_model()` functions. In recursive feature addition (RFA) variable selection, can obtain initial variable order either via their feature contribution in a full model, or from univariate model performances. Former (default) is about 2x faster.

## 0.2.0

### Added

* ability to obtain feature contributions to the model via `model.feature_contribution()` function
* automatic variable selection given a set of hyperparameters via `variable_selection()` function in `LSTM.model_selection`
* automatic hyperparameter tuning given a set of variables via `hyperparameter_tuning()` function in `LSTM.model_selection`
* automatic variable and hyperparameter tuning via `select_model()` function in `LSTM.model_selection`

### Changed

* hide printing of `Training model n` when `quiet=True`