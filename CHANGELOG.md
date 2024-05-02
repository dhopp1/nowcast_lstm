# Change Log
## 0.2.7
### Added
* reproducible results on a single machine via the `seeds` parameter.
* proper CUDA support. Automatic detection if you have a CUDA-enabled machine.

## 0.2.6
### Added
* enable logistic output/binary classification. Occurs automatically when `torch.nn.BCELoss()` function passed to the `criterion` parameter.

## 0.2.5
### Fixed
* fixed `init_test_size` of model selection functions working on the full dataset including NAs in the target variable

## 0.2.4

### Added

* make model instantiation and model selection robust to variables with no data in them

## 0.2.3

### Added

* feature contribution weighting for data availabilty in interval predict functions

### Fixed

* got rid of hardcoded references to _date_ column in interval predict code

## 0.2.2

### Added

* ability to generate uncertainty intervals via the `model.interval_predict()` function
* ability to generate uncertainty intervals on synthetic vintages via the `model.ragged_interval_predict()` function

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