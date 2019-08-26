# bayes-classifier

Implementation of a Naive Bayes classifier.
Takes in continuous or discrete data, providing a scipy-like interface for fitting and prediction.

## Usage

To use the classifier, pass in
* a list describing whether each column is continuous/binary
* number of classes in the output

```python
import bayes_classifier as bayes

nbc = bayes.NBC(feature_types=['b', 'c', 'b'], num_classes=4)
nbc.fit(X_train, y_train)
y_val = nbc.predict(X_val)
```
Were `b` denotes a binary-data type and a `c` denotes a continuous datatype.

## License
[MIT](https://choosealicense.com/licenses/mit/)
