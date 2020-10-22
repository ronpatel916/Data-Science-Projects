In this project, I apply several machine learning models to build a classification model to identify digits from low resolution images of handwritten digits.

The models I apply are:
1. Artificial Neural Networks
2. K-Nearest Neighbors Algorithm
3. Decision Trees.
4. Random Forest Ensemble techniques.

I found that the k-nearest neighbors algorithm produced the highest accuracy of roughy 96.8%. Despite being the most accurate, drawbacks of this algorithm include high memory usage and a lack of model representation to debug and explore.

I also explored building a deep neural network model to classify the handwritten digits. We found that this model did not produce as accurate results as the KNN algorithm. As we added more layers, we found that the model's accuracy improved. One risk of adding more layers to increase accuracy is overfitting the model. To prevent overfitting, we increased the number of folds our model used to train and test.

We also explored which activation functions in the MLPClassifier class produced the best results. We found that the logistic and tanh activation functions produce models with higher accuracies than ReLU.
