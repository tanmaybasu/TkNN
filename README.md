# A Tweak on k-Nearest Neghbor Decision Rule (TkNN)
A tweak on k-nearest neighbor decision rule. The proposed method restricts the majority voting of knn with a predefined positive integer threshold, say β, to assign a data point to a predefined class. For the proposed method there is no need to select a fixed k prior to classifying a point. The method will start with an
initial value of k as β. If the difference between the number of representatives of the best and the second best competing classes is β, then the point is classified to the best competing class. Otherwise the value of the neighborhood parameter k will be increased by one and the process will continue until the point is classified to a particular class. If the test data point is not classified till the process reaches the last point of the training set, then the test data point will remain unclassified. 

The method is explained in the following papers and the steps to implement the method are stated below.

[Tanmay Basu and C. A. Murthy. Towards Enriching the Quality of  $k$-Nearest Neighbor Decision Rule for 
Document Classification. International Journal of Machine Learning and Cybernetics, Springer, vol. 5(6), pp. 897-905, 2014](https://doi.org/10.1007/s13042-013-0177-1).


[Tanmay Basu, C. A. Murthy and Himadri Chakraborty. A Tweak on K Nearest Neighbour Decision Rule, published in Proceedings of the International Conference on Image Processing, Computer Vision, and Pattern Recognition, pp. 929-935, USA, 2012](https://pdfs.semanticscholar.org/7f8e/304d99bc4bb48a3a63600a20fd4ddaaf75b3.pdf}{ISBN: 1-60132-223-2).

## How to run the model?

The model is implemented in MedNN.py. Run the following lines to train the classifier on a set of data samples and subsequently test it's performance on another set of data samples. 

```
clf = TkNN(theta = 0.25,beta=2,metric='cosine')
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)
```

Here `X_train` is the training data and it is an array or matrix and has shapes '[n_samples, n_features]'. 'y_train' is the class labels of individual samples in 'X_train'. Similarly, 'X_test' is the test data and it is also an array or matrix and has shapes '[n_samples, n_features]'. The following options of distance metric s are available: 'cosine', 'chebyshev', 'cityblock', 'euclidean', 'minkowski'. 

An example code to implement TkNN using a sample data is uploaded as `testing.py`. For any further query, you may reach out to me at welcometanmay@gmail.com
