# A Tweak on k-Nearest Neghbor Decision Rule (TkNN)
A nearest neighbor decision rule is developed here for data classification based on a tweak on k-nearest neighbor (kNN) decision rule. This method restricts the majority voting of kNN decision rule with a predefined positive integer threshold, say β, to assign a data point to a given class. For the proposed method, there is no need to select a fixed k prior to classifying a data point. The method starts with an initial value of k as β. If the difference between the number of members of the best and the second best competing classes is β, then the data point is classified to the best competing class. Otherwise the value of the neighborhood parameter k is increased by one and the process continues till the point is classified to a particular class. If the test data point is not classified till the process reaches the last data point of the training set, then it remain unclassified, which is rare in practice. The method has performed well for text classification.

The method is explained in the following papers and the steps to implement the method are stated below.

[Tanmay Basu and C. A. Murthy. Towards Enriching the Quality of  k-Nearest Neighbor Decision Rule for Document Classification. International Journal of Machine Learning and Cybernetics, Springer, vol. 5(6), pp. 897-905, 2014](https://doi.org/10.1007/s13042-013-0177-1).


[Tanmay Basu, C. A. Murthy and Himadri Chakraborty. A Tweak on K Nearest Neighbour Decision Rule, published in Proceedings of the International Conference on Image Processing, Computer Vision, and Pattern Recognition, pp. 929-935, USA, 2012](https://pdfs.semanticscholar.org/7f8e/304d99bc4bb48a3a63600a20fd4ddaaf75b3.pdf).

## Prerequsites
[Python 3 version](https://www.python.org/downloads/), [NumPy](https://numpy.org/install/), [Scipy](https://pypi.org/project/scipy/) [Scikit-Learn](https://scikit-learn.org/0.16/install.html)

## How to run the model?

The model is implemented in MedNN.py. Run the following lines to train the classifier on a set of data samples and subsequently test it's performance on another set of data samples. 

```
clf = TkNN(theta = 0.25,beta=2,metric='cosine')
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)
```

Here `X_train` is the training data and it is a numeric  array or matrix and has shapes '[n_samples, n_features]'. `y_train` is the class labels of individual samples in X_train. Similarly, `X_test` is the test data and it is also an array or matrix and has shapes '[n_samples, n_features]'. The following options of distance metrics are available: 'cosine', 'chebyshev', 'cityblock', 'euclidean', 'minkowski' and the `default` is `cosine distance`. `Beta` is the threshold on majority voting and `theta` is the threshold on similarity between two data points.

## Contact

For any further query, comment or suggestion, you may reach out to me at welcometanmay@gmail.com

## Citing
```
@inproceedings{basu12tknn,
 author    = "T. Basu and C. A. Murthy  and H. Chakraborty ",
 title     = "A Tweak on K-Nearest Neighbor Decision Rule ",
 year      = "2012 ",
 pages     = "929-935 ",
 editor    = " ",
 booktitle = "Proceedings of the International Conference on Image Processing, Computer Vision, and Pattern Recognition",
 address   = "Las Vegas, USA ",
 publisher = " "
}

@Article{basu14tknntext ,
  author = 	 "T. Basu and C. A. Murthy ",
  title = 	 "Towards Enriching the Quality of k-Nearest Neighbor Rule for Document Classification ",
  journal =	 "International Journal of Machine Learning and Cybernetics",
  year =	 "2014 ",
  volume =	 "5 ",
  number =	 "6 ",
  pages =	 "897-905 "
}
```
