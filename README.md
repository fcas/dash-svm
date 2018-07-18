# Support Vector Machine (SVM) Explorer

This is a demo of the Dash interactive Python framework developed by [Plotly](https://plot.ly/).

Dash abstracts away all of the technologies and protocols required to build an interactive web-based application and is a simple and effective way to bind a user interface around your Python code.

To learn more check out our [documentation](https://plot.ly/dash).

You can find the latest [dev version of the app here](https://dash-svm-dev.herokuapp.com/),
and the [official version here](https://dash-svm.herokuapp.com/).

<Start Description>

## Getting Started with the Demo

This demo lets you interactive explore Support Vector Machine (SVM). 

It includes a few artificially generated datasets that you can choose from the dropdown, and that you can modify by changing the sample size and the noise level of those datasets.

The other dropdowns and sliders lets you change the parameters of your classifier, such that it could increase or decrease its accuracy.

## How does it work?

This app is fully written in Dash + scikit-learn. All the components are used as input parameters for scikit-learn functions, which then generates a model with respect to the parameters you changed. The model is then used to perform predictions that are displayed on a contour plot, and its predictions are evaluated to create the ROC curve and confusion matrix.

In addition to creating models, scikit-learn is used to generate the datasets you see, as well as the data needed for the metrics plots.

## What is an SVM?

An SVM is a popular Machine Learning model used in many different fields. You can find an [excellent guide to how to use SVMs here](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

<End Description>

## Screenshots
![animated1](images/animated1.gif)
