"""
This app's data and plots are heavily inspired from the scikit-learn Classifier
comparison tutorial. Part of the app's code is directly taken from it. You can
find it here:
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
"""
import os
import time

import colorlover as cl
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import utils.dash_reusable_components as drc

app = dash.Dash(__name__)
server = app.server

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


def generate_data(dataset, noise):
    if dataset == 'moons':
        return datasets.make_moons(noise=noise, random_state=0)

    elif dataset == 'circles':
        return datasets.make_circles(noise=noise, factor=0.5, random_state=1)

    elif dataset == 'linear':
        X, y = datasets.make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            'Data type incorrectly specified. Please choose an existing '
            'dataset.')


app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        # Change App Name here
        html.Div(className='container scalable', children=[
            # Change App Name here
            html.H2('App Name'),

            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                drc.Card([
                    drc.NamedDropdown(
                        name='Select Dataset',
                        id='dropdown-select-dataset',
                        options=[
                            {'label': 'Moons', 'value': 'moons'},
                            {'label': 'Linearly Separable', 'value': 'linear'},
                            {'label': 'Circles', 'value': 'circles'}
                        ],
                        clearable=False,
                        searchable=False,
                        value='moons'
                    ),

                    drc.NamedSlider(
                        name='Noise Level',
                        id='slider-dataset-noise-level',
                        min=0,
                        max=1,
                        marks={i / 10: str(i / 10) for i in range(0, 11)},
                        step=0.1,
                        value=0.2,
                    ),
                ]),

                drc.Card([
                    drc.NamedDropdown(
                        name='Kernel',
                        id='dropdown-svm-parameter-kernel',
                        options=[
                            {'label': 'Radial basis function (RBF)',
                             'value': 'rbf'},
                            {'label': 'Linear', 'value': 'linear'},
                            {'label': 'Polynomial', 'value': 'poly'},
                        ],
                        value='rbf',
                        clearable=False,
                        searchable=False
                    ),

                    drc.NamedSlider(
                        name='Degree',
                        id='slider-svm-parameter-degree',
                        min=2,
                        max=10,
                        marks={i: i for i in range(2, 11, 2)},
                        step=1,
                        value=3,
                    ),

                    drc.NamedSlider(
                        name='Gamma',
                        id='slider-svm-parameter-gamma-power',
                        min=-3,
                        max=1,
                        value=-1,
                        marks={i: '{}'.format(10 ** i) for i in range(-3, 2)}
                    ),

                    drc.FormattedSlider(
                        style={'padding': '5px 10px 25px'},
                        id='slider-svm-parameter-gamma-coef',
                        min=1,
                        max=9,
                        value=5
                    )
                ])
            ]),

            html.Div(className='six columns', children=[
                dcc.Graph(id='graph-sklearn-svm', style={'height': '80vh'})
            ])
        ])
    ])
])


@app.callback(Output('slider-svm-parameter-gamma-coef', 'marks'),
              [Input('slider-svm-parameter-gamma-power', 'value')])
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10)}


@app.callback(Output('graph-sklearn-svm', 'figure'),
              [Input('dropdown-svm-parameter-kernel', 'value'),
               Input('slider-svm-parameter-degree', 'value'),
               Input('slider-svm-parameter-gamma-coef', 'value'),
               Input('slider-svm-parameter-gamma-power', 'value'),
               Input('dropdown-select-dataset', 'value'),
               Input('slider-dataset-noise-level', 'value')])
def update_svm_graph(kernel, degree, gamma_coef, gamma_power, dataset, noise):
    h = .02  # step size in the mesh

    # Data Pre-processing
    X, y = generate_data(dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    gamma = gamma_coef * 10 ** gamma_power

    clf = SVC(
        kernel=kernel,
        degree=degree,
        gamma=gamma
    )
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Colorscale
    bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]
    colorscale_zip = zip(np.arange(0, 1.01, 1 / 8),
                         cl.scales['9']['div']['RdBu'])
    cscale = list(map(list, colorscale_zip))

    # Create the plot
    Z = Z.reshape(xx.shape)

    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), h),
        y=np.arange(yy.min(), yy.max(), h),
        z=Z,
        hoverinfo='none',
        showscale=True,
        contours=dict(
            showlines=False
        ),
        colorscale=cscale
    )

    trace1 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        name=f'Training Data (accuracy={train_score:.3f})',
        marker=dict(
            color=y_train,
            colorscale=bright_cscale,
            line=dict(
                width=1
            )
        )
    )

    trace2 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode='markers',
        name=f'Test Data (accuracy={test_score:.3f})',
        marker=dict(
            color=y_test,
            colorscale=bright_cscale,
            line=dict(
                width=1
            ),
            opacity=0.6
        )
    )

    layout = go.Layout(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    data = [trace0, trace1, trace2]
    figure = go.Figure(data=data, layout=layout)

    return figure


external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet, replace this with your own base-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet, replace this with your own custom-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/638e683e245ea751bca62fd427e385ab/raw/fab9c525a4de5b2eea2a2b292943d455ade44edd/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
