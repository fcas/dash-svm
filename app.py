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


def get_linearly_separable():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    return linearly_separable


datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    get_linearly_separable
]

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
            html.Div(className='six columns', children=drc.Card([
                    drc.NamedDropdown(
                        name='Kernel',
                        id='dropdown-svm-parameter-kernel',
                        options=[
                            {'label': 'Radial basis function (RBF)', 'value': 'rbf'},
                            {'label': 'Linear', 'value': 'linear'},
                            {'label': 'Polynomial', 'value': 'poly'},
                        ],
                        value='rbf',
                        clearable=False,
                        searchable=False
                    )
            ])),

            html.Div(className='six columns', children=[
                dcc.Graph(id='graph-sklearn-svm', style={'height': '80vh'})
            ]),


        ])


    ])
])


@app.callback(Output('graph-sklearn-svm', 'figure'),
              [Input('dropdown-svm-parameter-kernel', 'value')])
def update_svm_figure(kernel):
    h = .02  # step size in the mesh

    # preprocess dataset, split into training and test part
    X, y = datasets[0]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    t_start = time.time()
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'SVC computed in {time.time() - t_start:.3f} seconds.')

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
        x=np.arange(xx.min(), xx.max(), 0.02),
        y=np.arange(yy.min(), yy.max(), 0.02),
        z=Z,
        hoverinfo='none',
        showscale=False,
        contours=dict(
            showlines=False
        ),
        colorscale=cscale
    )

    trace1 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        name='Training Data',
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
        name='Test Data',
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
        margin=dict(l=0,r=0, t=0, b=0)
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
