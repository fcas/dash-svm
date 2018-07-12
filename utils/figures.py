import colorlover as cl
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics


def serve_prediction_plot(model,
                          X_train,
                          X_test,
                          y_train,
                          y_test,
                          Z,
                          xx,
                          yy,
                          mesh_step,
                          colorscale_selected,
                          threshold):
    # Get train and test score from model
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()),
                abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]

    if colorscale_selected == 'default':
        cscale = [[0, 'rgb(178,24,43)'], [1, 'rgb(33,102,172)']]
    elif colorscale_selected == 'Viridis':
        cscale = "Viridis"
    else:
        colorscale_zip = zip(np.arange(0, 1.01, 1 / 8),
                             cl.scales['9']['div'][colorscale_selected])
        cscale = list(map(list, colorscale_zip))

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo='none',
        showscale=False,
        contours=dict(
            showlines=False,
        ),
        colorscale=cscale,
        opacity=0.9
    )

    trace1 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        name=f'Training Data (accuracy={train_score:.3f})',
        marker=dict(
            size=10,
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
            size=10,
            symbol='triangle-up',
            color=y_test,
            colorscale=bright_cscale,
            line=dict(
                width=1
            ),
        )
    )

    # Plot the threshold
    trace3 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo='none',
        contours=dict(
            showlines=False,
            type='constraint',
            operation='=',
            value=scaled_threshold,
        ),
        name=f'Threshold ({scaled_threshold:.3f})',
        line=dict(
            color='#222222'
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

    data = [trace0, trace1, trace2, trace3]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_roc_curve(model,
                    X_test,
                    y_test):
    decision_test = model.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode='line',
        name='Test Data',
    )

    layout = go.Layout(
        title=f'ROC Curve of Test Data (AUC = {auc_score:.3f})',
        xaxis=dict(
            title='False Positive Rate'
        ),
        yaxis=dict(
            title='True Positive Rate'
        ),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_pie_confusion_matrix(model,
                               X_test,
                               y_test,
                               threshold):
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)

    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fp, fn, tn]
    labels = ["True Positive",
              "False Positive",
              "False Negative",
              "True Negative"]
    trace0 = go.Pie(
        labels=labels,
        values=values,
        hoverinfo='label+percent',
        textinfo='label+value',
        sort=False,
    )

    layout = go.Layout(
        title=f'Confusion Matrix of Test Data',
        margin=dict(l=20, r=20, t=45, b=20),
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure
