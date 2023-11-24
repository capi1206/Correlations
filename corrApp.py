import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go  # Import plotly.graph_objs
import math
import statsmodels.api as sm
from datetime import datetime
import numpy as np

# Read your data file into a DataFrame (df)
df = pd.read_excel("data/Correlación IPC TRM.xlsx", sheet_name="Hoja2", header=2)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")
df["TRM_d1"]=(df["TRM"]-df['TRM'].shift(-1))/df['TRM'].shift(-1)
df["TRM_d2"]=(df["TRM"]-df['TRM'].shift(-2))/(df['TRM'].shift(-2))
df["TRM_d3"]=(df["TRM"]-df['TRM'].shift(-3))/(df['TRM'].shift(-3))
df["TRM_d4"]=(df["TRM"]-df['TRM'].shift(-4))/(df['TRM'].shift(-4))
df["TRM_d5"]=(df["TRM"]-df['TRM'].shift(-5))/(df['TRM'].shift(-5))
dicc={1:"TRM_d1",2:"TRM_d2",3:"TRM_d3", 4:"TRM_d4", 5:"TRM_d5"}
dicc2={1:"sIPC",2:"IPC2",3:"IPC3", 4:"IPC4", 5:"IPC5"}
unique_dates = df["Fecha"].unique()
max_dt = df["Fecha"].max()
min_dt = df["Fecha"].min()





app = dash.Dash(__name__)


def coef_corr_trm_ipc(data, dif_trm):
    
    N = len(data)
    ipc_bar = data["sIPC"].mean()
    trm_bar = data[dicc[dif_trm]].mean()
    s_ipc, s_trm, coef = 0, 0, 0
    for _, row in data.iterrows():
        ipc_diff = row["sIPC"] - ipc_bar
        trm_diff = row[dicc[dif_trm]] - trm_bar
        s_ipc += ipc_diff**2
        s_trm += trm_diff**2
        coef += ipc_diff * trm_diff
    return coef / (N - 1) / (math.sqrt(s_ipc / (N - 1)) * math.sqrt(s_trm / (N - 1)))


app.layout = html.Div(
    [
        html.H1("Correlacion variación TRM vs IPC"),
         html.Div([
        html.Label("Lag"),
        dcc.Input(placeholder="Enter lag value", min=0,  # Set the minimum allowed value
            max=20,  # Set the maximum allowed value
            value=0, type="number", id="lag"),
    ]),
    html.Div([
        html.Label("Variación TRM"),
        dcc.Input(placeholder="Diferencia de TRM", min=1,  # Set the minimum allowed value
            max=5,  # Set the maximum allowed value
            value=1, type="number", id="dif_trm"),
    ]),
    html.Div([
        html.Label("Variación IPC"),
        dcc.Input(placeholder="Diferencia de IPC", min=1,  # Set the minimum allowed value
            max=5,  # Set the maximum allowed value
            value=1, type="number", id="dif_ipc"),
    ]),
        dcc.Dropdown(
            id="stdate",
            options=[
                {"label": str(date)[:10], "value": date}
                for date in unique_dates
            ],
            placeholder="Fecha de inicio.",
            value=min_dt,
        ),
        dcc.Dropdown(
            id="nddate",
            options=[
                {"label": str(date)[:10], "value": date}
                for date in unique_dates
            ],
            placeholder="Fecha de fin.",
            value=max_dt,
        ),
    #    dcc.Input(id="lag", type="number", placeholder="Lag", value=4),
    #    dcc.Input(id="dif_trm", type="number", placeholder="Variación TRM", value=1),
        dcc.Graph(id="trm-ipc-graph"),
        dcc.Graph(id="sipc-trm-graph"),
        html.Div(id="pearson-coefficient"),  # Add another graph component
    ]
)


@app.callback(
    [
        Output("trm-ipc-graph", "figure"),
        Output("sipc-trm-graph", "figure"),  # Output for the additional graph
        Output("pearson-coefficient", "children"),
    ],
    Input("stdate", "value"),
    Input("nddate", "value"),
    Input("lag", "value"),
    Input("dif_trm", "value"),
    Input("dif_ipc", "value"),
)
def update_graph(stdate, nddate, lag, dif_trm, dif_ipc):
    if stdate is None or nddate is None:
        # Handle the case when the user hasn't selected both start and end dates
        return go.Figure(), go.Figure(), "Pearson Coefficient: N/A"
    print(unique_dates)
    df["sIPC"] = df["IPC Mensual"].shift(lag)/100

    df_aux = df[(df["Fecha"] >= stdate) & (df["Fecha"] <= nddate)]
    df_aux = df_aux.dropna(subset=["sIPC"])
    df_aux = df_aux.dropna(subset=[dicc[dif_trm]])
    
    
    df_aux["IPC2"]=(1-df_aux['sIPC'])*(1-df_aux['sIPC'].shift(-1))-1
    df_aux["IPC3"]=(1-df_aux['sIPC'])*(1-df_aux['sIPC'].shift(-1))*(1-df_aux['sIPC'].shift(-2))-1
    df_aux["IPC4"]=(1-df_aux['sIPC'])*(1-df_aux['sIPC'].shift(-1))*(1-df_aux['sIPC'].shift(-2))*(1-df_aux['sIPC'].shift(-3))-1
    df_aux["IPC5"]=(1-df_aux['sIPC'])*(1-df_aux['sIPC'].shift(-1))*(1-df_aux['sIPC'].shift(-2))*(1-df_aux['sIPC'].shift(-3))*(1-df_aux['sIPC'].shift(-4))-1
    df_aux = df_aux.dropna(subset=[dicc2[dif_ipc]])

    X = df_aux[dicc2[dif_ipc]]
    X = sm.add_constant(X)  # Add a constant to the independent variable
    y = df_aux[dicc[dif_trm]]
    # Fit a linear regression model for TRM vs IPC
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)

    # Calculate MSE and Pearson's coefficient
    mse = ((y - y_pred) ** 2).mean()
    pearson_coefficient = coef_corr_trm_ipc(df_aux, dif_trm)

    # Create a scatter plot with a line for TRM vs IPC
    trm_ipc_trace = go.Scatter(
        x=df_aux[dicc2[dif_ipc]],
        y=y,
        mode="markers",
        marker=dict(size=4, color="red"),
        name="(IPC,var TRM)",
    )
    trm_ipc_line = go.Scatter(
        x=df_aux[dicc2[dif_ipc]],
        y=y_pred,
        mode="lines",
        line=dict(color="blue"),
        name="Regresion lineal (TRM vs IPC)",
    )

    # Create line plots for sIPC vs Fecha and TRM vs Fecha
    sipc_trace = go.Scatter(
        x=df_aux["Fecha"],
        y=df_aux[dicc2[dif_ipc]]  ,
        mode="lines",
        line=dict(color="green"),
        name="IPC (lag) x3000 vs Fecha",
    )
    trm_trace = go.Scatter(
        x=df_aux["Fecha"],
    y=df_aux[dicc[dif_trm]],
        mode="lines",
        line=dict(color="orange"),
        name="TRM vs Fecha",
    )

    layout_trm_ipc = go.Layout(
        title=f"TRM vs IPC Regresion lineal (Lag = {lag})",
        xaxis=dict(title=f"IPC Mensual (Corrida por {lag} meses)"),
        yaxis=dict(title=f"TRM"),
    )

    layout_sipc_trm = go.Layout(
        title="sIPC vs Fecha & TRM vs Fecha",
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="sIPC & TRM"),
    )

    figure_trm_ipc = go.Figure(
        data=[trm_ipc_trace, trm_ipc_line], layout=layout_trm_ipc
    )
    figure_sipc_trm = go.Figure(data=[sipc_trace, trm_trace], layout=layout_sipc_trm)

    return (
        figure_trm_ipc,
        figure_sipc_trm,
        f"Coeficiente de Pearson: {pearson_coefficient:.4f}, error medio cuadrado: {mse:.4f}",
    )


if __name__ == "__main__":
    app.run_server(debug=True)
