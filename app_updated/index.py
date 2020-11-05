import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
from apps.clean_data import clean_data
from dash.dependencies import Output, Input, State 
from app import app


df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',low_memory=False)
df = clean_data(df)
df = df.select_dtypes(include=["float"])

app.layout = html.Div([
    dcc.Location(id='sub_app_url'),
    html.Div(id="layouts")
])

index = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("overview",href="overview")),
            dbc.NavItem(dbc.NavLink("data_analysis",href="data_analysis"))
        ]
    ,
    brand = "ECS 289H Assignment 2",
    brand_href = "index",
    color = "primary",
    dark = True),
    html.H1("correlation analysis", className = "header"),
        dbc.Row(dbc.Col(
    html.Label(
        [
            "feature selector",
            dcc.Dropdown(
                    id="feature-selection-dropdown",
                    options = list(map(lambda x:{'label':x,'value':x},df.columns)),
                    placeholder = "select features...",
                    clearable = False,
                    value = [df.columns[0]],
                    multi = True                
                    )
        ],
        id = "feature-selection-dropdown-label"
    )
    )),
    dcc.Graph(id="correlation-analysis-output")

    ]
)


@app.callback(Output("layouts","children"),
[
    Input("sub_app_url","pathname")
])
def router(pathname):
    if pathname == "/overview":
        return overview.layout
    if pathname == "/data_analysis":
        return data_analysis.layout
    
    return index


@app.callback(
    Output("correlation-analysis-output","figure"),
    Input("feature-selection-dropdown","value")
)
def update_correlation_matrix(value):
    if not isinstance(value,list):
        value = [value]
    corr_df = df.loc[:,value].corr()
    fig = px.imshow(corr_df)
    fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="LightSteelBlue",
            title = "correlation matrix"
            )
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)