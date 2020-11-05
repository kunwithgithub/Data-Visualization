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

layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("overview",href="overview")),
            dbc.NavItem(dbc.NavLink("data_analysis",href="data_analysis")),
            dbc.NavItem(dbc.NavLink("home",href="index"))
        ]
    ,
    brand = "ECS 289H Assignment 2",
    brand_href = "index",
    color = "primary",
    dark = True
    ),
    html.H1("data analysis", className = "header"),
    dbc.Tabs(
        [
            dbc.Tab(label="pre-PCA", tab_id = "pre-PCA"),
            dbc.Tab(label="post-PCA",tab_id="post-PCA")
        ],
        id = "pca_process-selection-tabs"
    ),
    dbc.Row(
        dbc.Col(
        html.Label(
        [
            "Principle Component slider",
            dcc.Slider(
                id='principle-component-slider',
                min = 1,
                max = df.columns.size,
                step = 1,
                value = 1,
                marks = dict((i,{"label":str(i)}) for i in range(1,df.columns.size)),
            )
        ],
        id = "principle-component-slider-label"
    )
        )
    ),
    dbc.Row(dbc.Col(
    html.Label(
        [
            "PCA feature selector",
            dcc.Dropdown(
                    id="pca-feature-selection-dropdown",
                    options = list(map(lambda x:{'label':x,'value':x},df.columns)),
                    placeholder = "select features...",
                    clearable = False,
                    value = [df.columns[0]],
                    multi = True                
                    )
        ],
        id = "pca-feature-selection-dropdown-label"
    )
    )),
    dcc.Graph(id="data-analysis-output")
])


@app.callback(
    [Output("principle-component-slider","max"),
    Output("principle-component-slider","value"),
    Output("principle-component-slider","marks")],
    [
        Input("pca-feature-selection-dropdown","value")
    ]
)
def update_slider(value):
    print(value)
    if isinstance(value,list):
        return len(value)-1,1,dict((i,{"label":str(i)}) for i in range(1,len(value)))
    else:
        return 1,1

@app.callback(
    Output("data-analysis-output","figure"),
    [
        Input("pca-feature-selection-dropdown","value"),
        Input("principle-component-slider","value"),
        Input("pca_process-selection-tabs","active_tab")
    ]
)
def update_figure(features, n_components, process):
    if not isinstance(features,list):
        features = [features]
    if process == "pre-PCA":
        fig = px.scatter_matrix(df, dimensions = features)
        fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="LightSteelBlue",
                title="Pre-PCA scatter plots matrix"
        )
        return fig
    pca = PCA(n_components=n_components)
    sc = StandardScaler()
    print(df.columns)
    df_normalized = sc.fit_transform(df.loc[:,features])

    components = pca.fit_transform(df_normalized)
    labels = {
        str(i):f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_*100)
    }
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(n_components)
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="LightSteelBlue",
        title= "Post-PCA Components scatter total explained variance: "+str(pca.explained_variance_ratio_.sum())
    )
    return fig
