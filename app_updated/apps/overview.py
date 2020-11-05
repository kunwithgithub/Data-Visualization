import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from apps.clean_data import clean_data
from dash.dependencies import Output, Input, State 
from app import app


df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',low_memory=False)
df = clean_data(df)

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
    html.H1("Overview", className = "header"),
    dbc.Row(
        dbc.Col(html.Label([
        "features",
        dcc.Dropdown(
            id="feature-selection-dropdown",
            options = list(map(lambda x:{'label':x,'value':x},df.columns.tolist())),
            placeholder = "select column...",
            clearable = False,
            value = df.columns[0],
            style = {"width":"50vw", "margin-left":"1vw"}
        )
    ],style = {"width":"50vw", "margin-left":"5vw"}))
    ),
    dbc.Row(dbc.Col(
    html.Label([
        "target",
        dcc.Dropdown(
            id="target-selection-dropdown",
            options = list(map(lambda x:{'label':x,'value':x},df.columns.tolist()+["None"])),
            placeholder = "select column...",
            clearable = False,
            value = "None",
            style = {"width":"50vw", "margin-left":"1vw"}
        )
    ],style = {"width":"50vw", "margin-left":"5vw"})
    )),
    dcc.Graph(id="overview-output")
])


@app.callback(
    Output("overview-output","figure"),
    [
        Input("feature-selection-dropdown","value"),
        Input("target-selection-dropdown","value")
    ]
    
)
def update_figure(feature, target):
    if target!="None":
        fig = px.histogram(df,x=feature,color = target)
    else:
        fig = px.histogram(df,x=feature)
    fig.update_layout(
        margin=dict(l=20,r=20,t=40,b=10),
        paper_bgcolor= "LightSteelBlue",
        title = "overview for dimension: "+ feature
    )
    return fig
