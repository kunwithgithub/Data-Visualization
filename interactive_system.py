import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None
app = dash.Dash(__name__)

def clean_data(df):
    df = df.replace('?',np.nan)
    df_isna = pd.isna(df)
    num_rows_with_excessive_na = np.sum(np.sum(df_isna,axis=1)>(df_isna.shape[1]/2))
    excessive_na_row_percentage = num_rows_with_excessive_na/df_isna.shape[0]
    num_fields_with_excessive_na = np.sum(np.sum(df_isna,axis=0)>(df_isna.shape[0]/2))
    excessive_na_field_percentage = num_fields_with_excessive_na/df_isna.shape[1]
    num_fields_with_excessive_na
    field_contain_na = np.sum(df_isna,axis=0)>(df_isna.shape[1]/2)
    field_key = field_contain_na.keys()
    for each_key in ["iyear","gname","nkill","nwound","weaptype1_txt","nperps","individual","nkillus"]:
        if each_key in field_key:
            field_contain_na[each_key]=False
    df_drop_na_field = df.loc[:,field_contain_na!=True]
    fields_with_na = df_drop_na_field.columns[(np.sum(pd.isna(df_drop_na_field),axis=0)>0)]
    for each_col in fields_with_na:
        df_drop_na_field.loc[pd.isna(df_drop_na_field)[each_col],each_col] = round(np.mean(df_drop_na_field.loc[pd.isna(df_drop_na_field)[each_col]!=True,each_col]))
    df = df_drop_na_field
    return df


df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',low_memory=False)



features = ['gname','iyear', 'imonth', 'iday', 'extended', 'country', 'region', 'success', 'suicide', 'attacktype1', 'targtype1', 'nperps', 'weaptype1', 'nkill', 'nkillus', 'nwound']
main_df = clean_data(df).loc[:,features]
view = ["correlation analysis","overview","PCA"]



app.layout = html.Div([
    html.H1("ECS 289G Assignment 2", style={'text-align':"center"}),
    html.Div(
    [
        html.Label([
        "columns",
        dcc.Dropdown(
        id="column-selection-dropdown",
        options = list(map(lambda x:{'label':x,'value':x},main_df.columns)),
        placeholder = "select column...",
        clearable = False,
        value = main_df.columns[0]
    )
    ]),
    html.Label(
        [
            "view",
            dcc.Dropdown(
            id="data_analysis-method-selection-dropdown",
            options = list(map(lambda x:{'label':x,'value':x},view)),
            placeholder = "select a view...",
            clearable = False,
            value = view[0]
            )
        ]
    ),
    html.Label(
        [
            "PCA process visualization",
            dcc.Dropdown(
            id="pca_process-selection-dropdown",
            options = list(map(lambda x:{'label':x,'value':x},["pre-PCA","post-PCA"])),
            placeholder = "select a PCA process...",
            clearable = False,
            value = "pre-PCA"
            )
        ]
    )
    ]
    ,
    id = "flex-container"),
    dcc.Graph(id="dropdown-output")

])

@app.callback(
    dash.dependencies.Output("dropdown-output","figure"),
    [dash.dependencies.Input("column-selection-dropdown","value"),
     dash.dependencies.Input("data_analysis-method-selection-dropdown","value")
    ]
)
def update_figure(column_selected,view_selected):
    if view_selected == "overview":
        fig = px.histogram(main_df,x=column_selected)
        fig.update_layout(transition_duration=500)
        return fig
    elif view_selected == "correlation analysis":
        corr_df = main_df.corr()
        fig = px.imshow(corr_df)
        fig.update_layout(transition_duration=500)
        return fig
    elif view_selected == "PCA":
        print(view_selected)
        fig = px.scatter_matrix(main_df,dimensions=['attacktype1', 'targtype1', 'weaptype1', 'nkill', 'nkillus', 'nwound'])
        fig.update_layout(transition_duration=500)
        return fig

@app.callback(
    [dash.dependencies.Output("column-selection-dropdown","disabled"),
    dash.dependencies.Output("pca_process-selection-dropdown","disabled")],
    [
        dash.dependencies.Input("data_analysis-method-selection-dropdown","value")
    ]
)
def update_column_selection_dropdown(data_analysis_method_selection):
    if data_analysis_method_selection == "overview":
        return False, True
    elif data_analysis_method_selection == "PCA":
        return True,False
    else:
        return True,True


if __name__ == "__main__":
    app.run_server(debug=True)