import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
app = dash.Dash(__name__,external_stylesheets=["interactive_system.css"])

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



features = ['targtype1', 'weaptype1', 'nkill', 'nkillus', 'nwound']
target = 'attacktype1'
main_df = clean_data(df).loc[:,['targtype1', 'weaptype1', 'nkill', 'nkillus', 'nwound','attacktype1','gname']]
view = ["correlation analysis","overview","PCA"]



app.layout = html.Div([
    html.H1("ECS 289G Assignment 2", className = "header"),
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
    id = "control_console",className = "flex-container"),
        html.Label(
        [
            "Principle Component slider",
            dcc.Slider(
                id='principle-component-slider',
                min = 1,
                max = len(features),
                step = 1,
                value = 1
                
            )
        ],
        id = "principle-component-slider-label"
    ),
    html.Label(
        [
            "PCA feature selector",
            dcc.Dropdown(
                    id="pca-feature-selection-dropdown",
                    options = list(map(lambda x:{'label':x,'value':x},features)),
                    placeholder = "select features...",
                    clearable = False,
                    value = features[0],
                    multi = True
                )
        ],
        id = "pca-feature-selection-dropdown-label"
    ),
    dcc.Graph(id="dropdown-output")

])

@app.callback(
    dash.dependencies.Output("dropdown-output","figure"),
    [dash.dependencies.Input("column-selection-dropdown","value"),
     dash.dependencies.Input("data_analysis-method-selection-dropdown","value"),
     dash.dependencies.Input("pca_process-selection-dropdown","value"),
     dash.dependencies.Input("principle-component-slider","value"),
     dash.dependencies.Input("pca-feature-selection-dropdown","value")
    ]
)
def update_figure(column_selected,view_selected,pca_process_selected,n_principles_components,pca_features_selected):
    print(pca_features_selected)
    if not isinstance(pca_features_selected,list): 
        pca_features_selected = [pca_features_selected]
    if view_selected == "overview":
        print("here")
        fig = px.histogram(main_df,x=column_selected, color = "gname")
        fig.update_layout(
                transition_duration=100,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="LightSteelBlue",
                title ="Terrorist activities occurence histogram for: "+column_selected)
        return fig
    elif view_selected == "correlation analysis":
        corr_df = main_df.corr()
        fig = px.imshow(corr_df)
        fig.update_layout(
                transition_duration=100,                
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="LightSteelBlue",
                title = view_selected
                )
        return fig
    elif view_selected == "PCA":
        if pca_process_selected == "pre-PCA":
            fig = px.scatter_matrix(main_df,dimensions=pca_features_selected,color=target)
            fig.update_layout(
                transition_duration=100,                
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="LightSteelBlue",
                 title="Pre-PCA scatter plots matrix"
            )
            return fig
        elif pca_process_selected == "post-PCA":
            pca = PCA(n_components=n_principles_components)
            sc = StandardScaler()
            print(main_df.columns)
            main_df_normalized = sc.fit_transform(main_df.loc[:,pca_features_selected])

            components = pca.fit_transform(main_df_normalized)
            labels = {
                str(i):f"PC {i+1} ({var:.1f}%)"
                for i, var in enumerate(pca.explained_variance_ratio_*100)
            }
            fig = px.scatter_matrix(
                components,
                labels=labels,
                dimensions=range(n_principles_components),
                color=main_df[target]
            )
            fig.update_layout(
                transition_duration=100,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="LightSteelBlue",
                title= "Post-PCA Components scatter total explained variance: "+str(pca.explained_variance_ratio_.sum())
            )
            return fig

@app.callback(
    [
        dash.dependencies.Output("column-selection-dropdown","disabled"),
        dash.dependencies.Output("pca_process-selection-dropdown","disabled"),
        dash.dependencies.Output("principle-component-slider","disabled"),
        dash.dependencies.Output("pca-feature-selection-dropdown","disabled")
    ],
    [
        dash.dependencies.Input("data_analysis-method-selection-dropdown","value"),
        dash.dependencies.Input("pca_process-selection-dropdown","value")
    ]
)
def update_column_selection_dropdown(data_analysis_method_selection, pca_process_selection):
    if data_analysis_method_selection == "overview":
        return False, True, True, True
    elif data_analysis_method_selection == "PCA":
        if pca_process_selection == "pre-PCA":
            return True,False, True, False
        else:
            return True, False, False, False
    else:
        return True,True, True, True



@app.callback(
    dash.dependencies.Output("principle-component-slider","max"),
    dash.dependencies.Input("pca-feature-selection-dropdown","value")
)
def update_PC_slider(pca_features_selected):
    return len(pca_features_selected)-1

if __name__ == "__main__":
    app.run_server(debug=True)