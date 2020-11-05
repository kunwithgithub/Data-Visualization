#reference: http://liangfan.tech/2018/12/10/Python-Dash%E5%8D%81%E8%AE%B2%E4%B9%8B8-%E5%A4%9A%E9%A1%B5DASH-APP/

import dash
import dash_bootstrap_components as dbc
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True
app.title = "ecs 289H assignment 2"