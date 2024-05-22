#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install pandas

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

#Load the dataset
df = pd.read_csv('data_cleaned.csv')
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

#Initialize the app
app = Dash()

#App layout
app.layout = [
    html.Div(children='Test'),
    html.Hr(),
    dcc.RadioItems(options=['Gender','treatment', 'family_history'], value='Gender', id='controls-and-radio-items'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    dcc.Graph(figure={}, id='controls-and-graph')
    #dcc.Graph(figure=px.histogram(df, x='Gender'))
    ]


#Add controls to build interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-radio-items', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df, x=col_chosen)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
