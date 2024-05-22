#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install dash-bootstrap-components
#pip install pandas

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc #Used for creating more advanced layouts

#Load the dataset
df = pd.read_csv('data_cleaned.csv')

#Initialize the app and incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SLATE]
app = Dash(__name__, external_stylesheets=external_stylesheets)

#App layout
app.layout = dbc.Container([
    #Construct the title
    dbc.Row([
        html.Div('Initial version of our dash application')
    ]),

    #Construct the dropdown menu of all columns except for index and Timestamp, 
    #set the 3nd column (so first after index and Timestamp) as the standard selected option
    dbc.Row([
        dcc.Dropdown(df.columns[2:], df.columns[2], id='feature-dropdown'),
        html.Div(id='feature-dropdown-container')
    ]),

    #Construct a Row (visually speaking) with 2 columns, 1 for the dataset, 1 for the plot
    dbc.Row([
        #Show thethe Time, treatment and history features of the first 10 rows the the dataset (probably not needed in final product)
        dbc.Col([
            dash_table.DataTable(data=df[['Timestamp', 'family_history', 'treatment']].to_dict('records'), page_size=10, id='table')
        ], width=6),
        #Show a simple histogram for the selected feature
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
        ], width=6)
    ])
])

#Builds interaction between the displayed text below the dropdown menu and the graph
@callback(
    Output(component_id='feature-dropdown-container', component_property='children'),
    Output(component_id='feature-distribution', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value'),
)

def update_values(selected_feature):
    printed_text = 'You have selected {}'.format(selected_feature)
    fig = px.histogram(df, x=selected_feature)
    return printed_text, fig


if __name__ == '__main__':
    app.run(debug=True)