#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install dash-bootstrap-components
#pip install pandas

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    #set the 2nd column (so first after index and Timestamp) as the standard selected option
    dbc.Row([
        dcc.Dropdown(df.columns[1:], df.columns[1], id='feature-dropdown', multi=True),
        html.Div(id='feature-dropdown-container')
    ]),

    #Construct a Row (visually speaking) with 2 columns, 1 for the dataset, 1 for the plot
    dbc.Row([
        #Show the Time, treatment and history features of the first 10 rows the the dataset (probably not needed in final product)
        dbc.Col([
            dash_table.DataTable(data=df[['Timestamp', 'family_history', 'treatment']].to_dict('records'), page_size=10, id='table')
        ], width=6),
        #Show a simple histogram for the selected feature
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
        ], width=6)
    ]),
    
    dbc.Row([
        dcc.Dropdown([f"{col}{sign}" for col in df.columns[1:] for sign in [' = True', ' = False']], id='filter', multi=True)
    ])

])

#Builds interaction between the displayed text below the dropdown menu and the graph
@callback(
    Output(component_id='feature-dropdown-container', component_property='children'),
    Output(component_id='feature-distribution', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value'),
)

def update_values(selected_features):

    #Generate the text message that shows which features were selected
    if type(selected_features) == str:
        selected_features = [selected_features]
    printed_text = 'You have selected the following feature(s): {}'.format(', '.join(selected_features))

    #Create a filtered dataset based on the 2nd dropdown and make the figure, model etc. with that filtered dataset










    #If no features were selected, return an empty figure
    if len(selected_features) == 0:
        fig = go.Figure()
        return printed_text, fig

    #Construct the updated barchart
    count_data = df[selected_features].apply(lambda x: x.value_counts(normalize=True)).T
    count_data.columns = ['count_0', 'count_1']
    count_data.reset_index(inplace=True)
    count_data.rename(columns={'index': 'variable'}, inplace=True)

    fig = go.Figure()

    for _, row in count_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_0']],
            name=f"{row['variable']} = 0",
            marker_color='blue'
            ))
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_1']],
            name=f"{row['variable']} = 1",
            marker_color='orange'
            ))
    
    # Update layout
    fig.update_layout(
        barmode='group',
        title="Counts of Binary Variables",
        xaxis_title="Variable",
        yaxis_title="Ratio",
        legend_title="Value"
    )
    fig.update_yaxes(range=[0, 1])

    return printed_text, fig


if __name__ == '__main__':
    app.run(debug=True)