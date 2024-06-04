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
df_filtered = df.copy()

#Initialize the app and incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SLATE]
app = Dash(__name__, external_stylesheets=external_stylesheets)

#App layout
app.layout = dbc.Container([
    #Construct the title
    dbc.Row([
        html.Div('Select features for analysis in the dropdown menu below')
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
            html.Div(style={'height': '20px'}),
            html.Div('Select features you would like to view in the table in the dropdown menu below'),
            dcc.Dropdown(df.columns, ['Timestamp', 'family_history', 'treatment'], id='table-feature-selection', multi=True),
            dash_table.DataTable(page_size=10, id='table', style_table={'overflowX': 'auto'}),
            html.Div('The dropdown menu below can be used to filter the dataset'),
            dcc.Dropdown([f"{col}{sign}" for col in df.columns[1:] for sign in [' = True', ' = False']], id='filter-dropdown', multi=True)
            
        ], width=5),
        #Show a simple histogram for the selected feature
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
        ], width=7)
    ]),
])

#Builds interaction between the displayed text below the dropdown menu and the graph
@callback(
    Output(component_id='feature-dropdown-container', component_property='children'),
    Output(component_id='feature-distribution', component_property='figure'),
    Output(component_id='table', component_property='data'),
    Input(component_id='table-feature-selection', component_property='value'),
    Input(component_id='feature-dropdown', component_property='value'),
    Input(component_id='filter-dropdown', component_property='value')
)

def update_values(table_features, selected_features, filters):

    #Generate the text message that shows which features were selected
    if type(selected_features) == str:
        selected_features = [selected_features]
    printed_text = 'You have selected the following feature(s): {}'.format(', '.join(selected_features))

    #Create a filtered dataset based on the 2nd dropdown and make the figure, model etc. with that filtered dataset
    df_filtered = df.copy()
    if type(filters) == list:
        for i in filters:
            filter_column = i.split(' ')[0] #Select the column name
            if i.split(' ')[2] == 'True':
                filter_value = True
            else:
                filter_value = False
            df_filtered = df_filtered[df_filtered[filter_column] == filter_value]


    #If no features were selected, return an empty figure
    if len(selected_features) == 0:
        fig = go.Figure()
        return printed_text, fig

    #Construct the updated barchart
    count_data = df_filtered[selected_features].apply(lambda x: x.value_counts(normalize=True)).T
    if not True in count_data.columns:
        count_data[True] = 0
    if not False in count_data.columns:
        count_data[False] = 0
        count_data = count_data.reindex(columns=[False, True]) #re-orders the columns for consistency
    count_data.columns = ['count_0', 'count_1']
    count_data.reset_index(inplace=True)
    count_data.rename(columns={'index': 'variable'}, inplace=True)

    fig = go.Figure()
    legend_check = True
    
    for _, row in count_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_0']],
            name="False",
            marker_color='red',
            showlegend = legend_check,
            width=0.25
            ))
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_1']],
            name="True",
            marker_color='green',
            showlegend = legend_check,
            width=0.25
            ))
        legend_check = False
    
    # Update layout
    fig.update_layout(
        barmode='stack',
        title="Counts of Binary Variables",
        xaxis_title="Variable",
        yaxis_title="Ratio",
        legend_title="Value",
        xaxis_tickangle = -45,
        height = 600,
        bargap = 0.5, #Space between groups
        bargroupgap = 0.5 #Space within groups
    )
    fig.update_yaxes(range=[0, 1])

    table = df_filtered[table_features].to_dict('records')

    return printed_text, fig, table


if __name__ == '__main__':
    app.run(debug=True)