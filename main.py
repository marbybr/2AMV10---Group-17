
import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import dash_bootstrap_components as dbc # Used for creating more advanced layouts
import plotly.graph_objects as go
import os

#Load the dataset
df = pd.read_csv('data_cleaned.csv')

# Map Mental_Health_History to numeric
df["Mental_Health_History"] = df["Mental_Health_History"].map({"No": 0, "Maybe": 1, "Yes": 2})

# Set default name for `Country` column
COUNTRY_COL = "Country"

# Specify the columns you want to include in the dropdown
target_columns = ['family_history', 'treatment', 'Mental_Health_History']

#Initialize the app and incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SLATE] #[dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Mental Health Dataset Analysis"

# App layout
app.layout = dbc.Container([
    #Construct the title and subtitles
    dbc.Row([
        html.Div('Mental Health Dataset Analysis', style={
            'fontSize': '36px', 
            'fontWeight': 'bold', 
            'textAlign': 'center', 
            'marginBottom': '20px'
        })
    ]),

    dbc.Row([
        html.Div('Multiple Features Selection', style={
            'fontSize': '24px', 
            'fontWeight': 'normal', 
            'textAlign': 'center', 
            'marginBottom': '20px'
        })
    ]),

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
    
    #Construct the subtitle
    dbc.Row([
        html.Div('Feature Selection per country', style={
            'fontSize': '24px', 
            'fontWeight': 'normal', 
            'textAlign': 'center', 
            'marginTop': '20px', 
            'marginBottom': '20px'
        })
    ]),

    #Construct the drop down menu for the map visualization and the histogram
    dbc.Row([
        dcc.Dropdown(id='columns_dropdown',
                        options=[{'label': column, 'value': column} for column in target_columns if column not in ['Timestamp', "Country"]],
                        value='treatment',
                        style={'width':'100%'}
            ) #Dropdown
    ]),
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Graph(id='percentages_map', clear_on_unhover=True, style={'height': '70vh'})
        ], width=7),  # Map  
        dbc.Col([
            html.Br(),
            dcc.Graph(id='country_barplot', clear_on_unhover=True, style={'height': '70vh'})
        ], width=5)  # Histogram  
    ])
], fluid=True)

# Dropdown menu and bar chart
#Builds interaction between the displayed text below the dropdown menu and the graph
@app.callback(
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


# Helper functions for Map visualization
def group_by_country(df: pd.DataFrame, col: str):
    """Groups a dataframe by country and selects a column."""
    
    def inner(country_col="Country"):
        df_grouped = df.copy(deep=True)
        df_grouped = df_grouped.loc[:, [country_col, col]]
        df_grouped = df_grouped.groupby(country_col)[col].mean()
        df_grouped = df_grouped.reset_index()
        return df_grouped
    
    return inner

def find_closest(num, find_min=True, num_list=np.linspace(0, 1, 51)):
    """
    Find the closest number to a given percentage (between 0 and 1) that is smaller than that
    percentage (find_min=True) or greater than that percentage (find_min=False)
    """
    for i, num_ in enumerate(num_list):
        if find_min:
            lb = num_list[max(i-1, 0)]
            if lb >= num:
                return lb
        else:
            ub = num_
            if ub >= num:
                return ub

# Define callback function for updating map display
@app.callback(
    Output(component_id='percentages_map', component_property='figure'),
    Input(component_id='columns_dropdown', component_property='value')
)
def update_percentages_map(dropdown_value):
    data = group_by_country(df, dropdown_value)()

    # Load world map from Geopandas datasets
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the dataframe with the world GeoDataFrame
    world = world.merge(data, how='left', left_on='name', right_on=COUNTRY_COL)
    
    min_dropdown_value = find_closest(world[dropdown_value].min(), True)
    max_dropdown_value = find_closest(world[dropdown_value].max(), False)

    fig = px.choropleth(
        world, locations='iso_a3', color=dropdown_value,
        hover_name=COUNTRY_COL, hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'],
        color_continuous_scale=px.colors.sequential.Viridis_r,
        range_color=(min_dropdown_value, max_dropdown_value)
    )

    fig.update_layout(
        title=dict(text=f"{dropdown_value.title()} (%) per Country",
                   font=dict(size=30), automargin=True, yref='container',
                   y=0.95),
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=25)),
            tickfont=dict(size=20)
        )
    )

    return fig

# Callback for updating histogram
@app.callback(
    Output(component_id="country_barplot", component_property="figure"),
    Input(component_id="percentages_map", component_property="hoverData"),
    Input(component_id='columns_dropdown', component_property='value')
)
def update_barplot(hoverDataMap, dropdown_value):
    country = "United States of America"
    if hoverDataMap:
        country = hoverDataMap['points'][0]['customdata'][0]
    data = df.loc[df[COUNTRY_COL].isin([country]), :]

    fig = px.histogram(
        data_frame=data, 
        x=dropdown_value, 
        color=dropdown_value,
        hover_name=COUNTRY_COL, 
        hover_data=[COUNTRY_COL, dropdown_value],
        color_discrete_map={True: 'blue', False: 'red'}
    )
    fig.update_layout(
        title=dict(
            text=f"{dropdown_value} in {country}",
            font=dict(size=24), 
            automargin=True, 
            yref='container',
            y=0.95
        ), 
        xaxis_title=dict(
            text=f"{dropdown_value}",
            font=dict(size=16)
        ),
        yaxis_title=dict(
            text='Count',
            font=dict(size=16)
        ),
        legend_title=dict(
            text=f"{dropdown_value}",
            font=dict(size=16)
        ),
        font=dict(color='black'), 
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='white',
            zeroline=False,
            showline=True,
            linecolor='white'
        )
    )
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)