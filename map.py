#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install dash-bootstrap-components
#pip install pandas

import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import dash_bootstrap_components as dbc #Used for creating more advanced layouts
import plotly.graph_objects as go

#Load the dataset
df = pd.read_csv('data_cleaned.csv')

# Set default name for `Country column`
COUNTRY_COL = "Country"

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

# Create app layout
app = Dash()
app.title = "Mental Health Dataset Analysis"

app.layout = html.Div([

    # Dropdown menu to select column to be displayed in map
    html.Div([
        dcc.Dropdown(id='columns_dropdown',
                     options=[column for column in df.columns if column not in ['Timestamp', "Country"]],
                     value='treatment',
                     style={'width':'50%', 'display': 'inline-block'}
        ),        
    ]),

    html.Br(),

    # Figure -- Map
    dcc.Graph(id='percentages_map', clear_on_unhover=True, 
              style={'width': '70vw', 'height': '70vh', 'display': 'inline-block'}),
    
    # Figure -- Bar plot
    dcc.Graph(id='country_barplot', clear_on_unhover=True, style={'display': 'inline-block'}),
])

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
    # min_legend_range = 0 if min_dropdown_value < 0.25 else \
    #     (0.25 if 0.25 <= min_dropdown_value < 0.5 else 0.5)
    # max_legend_range = 0.5 if max_dropdown_value < 0.5 else \
    #     (0.75 if 0.5 <= max_dropdown_value <= 0.75 else 1)

    fig = px.choropleth(
        world, locations='iso_a3', color=dropdown_value,
        hover_name=COUNTRY_COL, hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'],
        color_continuous_scale=px.colors.sequential.Viridis_r,
        range_color=(min_dropdown_value, max_dropdown_value) # Inspired from https://community.plotly.com/t/fixed-legend-for-animation/56167
    )
    # fig.update_layout(
    #     title=dict(text=f"{dropdown_value.title()} (%) per Country", 
    #                font=dict(size=30), automargin=True, yref='container',
    #                y=0.95)
    # ) # Inspired from https://plotly.com/python/figure-labels/

    fig.update_layout(
        title=dict(text=f"{dropdown_value.title()} (%) per Country",
                   font=dict(size=30), automargin=True, yref='container',
                   y=0.95), # Inspired from https://plotly.com/python/figure-labels/
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=25)),  # Increase legend title font size
            tickfont=dict(size=20)  # Increase tick labels font size
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

    # Load world map from Geopandas datasets
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the dataframe with the world GeoDataFrame
    world = world.merge(data, how='left', left_on='name', right_on=COUNTRY_COL)

    fig = px.histogram(data_frame=world, x=dropdown_value, color=dropdown_value,
                       hover_name=COUNTRY_COL, 
                       hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'])
    fig.update_layout(
        title=dict(text=f"{dropdown_value} in {country}",
                   font=dict(size=30), automargin=True, yref='container',
                   y=0.95), # Inspired from https://plotly.com/python/figure-labels/
    )
    return fig
    

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)