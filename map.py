#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install dash-bootstrap-components
#pip install pandas

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
              style={'width': '90vw', 'height': '80vh', 'display': 'inline-block'})
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

    # Set legend range
    if min(world[dropdown_value].values) < 0.5:
        min_legend_range = 0
    else:
        min_legend_range = 0.5
    
    min_dropdown_value = world[dropdown_value].min()
    max_dropdown_value = world[dropdown_value].max()
    min_legend_range = 0 if min_dropdown_value < 0.25 else \
        (0.25 if 0.25 <= min_dropdown_value < 0.5 else 0.5)
    max_legend_range = 0.5 if max_dropdown_value < 0.5 else \
        (0.75 if 0.5 <= max_dropdown_value <= 0.75 else 1)

    fig = px.choropleth(
        world, locations='iso_a3', color=dropdown_value,
        hover_name=COUNTRY_COL, hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'],
        color_continuous_scale=px.colors.sequential.Viridis_r,
        # range_color=(min_legend_range, max_legend_range) # Inspired from https://community.plotly.com/t/fixed-legend-for-animation/56167
    )
    fig.update_layout(
        title=dict(text=f"{dropdown_value.title()} (%) per Country", 
                   font=dict(size=30), automargin=True, yref='container',
                   y=0.95)
    ) # Inspired from https://plotly.com/python/figure-labels/

    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)