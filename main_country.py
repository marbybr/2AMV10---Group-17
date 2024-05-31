#by country
from dash import Dash, html, dcc, Input, Output, callback_context
import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash_bootstrap_components as dbc

# Load the dataset
df = pd.read_csv(r'data_cleaned.csv')

# Set default name for `Country column`
COUNTRY_COL = "Country"

def group_by_country(df: pd.DataFrame, col: str):
    """Groups a dataframe by country and selects a column."""
    def inner(country_col=COUNTRY_COL):
        df_grouped = df.copy(deep=True)
        df_grouped = df_grouped.loc[:, [country_col, col]]
        df_grouped = df_grouped.groupby(country_col)[col].mean()
        df_grouped = df_grouped.reset_index()
        return df_grouped
    return inner

# Create app layout
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Mental Health Dataset Analysis"

app.layout = html.Div([

    # Dropdown menu to select column to be displayed in map
    html.Div([
        dcc.Dropdown(id='columns_dropdown',
                     options=[{'label': column, 'value': column} for column in df.columns if column not in ['Timestamp', COUNTRY_COL]],
                     value='treatment',
                     style={'width':'50%', 'display': 'inline-block'}
        ),        
    ]),

    html.Br(),

    # Dropdown menu to select a region
    html.Div([
        dcc.Dropdown(id='regions_dropdown',
                     options=[{'label': region, 'value': region} for region in df[COUNTRY_COL].unique()],
                     value=df[COUNTRY_COL].unique()[0],
                     style={'width':'50%', 'display': 'inline-block'}
        ),        
    ]),

    html.Br(),

    # Figure -- Map
    dcc.Graph(id='percentages_map', clear_on_unhover=True, 
              style={'width': '90vw', 'height': '80vh', 'display': 'inline-block'}),

    html.Br(),

    # Placeholder for additional visualizations based on selected region
    html.Div(id='region-output', style={'width': '90vw', 'height': '50vh', 'display': 'inline-block'})
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

    # Create the choropleth map
    fig = px.choropleth(
        world, locations='iso_a3', color=dropdown_value,
        hover_name=COUNTRY_COL, hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'],
        color_continuous_scale=px.colors.sequential.Viridis_r
    )
    fig.update_layout(
        title=dict(text=f"{dropdown_value.title()} (%) per Country", 
                   font=dict(size=30), automargin=True, yref='container', y=0.95)
    )

    return fig

# Define combined callback for updating region details display and dropdown based on map click
@app.callback(
    [Output(component_id='region-output', component_property='children'),
     Output('regions_dropdown', 'value')],
    [Input(component_id='regions_dropdown', component_property='value'),
     Input(component_id='columns_dropdown', component_property='value'),
     Input('percentages_map', 'clickData')]
)
def update_region_details(selected_region, selected_column, clickData):
    ctx = callback_context
    if ctx.triggered:
        triggered_by = ctx.triggered[0]['prop_id']
        if 'clickData' in triggered_by and clickData:
            selected_region = clickData['points'][0]['location']
    
    if not selected_region:
        return [dcc.Graph(), selected_region]
    
    region_data = df[df[COUNTRY_COL] == selected_region]
    fig = px.histogram(region_data, x='Gender', color=selected_column, barmode='group')
    fig.update_layout(
        title=dict(text=f"{selected_column.title()} in {selected_region}", 
                   font=dict(size=30), automargin=True, yref='container', y=0.95)
    )
    return [dcc.Graph(figure=fig), selected_region]

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
