#If this is the first time you're running this file, first install dash and pandas by executing the following commands in the terminal:
#pip install dash
#pip install dash-bootstrap-components
#pip install pandas

# from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import dash_bootstrap_components as dbc #Used for creating more advanced layouts
import plotly.graph_objects as go

#Load the dataset
df = pd.read_csv('data_cleaned.csv')

# # Get sample of the dataset
# df_sample = df.sample(1, random_state=0, replace=False)

# fig = go.Figure(data=
#     go.Parcoords(
#         line = dict(color = df['Gender'],
#                    colorscale = [[0,'purple'],[1,'gold']]),
#         dimensions = list([
#             dict(range = [0,1],
#                 constraintrange = [0,1],
#                 label = 'Family history', values = df['family_history']),
#             dict(range = [0,1],
#                 label = 'Coping struggles', values = df['Coping_Struggles']),
#             dict(range = [0,1],
#                 label = 'Treatment', values = df['treatment'])
#         ])
#     )
# )

# fig.update_layout(
#     plot_bgcolor = 'white',
#     paper_bgcolor = 'white'
# )

# fig.show()
# df_ = df.loc[:, [col for col in df.columns.values if col != "Timestamp"]]
# df_ = df_.sample(10000, random_state=0)
# df_tnse = TSNE(n_components=2, learning_rate='auto',
#                init='random', perplexity=30, random_state=0).fit_transform(df_.to_numpy())
# print(df_tnse.shape)
# plt.figure()
# plt.scatter(df_tnse[:, 0], df_tnse[:, 1])
# plt.show()

def group_by_country(df: pd.DataFrame, col: str):
    """Groups a dataframe by country and selects a column."""
    
    def inner(country_col="Country"):
        df_grouped = df.copy(deep=True)
        df_grouped = df_grouped.loc[:, [country_col, col]]
        df_grouped = df_grouped.groupby(country_col)[col].mean()
        df_grouped = df_grouped.reset_index()
        return df_grouped
    
    return inner

data = group_by_country(df, 'treatment')()

# Load world map from Geopandas datasets
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the dataframe with the world GeoDataFrame
world = world.merge(data, how='left', left_on='name', right_on='Country')
fig = px.choropleth(
    world, locations='iso_a3', color='treatment',
    hover_name='Country', hover_data=['Country', 'treatment', 'continent', 'pop_est'],
    color_continuous_scale=px.colors.sequential.Viridis_r
)
fig.show()    
