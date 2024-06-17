
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ceml.sklearn import generate_counterfactual

#Load the dataset
df = pd.read_csv('data_cleaned.csv')

# Set default name for `Country` column
COUNTRY_COL = "Country"

#Initialize the app and incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SLATE] #[dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Mental Health Dataset Analysis."

# App layout
app.layout = dbc.Container([
    #Construct the title and subtitles
    dbc.Row([
        html.Div('2AMV10, group 17: Mental Health Dataset Analysis', style={
            'fontSize': '15px', 
            'fontWeight': 'bold', 
            'textAlign': 'center', 
            'marginBottom': '10px'
        })
    ]),

    dbc.Row([
        dbc.Col([
            html.Div('Independant variables selection')
        ], width=4),
        dbc.Col([
            html.Div('Target variable selection')
        ], width=4),
        dbc.Col([
            html.Div('Filter selection')
        ], width=4)
    ]),

    #Construct the dropdown menu of all columns except for index, Timestamp and country, 
    #set the 2nd column (so first after index and Timestamp) as the standard selected option
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(df.columns[1:].drop(['Country']), df.columns[1], id='feature-dropdown', multi=True)
        ],width=4),
        dbc.Col([
            dcc.Dropdown(id='columns_dropdown',
                        options=[{'label': column, 'value': column} for column in df.columns if column not in ['Timestamp', "Country"]],
                        value='treatment',
                        style={'width':'100%'})
        ],width=4),
        dbc.Col([
            dcc.Dropdown([f"{col}{sign}" for col in df.columns[1:] for sign in [' = True', ' = False']], id='filter-dropdown', multi=True)
        ],width=4)
    ]),

    #Construct a Row (visually speaking) with 3 columns, 1 for the map, 1 for the target distribution and 1 for the feature distribution
    dbc.Row([
        dbc.Col([
            #html.Br(),
            dcc.Graph(id='percentages_map', clear_on_unhover=True, style={'height': '50vh'})
            ], width=4),  # Map  

        dbc.Col([
            dcc.Graph(id='country_barplot', clear_on_unhover=True, style={'height': '50vh'})
            ], width=3),  # Histogram
        #Show a simple histogram for the selected feature
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
            ], width=5)
    ]),

    #####
    # Add a button to train the model and display feature importances
    dbc.Row([
        dbc.Col([
            # html.Div('Train model and Feature Importance', style={
            # 'fontSize': '20px', 
            # 'fontWeight': 'normal', 
            # 'textAlign': 'center', 
            # 'marginTop': '20px', 
            # 'marginBottom': '20px'
            # }),
            dbc.Button("Train Selected Features", id='train_button', color="primary", className="mr-2"),
            dcc.Graph(id='feature_importances', style={'height': '60vh'})
        ], width=6),

        dbc.Col([
            # html.Div('Counterfactual Explanations', style={
            # 'fontSize': '20px', 
            # 'fontWeight': 'normal', 
            # 'textAlign': 'center', 
            # 'marginTop': '20px', 
            # 'marginBottom': '20px'
            # }),

            #Replace the line below with the counterfactuals plot
                        #Replace the code below with the counterfactuals plot
            html.Div('placeholder for counterfactuals plot', style={
            'fontSize': '60px', 
            'fontWeight': 'normal', 
            'textAlign': 'center', 
            'marginTop': '20px', 
            'marginBottom': '20px'
            })
        ], width=6)
    ]),
], fluid=True)

# Dropdown menu and bar chart
#Builds interaction between the table, filters, bar charts and world map
@app.callback(
    #####
    Output(component_id='feature-distribution', component_property='figure'),
    Output(component_id='percentages_map', component_property='figure'),
    Output(component_id="country_barplot", component_property="figure"),
    Output(component_id='feature_importances', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value'),
    Input(component_id='filter-dropdown', component_property='value'),
    Input(component_id='columns_dropdown', component_property='value'),
    Input(component_id="percentages_map", component_property="hoverData"),
    Input(component_id='columns_dropdown', component_property='value'),
    Input(component_id='train_button', component_property='n_clicks')
)
def update_values(selected_features, filters, dropdown_value, hoverDataMap, dropdown_value_hist, selected_n_clicks):

    #Generate the text message that shows which features were selected
    if type(selected_features) == str:
        selected_features = [selected_features]

    #Create filtered dataset based on the 2nd dropdown and make the figure, model etc. with that filtered dataset
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
        # fig2 = go.Figure()
        # fig3 = go.Figure() 
        return fig #, fig2, fig3

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
        title=dict(
            text="Distributions of selected features",
            font=dict(size=20), 
            automargin=True, 
            yref='container',
            y=0.95
            ),
        xaxis_title="Variable",
        yaxis_title="Ratio",
        legend_title="Value",
        xaxis_tickangle = -45,
        height = 375,
        bargap = 0.5, #Space between groups
        bargroupgap = 0.5, #Space within groups
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
    )
    fig.update_yaxes(range=[0, 1])

    #Now we make the worldmap plot
    data = group_by_country(df_filtered, dropdown_value)()

    # Load world map from Geopandas datasets
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the dataframe with the world GeoDataFrame
    world = world.merge(data, how='left', left_on='name', right_on=COUNTRY_COL)
    
    min_dropdown_value = find_closest(world[dropdown_value].min(), True)
    max_dropdown_value = find_closest(world[dropdown_value].max(), False)

    fig2 = px.choropleth(
        world, locations='iso_a3', color=dropdown_value,
        hover_name=COUNTRY_COL, hover_data=[COUNTRY_COL, dropdown_value, 'continent', 'pop_est'],
        color_continuous_scale=px.colors.sequential.Viridis_r,
        range_color=(min_dropdown_value, max_dropdown_value)
    )

    fig2.update_layout(
        title=dict(text=f"{dropdown_value.title()} (%) per Country",
                   font=dict(size=20), automargin=True, yref='container',
                   y=0.95),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=25)),
            tickfont=dict(size=20)
        )
    )

    #Now we update the histogram associated with the worldmap plot
    country = "United States of America"
    if hoverDataMap:
        country = hoverDataMap['points'][0]['customdata'][0]
    data = df_filtered.loc[df_filtered[COUNTRY_COL].isin([country]), :]

    fig3 = px.histogram(
        data_frame=data, 
        x=dropdown_value_hist, 
        color=dropdown_value_hist,
        hover_name=COUNTRY_COL, 
        hover_data=[COUNTRY_COL, dropdown_value_hist],
        color_discrete_map={True: 'blue', False: 'red'}
    )
    fig3.update_layout(
        title=dict(
            text=f"{dropdown_value_hist} in {country}",
            font=dict(size=20), 
            automargin=True, 
            yref='container',
            y=0.95
        ), 
        xaxis_title=dict(
            text=f"{dropdown_value_hist}",
            font=dict(size=16)
        ),
        yaxis_title=dict(
            text='Count',
            font=dict(size=16)
        ),
        legend_title=dict(
            text=f"{dropdown_value_hist}",
            font=dict(size=16)
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
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

    #### 
    if selected_n_clicks is None:
        feature_importances_fig = px.bar(height = 250)
        return fig, fig2, fig3, feature_importances_fig
    
    # train model 
    df_train = df_filtered[selected_features]
    target = 'treatment'  
    X, clf = train(df_filtered, df_train, target, selected_features)

    #feature importance
    importances = clf.coef_[0]
    feature_importances = pd.Series(importances, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=True)
    #feature_importances = feature_importances[feature_importances >= 0]

    feature_importances_fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index,
                                     labels={'x': 'Importance', 'y': 'Feature'},
                                     title='Feature Importances',
                                     height = 250)

    feature_importances_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white')
    )
    return fig, fig2, fig3, feature_importances_fig


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

# train a model
def train(df_filtered, df_train, target, selected_features):
    # Preprocess data for logistic regression
    if target in selected_features: 
        X = df_train.drop(columns=[target])
    else:
        X = df_train
    y = df_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
    # print(f"Accuracy {accuracy_score(y_test, y_pred)}")
    return X, clf

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)