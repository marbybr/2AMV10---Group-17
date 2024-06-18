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
import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers

#Load the dataset
df = pd.read_csv('data_cleaned.csv')
df["Mental_Health_History"] = df['Mental_Health_History'].map({'No': 0, 'Maybe': 0, 'Yes': 1})

# Set default name for `Country` column
COUNTRY_COL = "Country"

# Get mutable and immutable features
immutable_features = ['Australia_and_New_Zealand', 'Central_America', 'Eastern_Europe', 'Northern_America', 'Northern_Europe', 
                      'South_America', 'Southeastern_Asia', 'Southern_Africa', 'Southern_Asia', 'Southern_Europe', 'Western_Africa', 
                      'Western_Asia', 'Western_Europe', 'Female', 'family_history', 'Mental_Health_History']

mutable_features = [feature for feature in df.columns if feature not in immutable_features if feature != 'treatment' 
                    if feature not in ['Timestamp', "Country"]]

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
            dcc.Dropdown(df.columns[1:].drop(['Country']), mutable_features[0], id='feature-dropdown', multi=True)
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
            dcc.Graph(id='percentages_map', clear_on_unhover=True, style={'height': '70vh'})
            ], width=7),  # Map  

        dbc.Col([
            dcc.Graph(id='country_barplot', clear_on_unhover=True, style={'height': '50vh'})
            ], width=2),  # Histogram
        #Show a simple histogram for the selected feature
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
            ], width=3)
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
            dbc.Button("Train Full Dataset", id='train_full_button', color="primary", className="mr-2"),
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
            # dcc.Dropdown(id="counterfactuals_dd",
            #              multi=True),
            dcc.Dropdown([f"{col}{sign}" for col in df.columns[1:] for sign in [' = True', ' = False']], 
                         id='counterfactuals_dd', multi=True),
            # html.Div('placeholder for counterfactuals plot', style={
            # 'fontSize': '60px', 
            # 'fontWeight': 'normal', 
            # 'textAlign': 'center', 
            # 'marginTop': '20px', 
            # 'marginBottom': '20px'
            # })
            dcc.Graph(id="cf_barplot", style={'height': '60vh'})
        ], width=6)
    ]),
], fluid=True)

# # Callback for updating counterfactuals dropdown menu
# @app.callback(
#         Output(component_id="counterfactuals_dd", component_property="options"),
#         Input(component_id='feature-dropdown', component_property='value')
# )

# def update_counterfactual_dropdown(value):
#     # Set value to list if not already
#     if isinstance(value, str):
#         value = [value]
#     return [{'label': v, 'value': v} for v in value if v in mutable_features]

# Dropdown menu and bar chart
#Builds interaction between the table, filters, bar charts and world map
@app.callback(
    #####
    Output(component_id='feature-distribution', component_property='figure'),
    Output(component_id='percentages_map', component_property='figure'),
    Output(component_id="country_barplot", component_property="figure"),
    Output(component_id='feature_importances', component_property='figure'),
    Output(component_id='cf_barplot', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value'),
    Input(component_id='filter-dropdown', component_property='value'),
    Input(component_id='columns_dropdown', component_property='value'),
    Input(component_id="percentages_map", component_property="hoverData"),
    Input(component_id='columns_dropdown', component_property='value'),
    Input(component_id='train_button', component_property='n_clicks'),
    Input(component_id='train_full_button', component_property='n_clicks'),
    Input(component_id='counterfactuals_dd', component_property='value')
)
def update_values(selected_features, filters, dropdown_value, hoverDataMap, dropdown_value_hist, selected_n_clicks, full_n_clicks,
                  cf_filters):

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
        fig2 = go.Figure()
        fig3 = go.Figure()
        feature_importances_fig = go.Figure()
        fig_cf = go.Figure()
        return fig, fig2, fig3, feature_importances_fig, fig_cf

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
            showlegend = False,
        ))
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_1']],
            name="True",
            marker_color='blue',
            showlegend = False
        ))

    fig.add_trace(go.Bar(
        x=[None],  # Dummy value
        y=[None],  # Dummy value
        name="= False",
        marker_color='red',
    ))
    fig.add_trace(go.Bar(
        x=[None],  # Dummy value
        y=[None],  # Dummy value
        name="= True",
        marker_color='blue'
    ))
    
    # Update layout
    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Distributions of selected features",
            font=dict(size=15), 
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
        font=dict(color='white',
                  size=8)
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
            title=" ",
            tickfont=dict(size=10),
            tickformat=".0%",
            lenmode='fraction',
            len=0.9
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
            font=dict(size=10), 
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
    if selected_n_clicks is None and full_n_clicks is None:
        # return ""
        ###new
        feature_importances_fig = px.bar(height = 250)
        fig_cf = go.Figure()
        return fig, fig2, fig3, feature_importances_fig, fig_cf
    
    # df for training
    # df_train = df_filtered[selected_features]
    if selected_n_clicks:
        df_train = df_filtered[selected_features]
    elif full_n_clicks:
        df_train = df_filtered.drop(['Country', 'Timestamp'], axis=1)

    target = 'treatment'  
    X, clf, X_test, y_test = train(df_filtered.drop(['Country', 'Timestamp'], axis=1).astype(int), 
                                   df_train.astype(int), target, selected_features)

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

    # Lastly, the counterfactuals
    query_instance = {}
    
    # Return empty figure if no filters are provided
    if cf_filters is None:
        fig_cf = go.Figure()
        return fig, fig2, fig3, feature_importances_fig, fig_cf

    for cf_filter in cf_filters:
        column, value = cf_filter.split(' = ')
        if value == 'True':
            query_instance[column] = [1]
        else:
            query_instance[column] = [0]
    
    # Add dummy values for query instance
    for column in df_train.columns.values:
        if column not in list(query_instance.keys()):
            query_instance[column] = [0]
    query_instance = pd.DataFrame(query_instance)
    
    try:
        features_to_vary = [feature for feature in query_instance.columns if feature in mutable_features]
        _, differences = get_counterfactuals_from_model(X=query_instance, model=clf, features_to_vary=features_to_vary, 
                                                        outcome_name=target, total_CFs=1)
        
        # For now get first row
        sample_row = differences[0][0]

        # Make plot of data
        plot_data = pd.DataFrame({
        'Feature': sample_row.columns,
        'Value': sample_row.values.flatten()
        })
        plot_data['Color'] = plot_data['Value'].map({-1: 'red', 1: 'blue'})

        # Create the horizontal bar plot
        fig_cf = px.bar(plot_data, x='Value', y='Feature', orientation='h', color='Color',
                    title='Feature Values', labels={'Value': 'Value', 'Feature': 'Feature'},
                    color_discrete_map={'red': 'red', 'blue': 'blue'})
        # Update legend title and labels
        fig_cf.update_layout(
            legend_title_text='What to change'
        )
        fig_cf.for_each_trace(lambda t: t.update(name={'red': '1 --> 0', 'blue': '0 --> 1'}[t.name]))
    except:
        fig_cf = go.Figure()

    return fig, fig2, fig3, feature_importances_fig, fig_cf


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
    return X, clf, X_test, y_test

def get_counterfactuals_from_model(X: pd.DataFrame, model, features_to_vary, outcome_name: str, 
                                   total_CFs: int = 5, backend: str = "sklearn", continous_features = [], 
                                   desired_class: str = "opposite") \
                                    -> dice_ml.counterfactual_explanations.CounterfactualExplanations:
    """
    Generates counterfactual explanations using the dice_ml package.

    Parameters
    ----------
    X: pd.DataFrame 
        A dataframe containing the train data
    y: pd.Series 
        A series containing the test data
    model: fitted sklearn model
        A fully trained model
    features_to_vary: list 
        A list of features in the train data which can be used to generate counterfactuals
    outcome_name: str
        Name of the target variable
    total_CFS: int, optional
        Total number of counterfactuals required (default is 5)
    backend: str, optional
        "TF1" ("TF2") for TensorFLow 1.0 (2.0), "PYT" for PyTorch implementations, 
            "sklearn" for Scikit-Learn implementations of standard
    continuous_features: list, optional
        List of features in X which are continuous (default is [])
    desired_class: str, optional
        Desired counterfactual class - can take 0 or 1. Default value
            is "opposite" to the outcome class of query_instance for binary classification.
    
    Returns
    -------
    cf: dice_ml.counterfactual_explanations.CounterfactualExplanations
        A CounterfactualExplanations object that contains the list of
            counterfactual examples per query_instance as one of its attributes.
    """
    # Set target (treatment) to be True
    y = pd.DataFrame({'treatment': [1]})

    # Prepare data for DiCE
    data_interface = dice_ml.Data(dataframe=pd.concat([X, y], axis=1), 
                         continuous_features=continous_features, outcome_name=outcome_name)
    
    # Prepare model for DiCE
    model_interface = dice_ml.Model(model=model, backend=backend)

    # Initialize DiCE
    dice = dice_ml.Dice(data_interface = data_interface, model_interface=model_interface)

    # Set input data to be the query
    query_instances = X
    
    # Generate counterfactual explanations
    cf = dice.generate_counterfactuals(
        query_instances=X.iloc[0:],
        total_CFs=total_CFs,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        stopping_threshold=0.2,
        proximity_weight=1.0,
        sparsity_weight=1.0
    )

    # Get differences between query instances and counterfactuals
    def get_differences(cf, query_instances):
        """Returns the differences between the query instances and counterfactuals."""
        # Store differences in array
        differences = []
        for cf_, query_instance in zip(cf.__dict__['_cf_examples_list'], query_instances.to_numpy()):
            # Multiply by -1 because we want to know the change from the original entry
            difference = (cf_.final_cfs_df.to_numpy()[:, :-1] - query_instance) * -1
            difference = np.where(difference == -0., 0, difference)
            differences.append(difference)
        return differences
    
    # Get differences as dataframes
    differences = get_differences(cf, query_instances)
    differences_as_df = []

    for difference in differences:
        difference_as_df = []
        for row_difference in difference:
            diff_as_df = pd.DataFrame({X.columns[i]:  [row_difference[i]] for i in range(len(X.columns))})
            difference_as_df.append(diff_as_df)
        differences_as_df.append(difference_as_df)
    
    return cf, differences_as_df

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)