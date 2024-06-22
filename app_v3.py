import numpy as np
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash_bootstrap_components as dbc # Used for creating more advanced layouts
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import dice_ml

# Load the dataset
df = pd.read_csv('data_cleaned.csv')
 
# Turn the Mental_Health_History categorical variable into binary variable
df["Mental_Health_History"] = df['Mental_Health_History'].map({'No': 0, 'Maybe': 0, 'Yes': 1})
    
# Set default name for the `Country` column
COUNTRY_COL = "Country"

# Get mutable and immutable features for the counterfactual feature selection
immutable_features = ['Australia_and_New_Zealand', 'Central_America', 'Eastern_Europe', 'Northern_America', 'Northern_Europe', 
                      'South_America', 'Southeastern_Asia', 'Southern_Africa', 'Southern_Asia', 'Southern_Europe', 'Western_Africa', 
                      'Western_Asia', 'Western_Europe', 'Female', 'family_history', 'Mental_Health_History']

mutable_features = [feature for feature in df.columns if feature not in immutable_features if feature != 'treatment' 
                    if feature not in ['Timestamp', "Country"]]

# Initialize the app and incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SLATE]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Mental Health Dataset Analysis."

# App layout
app.layout = dbc.Container([
    # Construct the title and subtitles
    dbc.Row([
        html.Div('2AMV10, group 17: Mental Health Dataset Analysis', style={
            'fontSize': '15px', 
            'fontWeight': 'bold', 
            'textAlign': 'center', 
            'marginBottom': '10px'
        })
    ]),

    # Construct the text above the 3 dropdown menus at the top of the dashboard
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

    # Construct the three dropdown menus at the top of the dashboard
    dbc.Row([
        # Construct the feature selection dropdown menu of all columns except for index, Timestamp and country, 
        # set the 2nd column (so first after index and Timestamp) as the standard selected option
        dbc.Col([
            dcc.Dropdown(df.columns[1:].drop(['Country']), mutable_features[0], id='feature-dropdown', multi=True)
        ],width=4),

        # Construct the target variable selection dropdown menu and set the standard to treatment
        dbc.Col([
            dcc.Dropdown(id='columns_dropdown',
                        options=[{'label': column, 'value': column} for column in df.columns if column not in ['Timestamp', "Country"]],
                        value='treatment',
                        style={'width':'100%'})
        ],width=4),

        # Construct the filter selection dropdown menu
        dbc.Col([
            dcc.Dropdown([f"{col}{sign}" for col in df.columns[1:] for sign in [' = True', ' = False']], id='filter-dropdown', multi=True)
        ],width=4)
    ]),

    # Display the worldmap plot, the  Histogram for target distribution and the feature distribution plot next to each other
    dbc.Row([

        dbc.Col([
            dcc.Graph(id='percentages_map', clear_on_unhover=True, style={'height': '70vh'})
            ], width=7),  # Worldmap plot  

        dbc.Col([
            dcc.Graph(id='country_barplot', clear_on_unhover=True, style={'height': '50vh'})
            ], width=2),  # Histogram for target distribution
         
        dbc.Col([
            dcc.Graph(figure={}, id='feature-distribution')
            ], width=3)   # Feature selection plot
    ]),

    # Display the Training button, the feature importance plot, the dropdown menu for counteractual 
    # feature selection and the counterfactuals plot
    dbc.Row([

        # Button to train the model and display feature importances
        dbc.Col([
            dbc.Button("Train Selected Features", id='train_button', color="primary", className="mr-2"),
            dcc.Graph(id='feature_importances', style={'height': '60vh'})
        ], width=6),

        # Dropdown menu and counterfactual graph
        dbc.Col([
            dcc.Dropdown(id="counterfactuals_dd",
                         multi=True),
            dcc.Graph(id="cf_barplot", style={'height': '60vh'})
        ], width=6)
    ]),
], fluid=True)

# Callback for updating counterfactuals dropdown menu
@app.callback(
        Output(component_id="counterfactuals_dd", component_property="options"),
        Input(component_id='feature-dropdown', component_property='value')
)

def update_counterfactual_dropdown(value):
    """Updates the possible features that can be selected in the dropdown menu for counterfactual feature selection

    Parameter:
    value: a list that contains the features that can be selected for the counterfactuals. If one feature is selected, value is a string

    Returns a dictionary containing the labels and values of the mutable features
    """
    # Set value to list if not already
    if isinstance(value, str):
        value = [value]
    return [{'label': v, 'value': v} for v in value if v in mutable_features]

# Build interaction between the remaining components
@app.callback(
    Output(component_id='feature-distribution', component_property='figure'), # Feature distribution plot, top right on dashboard
    Output(component_id='percentages_map', component_property='figure'), # Worldmap plot, top left on dashboard
    Output(component_id="country_barplot", component_property="figure"), # Target variable distribution plot, top center on dashboard
    Output(component_id='feature_importances', component_property='figure'), # Feature importance plot for logistic regression model, buttom left on dashboard
    Output(component_id='cf_barplot', component_property='figure'), # Counterfactual plot, buttom right on dashboard
    Input(component_id='feature-dropdown', component_property='value'), # Dropdown menu to select which features to train with, top left on dashboard
    Input(component_id='filter-dropdown', component_property='value'), # Dropdown menu to select filter(s), top right on dashboard
    Input(component_id='columns_dropdown', component_property='value'), # Dropdown menu to select the target variable, top centre on dashboard
    Input(component_id="percentages_map", component_property="hoverData"), # Worldmap plot, top left on dashboard
    Input(component_id='train_button', component_property='n_clicks'), # Button that starts logistic regression training, button left on dashboard
    Input(component_id='counterfactuals_dd', component_property='value') # Dropdown menu to select features for counterfactualsm buttom right on dashboard
)
def update_values(selected_features, filters, dropdown_value, hoverDataMap, selected_n_clicks, cf_features):
    """Constructs all plots using inputs from the dropdown menus
    
    Parameters:

    selected_features: list of feature(s) to train with. If one feature is selected selected_features is a string
    filters: list of filters. If one filter is selected, filters is a string
    dropdown_value: string of the target variable
    hoverDataMap: value that represents the country that the user is hovering their mouse over in the worldmap plot
    selected_n_clicks: value that represents whether the button to train the model has been clicked by the user
    cf_features: list of features to calculate the counterfactuals with. If one feature is selected, cf_features is a string
    
    Returns:

    fig: Barchart that shows the distributions of the features used to train the model
    fig2: Worldmap plot
    fig3: Histogram to show the target variable distribution for the country that was clicked last
    feature_importances_fig: Barchart that shows the feature importances for the model
    fig_cf: Barchart that shows the counterfactuals
    """

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

    #If no features were selected to train on, return empty figures
    if len(selected_features) == 0:
        fig = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()
        feature_importances_fig = go.Figure()
        fig_cf = go.Figure()
        return fig, fig2, fig3, feature_importances_fig, fig_cf

    #Construct the updated barchart for the distributions of the selected features
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

    # Add a stacked bar for each feature
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

    # Add the legend such that it only shows red and blue once, no matter the number of selected features
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

    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Distributions of selected features",
            font=dict(size=10), 
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


    # Collect data for the worldmap plot
    data = group_by_country(df_filtered, dropdown_value)()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the dataframe with the world GeoDataFrame
    world = world.merge(data, how='left', left_on='name', right_on=COUNTRY_COL)
    
    min_dropdown_value = find_closest(world[dropdown_value].min(), True)
    max_dropdown_value = find_closest(world[dropdown_value].max(), False)

    # Construct the worldmap plot as a choropleth
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

    # Create the histogram that shows the distribution of the target variable for the selected country
    country = "United States of America"
    if hoverDataMap:
        country = hoverDataMap['points'][0]['customdata'][0]
    data = df_filtered.loc[df_filtered[COUNTRY_COL].isin([country]), :]

    fig3 = px.histogram(
        data_frame=data, 
        x=dropdown_value, 
        color=dropdown_value,
        hover_name=COUNTRY_COL, 
        hover_data=[COUNTRY_COL, dropdown_value],
        color_discrete_map={True: 'blue', False: 'red'}
    )

    fig3.update_layout(
        title=dict(
            text=f"{dropdown_value} in {country}",
            font=dict(size=10), 
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

    # when train button is not clicked, return the figues made so far without constructing the model or its associated plots
    if selected_n_clicks is None:
        feature_importances_fig = px.bar(height = 250)
        fig_cf = go.Figure()
        return fig, fig2, fig3, feature_importances_fig, fig_cf
    
    # training the logistic regression model
    df_train = df_filtered[selected_features] 
    target = dropdown_value
    X, clf, X_test, y_test = train(df_filtered.drop(['Country', 'Timestamp'], axis=1).astype(int), 
                                   df_train.astype(int), target, selected_features)

    # store the feature importances
    importances = clf.coef_[0]
    feature_importances = pd.Series(importances, index=X.columns)
    feature_importances = feature_importances.apply(lambda x: max(x, 0))
    feature_importances = feature_importances.sort_values(ascending=True)

    #Plot the feature importances
    feature_importances_fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index,
                                     labels={'x': 'Importance', 'y': 'Feature'},
                                     title='Feature Importances',
                                     height = 250)
    
    feature_importances_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white')
    )

    # Calculate the counterfactuals
    if isinstance(cf_features, str):
        cf_features = [cf_features]
    
    # prints the features for the counterfactuals to allow for easier debugging
    if not(cf_features is None):
        features_to_vary = [feature for feature in cf_features if feature in mutable_features]
        print(features_to_vary)

    # if no features were selected, return the figues constructed so far and an empty figure for the counterfactuals
    else:
        fig_cf = go.Figure()
                # Update layout for black background
        fig_cf.update_layout(\
            title="Counterfactuals",
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig, fig2, fig3, feature_importances_fig, fig_cf
    cf, differences = get_counterfactuals_from_model(X=X_test, y=y_test, model=clf, features_to_vary=features_to_vary, 
                                                    outcome_name=target, idx=list(range(100)), total_CFs=1)
    
    # Prepares variables and data for the construction of the counterfactuals plot
    sample_row = compute_all_differences(differences)
    plot_data = pd.DataFrame({
        'Feature': sample_row.columns,
        'Value': sample_row.values.flatten()
    })
    plot_data['Color'] = ['red' if val < 0 else 'blue' for val in plot_data['Value']]
    plot_data = plot_data[plot_data['Value'].abs() > 0]
    plot_data = plot_data.sort_values(by=["Value"], ascending=True)
    plot_data = plot_data.reset_index(drop=True)

    #Construct the counterfactuals plot
    fig_cf = go.Figure()
    for index, row in plot_data.iterrows():
        fig_cf.add_trace(go.Bar(
            x=[row['Value']],
            y=[row['Feature']],
            orientation='h',
            marker_color=row['Color'],
            name='1 --> 0' if row['Color'] == 'red' else '0 --> 1'
        ))

    fig_cf.update_layout(
        title='Counterfactuals',
        xaxis_title='Value',
        yaxis_title='Feature',
        barmode='stack',
        showlegend=True
    )

    # Customize legend
    fig_cf.update_layout(
        legend=dict(
            title='What to change',
            itemsizing='constant',
            traceorder='normal'
        )
    )

    return fig, fig2, fig3, feature_importances_fig, fig_cf


# Helper functions for Map visualization
def group_by_country(df: pd.DataFrame, col: str):
    """Groups a dataframe by country and selects a column
    
    Parameters:
    df: pd.DataFrame that is to be grouped
    col: string that represents the column to group df by

    Returns:
    inner: function that creates and returns a grouped copy of df
    """
    
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

    Parameters:
    num: The location of the mouse on the worldmap plot
    find_min: Boolean value, True if we want to return lb, False if we want to return ub
    num_list: The locations of every country on the worldmap plot

    Returns one of the following:
    lb: the closest number to the given percentage that is smaller than that percentage
    ub: the closest number to the given percentage that is greater than that percentage
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
def train(df_filtered: pd.DataFrame, df_train: pd.DataFrame, target: str, selected_features: list):
    """Trains a logistic regression model
    
    Parameters:
    df_filtered: pd.DataFrame, the filtered dataset
    df_train: pd.DataFrame, the filtered dataset to be used for training
    target: string of the target variable
    selected_features: list of features to train the model on
    
    returns:
    X: pd.DataFrame, subset of df_train with only the features to be used for classification
    clf: Trained logistic regression model
    X_test: np.array, contains the feature values to be used for classification of the test set
    y_test: list, contains the target variable values to be used for classification of the test set
    """

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

    return X, clf, X_test, y_test

def get_counterfactuals_from_model(X: pd.DataFrame, y: pd.Series, model, features_to_vary, outcome_name: str, 
                                   idx = None, n_samples: int = 1, random_state: int = 0, total_CFs: int = 5, 
                                   backend: str = "sklearn", continous_features = [], desired_class: str = "opposite") \
                                    -> dice_ml.counterfactual_explanations.CounterfactualExplanations:
    """
    Generates counterfactual explanations using the dice_ml package.

    Parameters:
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
    idx: Any, optional
        An integer index or list of integers from which to sample. If None, 
            a random sample is used (default is None)
    n_samples: int, optional
        Number of samples to be used from the training data (default is 1)
    random_state: int, optional
        seed for random index generator (default is 0)
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
    
    Returns:
    cf: dice_ml.counterfactual_explanations.CounterfactualExplanations
        A CounterfactualExplanations object that contains the list of
            counterfactual examples per query_instance as one of its attributes.
    differences_as_df: pd.DataFrame
        the differences between the query instances and counterfactuals, stored as a DataFrame
    """
    # Get values where treatment = 1
    condition = (y == 1).values
    X_ = X[condition].reset_index(drop=True)
    y_ = y[condition].reset_index(drop=True)

    # Prepare data for DiCE
    data_interface = dice_ml.Data(dataframe=pd.concat([X_, y_], axis=1), 
                         continuous_features=continous_features, outcome_name=outcome_name)
    
    # Prepare model for DiCE
    model_interface = dice_ml.Model(model=model, backend=backend)

    # Initialize DiCE
    dice = dice_ml.Dice(data_interface = data_interface, model_interface=model_interface)

    # Choose an instance for which you want counterfactuals
    if idx is None:
        query_instances = X.sample(n=n_samples, random_state=random_state)
    else:
        query_instances = X.iloc[idx]
    
    # Generate counterfactual explanations
    cf = dice.generate_counterfactuals(
        query_instances=query_instances,
        total_CFs=total_CFs,
        desired_class=desired_class,
        features_to_vary=features_to_vary
    )

    # Get differences between query instances and counterfactuals
    def get_differences(cf, query_instances):
        """Returns the differences between the query instances and counterfactuals.
        
        Parameters:
        cf: dice_ml.counterfactual_explanations.CounterfactualExplanations
            A CounterfactualExplanations object that contains the list of
                counterfactual examples per query_instance as one of its attributes.
        query_instances: A query whose instances are used to calculate the differences with counterfactuals

        Returns:
        differences: list
            the differences between the query instances and counterfactuals
        """
        # Store differences in array
        differences = []
        for cf_, query_instance in zip(cf.__dict__['_cf_examples_list'], query_instances.to_numpy()):
            try:
                # Multiply by -1 because we want to know the change from the original entry
                difference = (cf_.final_cfs_df.to_numpy()[:, :-1] - query_instance) * -1
                difference = np.where(difference == -0., 0, difference)
                differences.append(difference)
            except:
                pass
        return differences
    
    # Get differences as pandas DataFrames
    differences = get_differences(cf, query_instances)
    differences_as_df = []

    for difference in differences:
        difference_as_df = []
        for row_difference in difference:
            diff_as_df = pd.DataFrame({X.columns[i]:  [row_difference[i]] for i in range(len(X.columns))})
            difference_as_df.append(diff_as_df)
        differences_as_df.append(difference_as_df)
    
    return cf, differences_as_df

def compute_all_differences(differences):
    """Finds the average of the first elements of the tuples stored in the input
    
    Parameter:
    differences: a list of tuples
    
    Returns the average of the first elements of the tuples in differences
    """
    differences_ = differences[0][0]
    for difference in differences[1:]:
        differences_ += difference[0]
    return differences_ / len(differences)

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)