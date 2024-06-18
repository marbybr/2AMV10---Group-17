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
<<<<<<< Updated upstream
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
=======
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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
=======
            #name=f"{row['variable']} = 0",
            marker_color='red',
            showlegend=False
        ))
        fig.add_trace(go.Bar(
            x=[row['variable']],
            y=[row['count_1']],
            #name=f"{row['variable']} = 1",
            marker_color='blue',
            showlegend=False
        ))
>>>>>>> Stashed changes
    
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
<<<<<<< Updated upstream
        title="Counts of Binary Variables",
=======
        title=dict(
            text="Distributions of selected features",
            font=dict(size=15), 
            automargin=True, 
            yref='container',
            y=0.95
            ),
>>>>>>> Stashed changes
        xaxis_title="Variable",
        yaxis_title="Ratio",
        legend_title="Value",
        xaxis_tickangle = -45,
        height = 600,
        bargap = 0.5, #Space between groups
<<<<<<< Updated upstream
        bargroupgap = 0.5 #Space within groups
=======
        bargroupgap = 0.5, #Space within groups
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white',
                  size=8)
>>>>>>> Stashed changes
    )
    fig.update_yaxes(range=[0, 1])

    table = df_filtered[table_features].to_dict('records')

<<<<<<< Updated upstream
    return printed_text, fig, table
=======
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
            tickfont=dict(size=20),
            tickformat=".0%"
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
    if isinstance(cf_features, str):
        cf_features = [cf_features]
    elif cf_features is None:
        fig_cf = go.Figure()
    
    try:
        features_to_vary = [feature for feature in cf_features if feature in mutable_features]
        print(features_to_vary)
        cf, differences = get_counterfactuals_from_model(X=X_test, y=y_test, model=clf, features_to_vary=features_to_vary, 
                                                        outcome_name=target, idx=list(range(3)), total_CFs=15)
        print(differences)
        
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
>>>>>>> Stashed changes


if __name__ == '__main__':
    app.run(debug=True)