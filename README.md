# 2AMV10, Group-17
This project involves the development of a visual analytics tool based on the Dash library to allow for interactive visualization. The project includes the following tasks:
- Loading our selected dataset regarding mental health into the `data_preperation.ipynb` notebook
- Pre-processing the dataset
- Saving the pre-processed dataset as `data_cleaned.csv`
- Loading the cleaned data into `app.py`
- Filtering the data based on user-selected values in the dropdown menus
- creating a logistic regression model and using the filtered dataset and the model in combination with multiple dropdown menus to visualize the data and model that the user is interested in

For more information regarding the used dataset, [click here](https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset)

## File Descriptions

- **`app.py`**: Our main code. This script creates our Dashboard and the visualizations.
- **`data_preperation.ipynb`**: Loads and pre-processes `Mental Health Dataset.csv`, creates and saves `data_cleaned.csv`. <br>
  **NOTE**: Since `data_cleaned.csv` is already present in this folder, there is no need to run this notebook. We submitted this notebook for completeness sake
- **`data_exploration.ipynb`**: 
- **`Mental Health Dataset.csv`**: The initial dataset before any pre-processing took place.
- **`data_cleaned.csv`**: The dataset after pre-processing has taken place.

## Setup

**Install the dependencies by executing the following command in the terminal of your IDE of choice**:
```sh
pip install numpy, pandas, matplotlib, geopandas, plotly, dash, dash_bootstrap_components, scikit-learn, dice-ml
```

## Running the code
In order to run the code, open `app.py` in Visual Studio Code or another IDE of your choice. <br>
After installing the dependencies, execute the following command to run the code:
```sh
python .\app.py
```
This will yield a URL that leads to the dashboard where the user can interact with the dataset and the visualizations

## Interacting with the dashboard
There are 3 ways to interact with the dashboard:<br>
1. Use one of the four dropdown menus to customize:<br>
   - The features that are used to train the logistic regression model.
   - The target variable of the logistic regression model.
   - The filter(s) that you wish to apply to the dataset before training the logistic regression model.
   - The features that are used to construct the visualization of counterfactuals. <br>
   **NOTE**: only mutable features that have been selected to train the model can be selected to construct the counterfactuals
2. By clicking the `Train Selected Features` button on the buttom left of the dashboard, the logistic regression model is trained and the feature importances and counterfactuals are calculated and displayed. <br>
**NOTE**: Once this button has been clicked, the application will slow down considerably since the model must be re-trained and the counterfactuals must be re-calculated every time the user changes the dataset after having clicked this button.
3. By hovering your mouse over the provided visualizations, more information is provided. Additionally, hovering your mouse over a country in the choropleth map changes the barchart next to it to display the distribution of the target variable for the selected country.


## Example
The following input was provided to obtain the visualizations shown in the video. 
- **Independant variables**: `self_employed`, `Female`, `family_history`, `Coping_Struggles`, `Occupation_Business`
- **Target variable**: `treatment`
- **Filters**: None
- **Features used to construct the counterfactuals**: `self_employed`, `Coping_Struggles`, `Occupation_Business`


