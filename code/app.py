import dash
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
from model import predict_selling_price  # MLflow helper

# -------------------------------
# Load dataset & encoders
# -------------------------------
df = pd.read_csv("Cars_Price.csv")

brand_enc_path = "Model/car_brand_encoder.model"
fuel_enc_path = "Model/car_fuel_encoder.model"

label_car = pickle.load(open(brand_enc_path, 'rb'))  # OneHotEncoder
fuel_car = pickle.load(open(fuel_enc_path, 'rb'))   # LabelEncoder

brand_cats = label_car.categories_[0].tolist()
fuel_cat = fuel_car.classes_.tolist()

default_values = {
    "brand": brand_cats[0],
    "year": 2017,
    "max_power": 82.4,
    "mileage": 19.42,
    "fuel": fuel_cat[0]
}

# -------------------------------
# Helpers
# -------------------------------
def create_input_card(label, component):
    return dbc.Card(
        dbc.CardBody([
            html.H5(label, style={"text-align": "center", "color": "#ffffff", "marginBottom": "10px"}),
            component
        ]),
        style={
            "marginBottom": "20px",
            "backgroundColor": "#1F1F2E",
            "borderRadius": "12px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.3)"
        }
    )

# -------------------------------
# Input cards
# -------------------------------
card_brand = create_input_card(
    "Car Brand",
    dcc.Dropdown(
        id="brand3",
        options=[{"label": b, "value": b} for b in brand_cats],
        value=default_values["brand"],
        style={"color": "black"}
    )
)

card_year = create_input_card(
    "Year of Manufacture",
    dcc.Dropdown(
        id="year3",
        options=[{"label": i, "value": i} for i in sorted(df["year"].unique())],
        value=default_values["year"],
        style={"color": "black"}
    )
)

card_fuel = create_input_card(
    "Fuel Type",
    dcc.Dropdown(
        id="fuel3",
        options=[{"label": f, "value": f} for f in fuel_cat],
        value=default_values["fuel"],
        style={"color": "black"}
    )
)

card_mileage = create_input_card(
    "Mileage (km/l)",
    dcc.Input(id="mileage3", type="number", value=default_values["mileage"], style={"width": "100%"})
)

card_power = create_input_card(
    "Max Power (bhp)",
    dcc.Input(id="max_power3", type="number", value=default_values["max_power"], style={"width": "100%"})
)

card_predicted_price = dbc.Card(
    dbc.CardBody([
        html.H4("Predicted Price Class", style={"color": "#00FFAA", "textAlign": "center"}),
        html.H2(" ", id="predicted_class", style={"fontWeight": "bold", "textAlign": "center", "color": "#FFFFFF"}),
        html.Hr(style={"borderColor": "#333"}),
        html.H6("Raw Model Output:", style={"marginTop": "10px", "color": "#AAAAAA", "textAlign": "center"}),
        html.H3(" ", id="raw_prediction", style={"fontWeight": "bold", "textAlign": "center", "color": "#FFFFFF"})
    ]),
    style={
        "marginBottom": "20px",
        "backgroundColor": "#1F1F2E",
        "borderRadius": "12px",
        "boxShadow": "0 4px 15px rgba(0,0,0,0.3)"
    }
)

# -------------------------------
# Sidebar
# -------------------------------
sidebar = html.Div(
    [
        html.H2("Car Price AI", className="display-6", style={"color": "#00FFAA", "marginBottom": "40px", "fontWeight": "bold"}),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="#", active=True, style={"color": "#FFFFFF"}),
                dbc.NavLink("About", href="#", style={"color": "#FFFFFF"}),
                dbc.NavLink("Contact", href="#", style={"color": "#FFFFFF"})
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "240px",
        "padding": "30px 20px",
        "backgroundColor": "#0E0E1F",
        "boxShadow": "2px 0 8px rgba(0,0,0,0.4)"
    },
)

# -------------------------------
# Main Content
# -------------------------------
content = html.Div(
    [
        # Header
        html.Div([
            html.H1("Car Price Prediction V3", style={"color": "#00FFAA", "fontSize": "2.5rem", "fontWeight": "bold"}),
            html.H5("Predict instantly based on your car’s features", style={"color": "#BBBBBB", "marginTop": "5px"}),
            html.Hr(style={"borderColor": "#333"})
        ], style={"padding": "40px 20px"}),

        # Input Section
        dbc.Row([dbc.Col(card_brand, md=4), dbc.Col(card_year, md=4), dbc.Col(card_fuel, md=4)]),
        dbc.Row([dbc.Col(card_power, md=6), dbc.Col(card_mileage, md=6)]),
        html.Br(),

        # Buttons
        dbc.Row([
            dbc.Col(dbc.Button("Calculate Price", id="submit3", color="success", className="me-2", style={"width": "100%"}), md=3),
            dbc.Col(dbc.Button("Clear", id="clear3", color="secondary", className="ms-2", style={"width": "100%"}), md=3)
        ]),
        html.Br(), html.Br(),

        # Prediction Section
        dbc.Row([dbc.Col(card_predicted_price, md=12)]),
    ],
    style={"marginLeft": "260px", "padding": "20px", "backgroundColor": "#121212", "minHeight": "100vh"}
)

# -------------------------------
# App Setup
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Google Font
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Car Price AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #121212; color: #FFFFFF; font-family: 'Poppins', sans-serif; }
            .dash-bootstrap { font-family: 'Poppins', sans-serif; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([sidebar, content])

# -------------------------------
# Callback
# -------------------------------
@callback(
    Output("predicted_class", "children"),
    Output("raw_prediction", "children"),
    Output("brand3", "value"),
    Output("year3", "value"),
    Output("max_power3", "value"),
    Output("mileage3", "value"),
    Output("fuel3", "value"),
    Input("submit3", "n_clicks"),
    Input("clear3", "n_clicks"),
    State("brand3", "value"),
    State("year3", "value"),
    State("max_power3", "value"),
    State("mileage3", "value"),
    State("fuel3", "value"),
    prevent_initial_call=True
)
def handle_buttons(submit_clicks, clear_clicks, brand, year, max_power, mileage, fuel):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "clear3":
        return "", "", default_values["brand"], default_values["year"], default_values["max_power"], default_values["mileage"], default_values["fuel"]
    elif triggered_id == "submit3":
        raw_pred, predicted_class = predict_selling_price(
            year=year, max_power=max_power, mileage=mileage, fuel=fuel, brand=brand
        )
        return predicted_class, f"{raw_pred:.2f}", brand, year, max_power, mileage, fuel

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=80, debug=True)
