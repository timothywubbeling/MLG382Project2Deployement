import os
import warnings
import dash
import joblib
import pandas as pd
from dash import html, dcc, Input, Output, State

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

temp_model       = joblib.load(os.path.join(BASE_DIR, "models", "temperature_forecast_model.pkl"))
rain_model       = joblib.load(os.path.join(BASE_DIR, "models", "rain_classifier.pkl"))
snow_model       = joblib.load(os.path.join(BASE_DIR, "models", "snow_classifier.pkl"))
visibility_model = joblib.load(os.path.join(BASE_DIR, "models", "high_visibility_classifier.pkl"))

# StandardScaler parameters from training data (df.describe() pre-scaling)
SCALE_MEAN = {
    "Temp_C":           8.798144,
    "Dew Point Temp_C": 2.555294,
    "Rel Hum_%":        67.431694,
    "Wind Speed_km/h":  14.945469,
    "Visibility_km":    27.664447,
    "Press_kPa":        101.051623,
}
SCALE_STD = {
    "Temp_C":           11.687883,
    "Dew Point Temp_C": 10.883072,
    "Rel Hum_%":        16.918881,
    "Wind Speed_km/h":  8.688696,
    "Visibility_km":    12.622688,
    "Press_kPa":        0.844005,
}

WEATHER_COLS = [
    "Blowing Snow", "Clear", "Cloudy", "Drizzle", "Fog",
    "Freezing Drizzle", "Freezing Fog", "Freezing Rain", "Haze",
    "Heavy Rain Showers", "Ice Pellets", "Mainly Clear", "Moderate Rain",
    "Moderate Rain Showers", "Moderate Snow", "Mostly Cloudy", "Rain",
    "Rain Showers", "Snow", "Snow Grains", "Snow Pellets", "Snow Showers",
    "Thunderstorms",
]

def sc(val, col):
    return (val - SCALE_MEAN[col]) / SCALE_STD[col]


app = dash.Dash(__name__)
server = app.server
app.title = "WXCAST-84 MK.II"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --green:  #39ff14;
            --green2: #00c853;
            --amber:  #ffb300;
            --red:    #ff3c3c;
            --blue:   #00e5ff;
            --bg:     #050d05;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Share Tech Mono', monospace;
            background: var(--bg);
            color: var(--green);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* CRT scanlines */
        body::after {
            content: '';
            position: fixed;
            inset: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0,0,0,0.15) 2px,
                rgba(0,0,0,0.15) 4px
            );
            pointer-events: none;
            z-index: 9999;
        }

        /* Phosphor vignette */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.72) 100%);
            pointer-events: none;
            z-index: 9998;
        }

        .page-wrap {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 36px 20px 56px;
        }

        @keyframes blink      { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes flicker    { 0%{opacity:.97} 5%{opacity:.94} 10%{opacity:.98} 30%{opacity:.93} 100%{opacity:.98} }
        @keyframes glow-pulse { 0%,100%{text-shadow:0 0 6px var(--green),0 0 12px var(--green)} 50%{text-shadow:0 0 10px var(--green),0 0 22px var(--green),0 0 40px var(--green2)} }

        /* ── Header ── */
        .header { text-align:center; margin-bottom:28px; animation:flicker 8s infinite; }

        .header-badge {
            display: inline-block;
            font-family: 'VT323', monospace;
            font-size: 13px;
            letter-spacing: 4px;
            color: var(--amber);
            border: 1px solid var(--amber);
            padding: 3px 14px;
            margin-bottom: 10px;
            text-shadow: 0 0 8px var(--amber);
            box-shadow: 0 0 8px rgba(255,179,0,.3), inset 0 0 8px rgba(255,179,0,.05);
        }

        .header h1 {
            font-family: 'VT323', monospace;
            font-size: 60px;
            color: var(--green);
            letter-spacing: 6px;
            line-height: 1;
            animation: glow-pulse 3s ease-in-out infinite;
        }

        .header h1 .amber { color:var(--amber); text-shadow:0 0 10px #ffb300; }
        .header h1 .cursor { animation:blink 1s step-end infinite; }

        .header-sub {
            font-size: 12px;
            letter-spacing: 3px;
            color: rgba(57,255,20,.4);
            margin-top: 6px;
            text-transform: uppercase;
        }

        /* ── Terminal window ── */
        .terminal {
            width: 100%;
            max-width: 760px;
            background: rgba(0,10,0,.94);
            border: 2px solid var(--green);
            box-shadow: 0 0 0 1px rgba(57,255,20,.12), 0 0 24px rgba(57,255,20,.22), inset 0 0 40px rgba(0,0,0,.6);
        }

        .term-bar {
            background: var(--green);
            padding: 6px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .term-bar-title { font-family:'VT323',monospace; font-size:18px; color:#000; letter-spacing:2px; }
        .term-bar-dots  { display:flex; gap:6px; }
        .term-dot       { width:10px; height:10px; border:1px solid #000; background:#000; }

        .term-body { padding: 26px 32px 30px; }

        .prompt-line {
            font-size: 13px;
            color: rgba(57,255,20,.5);
            margin-bottom: 18px;
            letter-spacing: 1px;
        }
        .prompt-line span { color:var(--amber); text-shadow:0 0 6px var(--amber); }

        .retro-divider { border:none; border-top:1px dashed rgba(57,255,20,.22); margin:20px 0; }

        /* ── Input grid ── */
        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }

        .input-grid-2col {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 16px;
            margin-bottom: 20px;
        }

        .input-group { display:flex; flex-direction:column; gap:6px; }

        .input-group label {
            font-size: 11px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--amber);
            text-shadow: 0 0 6px rgba(255,179,0,.5);
        }

        .input-prefix {
            font-size: 13px;
            color: rgba(57,255,20,.6);
            text-shadow: 0 0 4px var(--green);
        }

        .input-icon-wrap input {
            width: 100% !important;
            padding: 10px 10px 10px 14px !important;
            background: #000 !important;
            border: 1px solid var(--green) !important;
            outline: none !important;
            font-size: 18px !important;
            font-family: 'VT323', monospace !important;
            color: var(--green) !important;
            letter-spacing: 1px !important;
            box-shadow: 0 0 8px rgba(57,255,20,.18), inset 0 0 10px rgba(0,0,0,.8) !important;
            transition: box-shadow .2s, border-color .2s !important;
            border-radius: 0 !important;
        }

        .input-icon-wrap input::placeholder {
            color: rgba(57,255,20,.28) !important;
            font-family: 'VT323', monospace !important;
            font-size: 17px !important;
        }

        .input-icon-wrap input:focus {
            border-color: var(--amber) !important;
            color: var(--amber) !important;
            box-shadow: 0 0 14px rgba(255,179,0,.35), inset 0 0 10px rgba(0,0,0,.8) !important;
        }

        .input-icon-wrap input::-webkit-inner-spin-button,
        .input-icon-wrap input::-webkit-outer-spin-button { opacity:.35; }

        /* Chrome autofill override — prevents white background injection */
        .input-icon-wrap input:-webkit-autofill,
        .input-icon-wrap input:-webkit-autofill:hover,
        .input-icon-wrap input:-webkit-autofill:focus {
            -webkit-box-shadow: 0 0 0 1000px #000 inset !important;
            -webkit-text-fill-color: #39ff14 !important;
            caret-color: #39ff14 !important;
        }

        /* ══════════════════════════════════════════════════════
           RETRO DROPDOWN — Dash 4.x fix
           Two problems in Dash 4:
             1. Control border is purple — CSS scope not matching
             2. Menu background is white — menu is portalled to <body>
                so parent-scoped selectors can never reach it
           Fix: ID selectors for the control, GLOBAL selectors for menu
           ══════════════════════════════════════════════════════ */

        /* ── Closed control box: target by component ID ── */
        #month, #condition {
            background-color: #000 !important;
            border-radius: 0 !important;
            font-family: 'VT323', monospace !important;
        }

        /* Kill every child element inside the control area */
        #month *, #condition * {
            background-color: #000 !important;
            color: #39ff14 !important;
            font-family: 'VT323', monospace !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* The actual visible border box — all states */
        #month [class*="control"],   #condition [class*="control"],
        #month [class*="-control"],  #condition [class*="-control"],
        #month .Select-control,      #condition .Select-control {
            background-color: #000 !important;
            border: 1px solid #39ff14 !important;
            box-shadow: 0 0 8px rgba(57,255,20,.22) !important;
            min-height: 42px !important;
            cursor: pointer !important;
        }

        /* Focused / open state — kill Dash's purple ring */
        #month [class*="control"]:focus-within,
        #condition [class*="control"]:focus-within,
        #month .Select.is-focused .Select-control,
        #condition .Select.is-focused .Select-control {
            border: 1px solid #39ff14 !important;
            box-shadow: 0 0 12px rgba(57,255,20,.4) !important;
        }

        /* Placeholder */
        #month [class*="placeholder"], #condition [class*="placeholder"],
        #month .Select-placeholder,    #condition .Select-placeholder {
            color: rgba(57,255,20,.35) !important;
            font-size: 17px !important;
        }

        /* Arrow SVG */
        #month [class*="indicator"] svg,
        #condition [class*="indicator"] svg { fill: #39ff14 !important; }
        #month [class*="indicator-separator"],
        #condition [class*="indicator-separator"] {
            background-color: rgba(57,255,20,.2) !important;
        }

        /* ── Dropdown menu — GLOBAL (portalled to <body> in Dash 4) ── */
        /* These rules are NOT scoped so they reach the portalled menu   */

        [class*="-menu"], [class*="__menu"],
        .Select-menu-outer, .Select-menu {
            background-color: #000 !important;
            border: 1px solid #39ff14 !important;
            border-radius: 0 !important;
            box-shadow: 0 6px 20px rgba(57,255,20,.25) !important;
            z-index: 9999 !important;
        }

        [class*="-menu"] *, [class*="__menu"] *,
        .Select-menu-outer *, .Select-menu * {
            background-color: #000 !important;
            color: #39ff14 !important;
            font-family: 'VT323', monospace !important;
            border-radius: 0 !important;
        }

        /* Option rows */
        [class*="-option"], [class*="__option"],
        .Select-option {
            background-color: #000 !important;
            color: #39ff14 !important;
            font-size: 17px !important;
            padding: 8px 14px !important;
            cursor: pointer !important;
        }

        /* Hovered option */
        [class*="-option"]:hover,
        [class*="option--is-focused"],
        .Select-option.is-focused {
            background-color: rgba(57,255,20,.14) !important;
            color: #39ff14 !important;
        }

        /* Selected option */
        [class*="option--is-selected"],
        .Select-option.is-selected {
            background-color: #39ff14 !important;
            color: #000 !important;
        }

        /* ── Button ── */
        .predict-btn {
            width: 100%;
            padding: 13px;
            border: 2px solid var(--green);
            background: transparent;
            font-family: 'VT323', monospace;
            font-size: 22px;
            letter-spacing: 4px;
            color: var(--green);
            cursor: pointer;
            text-transform: uppercase;
            text-shadow: 0 0 8px var(--green);
            box-shadow: 0 0 12px rgba(57,255,20,.22), inset 0 0 12px rgba(57,255,20,.04);
            transition: background .15s, color .15s, box-shadow .15s;
            border-radius: 0;
        }

        .predict-btn:hover {
            background: var(--green);
            color: #000;
            text-shadow: none;
            box-shadow: 0 0 24px rgba(57,255,20,.6);
        }

        .predict-btn:active { background:var(--green2); border-color:var(--green2); }

        /* ── Results ── */
        .results-header {
            font-size: 12px;
            letter-spacing: 3px;
            color: rgba(57,255,20,.4);
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .result-tile {
            background: #000;
            border: 1px solid rgba(57,255,20,.3);
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            box-shadow: inset 0 0 12px rgba(0,0,0,.7);
            transition: border-color .2s, box-shadow .2s;
        }

        .result-tile:hover {
            border-color: var(--green);
            box-shadow: 0 0 10px rgba(57,255,20,.18), inset 0 0 12px rgba(0,0,0,.7);
        }

        .tile-label { font-size:11px; letter-spacing:3px; text-transform:uppercase; color:rgba(57,255,20,.4); }

        .tile-value {
            font-family: 'VT323', monospace;
            font-size: 36px;
            line-height: 1.1;
            color: var(--green);
            text-shadow: 0 0 10px var(--green);
        }

        .tile-sub { font-size:11px; color:rgba(57,255,20,.28); letter-spacing:1px; }

        .tile-yes  { border-color:rgba(0,229,255,.45); }
        .tile-yes  .tile-value { color:var(--blue);  text-shadow:0 0 10px var(--blue);  }
        .tile-no   { border-color:rgba(57,255,20,.18); }
        .tile-high { border-color:rgba(57,255,20,.55); }
        .tile-low  { border-color:rgba(255,60,60,.4);  }
        .tile-low  .tile-value { color:var(--red);   text-shadow:0 0 10px var(--red);   }

        /* ── Error ── */
        .error-msg {
            margin-top: 16px;
            padding: 12px 16px;
            border: 1px solid var(--red);
            color: var(--red);
            font-size: 14px;
            letter-spacing: 1px;
            text-shadow: 0 0 6px var(--red);
            box-shadow: 0 0 10px rgba(255,60,60,.12), inset 0 0 10px rgba(0,0,0,.8);
            background: rgba(255,60,60,.04);
        }

        .error-msg::before { content:'!! ERROR: '; color:var(--red); }

        /* ── Footer ── */
        .footer {
            margin-top: 20px;
            font-size: 11px;
            letter-spacing: 2px;
            color: rgba(57,255,20,.18);
            text-align: center;
            text-transform: uppercase;
        }

        @media (max-width: 580px) {
            .term-body       { padding: 18px 14px 22px; }
            .input-grid      { grid-template-columns: 1fr; }
            .input-grid-2col { grid-template-columns: 1fr; }
            .results-grid    { grid-template-columns: 1fr; }
            .header h1       { font-size: 42px; }
        }
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
"""

MONTH_OPTIONS = [
    {"label": "01 — JANUARY",   "value": 1},
    {"label": "02 — FEBRUARY",  "value": 2},
    {"label": "03 — MARCH",     "value": 3},
    {"label": "04 — APRIL",     "value": 4},
    {"label": "05 — MAY",       "value": 5},
    {"label": "06 — JUNE",      "value": 6},
    {"label": "07 — JULY",      "value": 7},
    {"label": "08 — AUGUST",    "value": 8},
    {"label": "09 — SEPTEMBER", "value": 9},
    {"label": "10 — OCTOBER",   "value": 10},
    {"label": "11 — NOVEMBER",  "value": 11},
    {"label": "12 — DECEMBER",  "value": 12},
]

WEATHER_OPTIONS = [{"label": w, "value": w} for w in WEATHER_COLS]

app.layout = html.Div(className="page-wrap", children=[

    html.Div(className="header", children=[
        html.Div("[ SYSTEM ONLINE — MK.II ]", className="header-badge"),
        html.H1(["WXCAST", html.Span("-84", className="amber"), html.Span("_", className="cursor")]),
        html.Div("CANADA TORONTO ATMOSPHERIC FORECAST ENGINE", className="header-sub"),
    ]),

    html.Div(className="terminal", children=[

        html.Div(className="term-bar", children=[
            html.Span("ATMOSPHERIC ANALYSIS SYSTEM — ADVANCED MODE", className="term-bar-title"),
            html.Div(className="term-bar-dots", children=[
                html.Div(className="term-dot"),
                html.Div(className="term-dot"),
                html.Div(className="term-dot"),
            ]),
        ]),

        html.Div(className="term-body", children=[

            html.Div(className="prompt-line", children=[
                html.Span("root@wxcast"),
                ":~$ INPUT ATMOSPHERIC CONDITIONS // ALL FIELDS REQUIRED",
            ]),

            # Row 1: Dew Point | Humidity | Wind Speed
            html.Div(className="input-grid", children=[
                html.Div(className="input-group", children=[
                    html.Label("Dew Point"),
                    html.Div("> DEW POINT", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="dew", type="number", placeholder="-28.5 to 24.4"),
                    ]),
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Humidity"),
                    html.Div("> HUM %", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="humidity", type="number", placeholder="18 — 100", min=0, max=100),
                    ]),
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Wind Speed"),
                    html.Div("> WIND SPEED", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="wind", type="number", placeholder="0 — 83 km/h", min=0),
                    ]),
                ]),
            ]),

            # Row 2: Visibility | Pressure | Hour
            html.Div(className="input-grid", children=[
                html.Div(className="input-group", children=[
                    html.Label("Visibility"),
                    html.Div("> VISIBILITY (KM)", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="visibility", type="number", placeholder="0.2 — 48.3 km", min=0),
                    ]),
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Pressure"),
                    html.Div("> PRESSURE (KPA)", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="pressure", type="number", placeholder="97.5 — 103.7", min=80),
                    ]),
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Hour of Day"),
                    html.Div("> Hour of Day", className="input-prefix"),
                    html.Div(className="input-icon-wrap", children=[
                        dcc.Input(id="hour", type="number", placeholder="0 — 23", min=0, max=23),
                    ]),
                ]),
            ]),

            # Row 3: Month | Weather Condition
            html.Div(className="input-grid-2col", children=[
                html.Div(className="input-group", children=[
                    html.Label("Month"),
                    html.Div("> MONTH", className="input-prefix"),
                    dcc.Dropdown(
                        id="month",
                        options=MONTH_OPTIONS,
                        placeholder="> SELECT MONTH...",
                        clearable=False,
                        className="retro-select",
                        style={
                            "backgroundColor": "#000",
                            "color": "#39ff14",
                            "border": "1px solid #39ff14",
                            "borderRadius": "0",
                            "fontFamily": "VT323, monospace",
                        },
                    ),
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Current Weather Condition"),
                    html.Div("> WEATHER_COND", className="input-prefix"),
                    dcc.Dropdown(
                        id="condition",
                        options=WEATHER_OPTIONS,
                        placeholder="> SELECT CONDITION...",
                        clearable=False,
                        className="retro-select",
                        style={
                            "backgroundColor": "#000",
                            "color": "#39ff14",
                            "border": "1px solid #39ff14",
                            "borderRadius": "0",
                            "fontFamily": "VT323, monospace",
                        },
                    ),
                ]),
            ]),

            html.Button("> RUN FULL FORECAST ANALYSIS [ENTER]", id="btn", n_clicks=0, className="predict-btn"),

            html.Div(id="output"),
        ]),
    ]),

    html.Div("// GRADIENT BOOSTING + HIST GRADIENT BOOSTING // TORONTO ECCC DATASET // BUILD 2012 //", className="footer"),
])


@app.callback(
    Output("output", "children"),
    Input("btn", "n_clicks"),
    State("dew",        "value"),
    State("humidity",   "value"),
    State("wind",       "value"),
    State("visibility", "value"),
    State("pressure",   "value"),
    State("hour",       "value"),
    State("month",      "value"),
    State("condition",  "value"),
    prevent_initial_call=True,
)
def predict(n_clicks, dew, humidity, wind, visibility, pressure, hour, month, condition):

    if any(v is None for v in [dew, humidity, wind, visibility, pressure, hour, month, condition]):
        return html.Div("All fields required before analysis can proceed.", className="error-msg")
    if not (0 <= humidity <= 100):
        return html.Div("Humidity out of range. Expected: 0 – 100 %.", className="error-msg")
    if wind < 0:
        return html.Div("Wind speed cannot be negative.", className="error-msg")
    if visibility < 0:
        return html.Div("Visibility cannot be negative.", className="error-msg")
    if not (0 <= hour <= 23):
        return html.Div("Hour must be 0 – 23.", className="error-msg")

    # Clip wind speed (same as training preprocessing)
    wind_clipped = min(float(wind), 36.5)

    # high_visibility flag (same as training: 1 if vis == 48.3)
    high_vis = 1 if float(visibility) >= 48.3 else 0

    # One-hot encode the selected weather condition
    weather_one_hot = {col: (1 if col == condition else 0) for col in WEATHER_COLS}

    # Base feature set (no Temp_C — that's predicted first)
    base = {
        "Dew Point Temp_C": sc(float(dew),        "Dew Point Temp_C"),
        "Rel Hum_%":        sc(float(humidity),    "Rel Hum_%"),
        "Wind Speed_km/h":  sc(wind_clipped,       "Wind Speed_km/h"),
        "Visibility_km":    sc(float(visibility),  "Visibility_km"),
        "Press_kPa":        sc(float(pressure),    "Press_kPa"),
        "hour":             int(hour),
        "day":              15,
        "month":            int(month),
        "day_of_week":      2,
        **weather_one_hot,
        "high_visibility":  high_vis,
    }

    try:
        # Step 1 — predict temperature (model trained on all cols except Temp_C)
        X_temp = pd.DataFrame([base])
        temp_scaled = temp_model.predict(X_temp)[0]
        temp_raw = temp_scaled * SCALE_STD["Temp_C"] + SCALE_MEAN["Temp_C"]

        # Full feature set including predicted Temp_C (for the other models)
        full = {"Temp_C": temp_scaled, **base}

        # Step 2 — predict rain (model trained on all cols except Rain)
        X_rain = pd.DataFrame([{k: v for k, v in full.items() if k != "Rain"}])
        rain = rain_model.predict(X_rain)[0]

        # Step 3 — predict snow (model trained on all cols except Snow)
        X_snow = pd.DataFrame([{k: v for k, v in full.items() if k != "Snow"}])
        snow = snow_model.predict(X_snow)[0]

        # Step 4 — predict visibility (model trained on all cols except high_visibility and Visibility_km)
        X_vis = pd.DataFrame([{k: v for k, v in full.items() if k not in ["high_visibility", "Visibility_km"]}])
        vis_pred = visibility_model.predict(X_vis)[0]

        rain_label = "DETECTED" if rain == 1 else "NONE"
        snow_label = "DETECTED" if snow == 1 else "NONE"
        vis_label  = "HIGH"     if vis_pred == 1 else "LOW"

        temp_color = "var(--blue)" if temp_raw <= 0 else ("var(--amber)" if temp_raw >= 25 else "var(--green)")

        return html.Div([
            html.Hr(className="retro-divider"),
            html.Div("// FORECAST OUTPUT — ANALYSIS COMPLETE //", className="results-header"),
            html.Div(className="results-grid", children=[
                html.Div(className="result-tile", children=[
                    html.Div("TEMPERATURE", className="tile-label"),
                    html.Div(f"{temp_raw:.1f} C", className="tile-value",
                             style={"color": temp_color, "textShadow": f"0 0 10px {temp_color}"}),
                    html.Div("PREDICTED AIR TEMP", className="tile-sub"),
                ]),
                html.Div(className=f"result-tile tile-{'yes' if rain == 1 else 'no'}", children=[
                    html.Div("RAINFALL", className="tile-label"),
                    html.Div(rain_label, className="tile-value"),
                    html.Div("PRECIPITATION STATUS", className="tile-sub"),
                ]),
                html.Div(className=f"result-tile tile-{'yes' if snow == 1 else 'no'}", children=[
                    html.Div("SNOWFALL", className="tile-label"),
                    html.Div(snow_label, className="tile-value"),
                    html.Div("SNOWFALL STATUS", className="tile-sub"),
                ]),
                html.Div(className=f"result-tile tile-{vis_label.lower()}", children=[
                    html.Div("VISIBILITY", className="tile-label"),
                    html.Div(vis_label, className="tile-value"),
                    html.Div("ATMOSPHERIC CLARITY", className="tile-sub"),
                ]),
            ]),
        ])

    except Exception as e:
        return html.Div(f"Forecast engine failure: {str(e)}", className="error-msg")



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )
