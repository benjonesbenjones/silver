# by ben with <3
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_colorscales
import pandas as pd
import cufflinks as cf
import numpy as np

app = dash.Dash(__name__)
server = app.server

df_lat_lon = pd.read_csv('lib/lat_lon_counties.csv')
df_lat_lon['FIPS '] = df_lat_lon['FIPS '].apply(lambda x: str(x).zfill(5))

df_full_data = pd.read_csv('lib/votes.csv')
per_point_diff_2016 = df_full_data['per_point_diff_2016']
per_point_diff_2012 = df_full_data['per_point_diff_2012']

print(per_point_diff_2016)
print(per_point_diff_2016)
print(df_lat_lon['FIPS '])

YEARS = [2012, 2016]

DEFAULT_COLORSCALE = ["#EF8A62", "#F7F7F7", "#67A9CF"]

DEFAULT_OPACITY = 0.8


mapbox_access_token = "pk.eyJ1IjoiYmVuam9uZXM0NzQ3IiwiYSI6ImNqczBoZWo1MzFlZ3Q0YW81YTFtcDVrN3AifQ.anJanZY5dXkl6JIC4P8RRQ"


'''
~~~~~~~~~~~~~~~~
~~ APP LAYOUT ~~
~~~~~~~~~~~~~~~~
'''

app.layout = html.Div(children=[

	html.Div([
		html.Div([
			html.Div([
				html.H4(children='SILVER'),
				html.P('RESULTS Select year:'),
			]),

			html.Div([
				dcc.Slider(
					id='years-slider',
					min=min(YEARS),
					max=max(YEARS),
					value=min(YEARS),
					marks={str(year): str(year) for year in YEARS},
				),
			], style={'width':400, 'margin':25}),

			html.Br(),

			html.P('Map transparency:',
				style={
					'display':'inline-block',
					'verticalAlign': 'top',
					'marginRight': '10px'
				}
			),

			html.Div([
				dcc.Slider(
					id='opacity-slider',
					min=0, max=1, value=DEFAULT_OPACITY, step=0.1,
					marks={tick: str(tick)[0:3] for tick in np.linspace(0,1,11)},
				),
			], style={'width':300, 'display':'inline-block', 'marginBottom':10}),

			html.Div([
				dash_colorscales.DashColorscales(
					id='colorscale-picker',
					colorscale=DEFAULT_COLORSCALE,
					nSwatches=16,
					fixSwatches=True
				)
			], style={'display':'inline-block'}),

			html.Div([
				dcc.Checklist(
				    options=[{'label': 'Hide legend', 'value': 'hide_legend'}],
					value=[],
					labelStyle={'display': 'inline-block'},
					id='hide-map-legend',
				)
			], style={'display':'inline-block'}),

		], style={'margin':20} ),

		dcc.Graph(
			id = 'county-choropleth',
			figure = dict(
				data=dict(
					lat = df_lat_lon['Latitude '],
					lon = df_lat_lon['Longitude'],
					text = df_lat_lon['Hover'],
					type = 'scattermapbox'
				),
				layout = dict(
					mapbox = dict(
						layers = [],
						accesstoken = mapbox_access_token,
						style = 'light',
						center=dict(
							lat=38.72490,
							lon=-95.61446,
						),
						pitch=0,
						zoom=2.5
					)
				)
			)
		),

		html.Div([
			html.P('† Error ~= 0.0241.'
			)
		], style={'margin':20})

	], className='six columns', style={'margin':0}),

	html.Div([
		dcc.Checklist(
		    options=[{'label': 'Log scale', 'value': 'log'},
					{'label': 'Hide legend', 'value': 'hide_legend'}],
			value=[],
			labelStyle={'display': 'inline-block'},
			id='log-scale',
			style={'position': 'absolute', 'right': 80, 'top': 10}
		),
		html.Br(),
		html.P('Select chart:', style={'display': 'inline-block'}),
		dcc.Dropdown(
		    options=[

            ],
			value='show_death_rate_single_year',
			id='chart-dropdown'
		),
		dcc.Graph(
			id = 'selected-data',
			figure = dict(
				data = [dict(x=0, y=0)],
				layout = dict(
					paper_bgcolor = '#F4F4F8',
					plot_bgcolor = '#F4F4F8',
					height = 700
				)
			),
			animate = True
		)
	], className='six columns', style={'margin':0}),
])

app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})

app.run_server(debug=True)
