import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import io
import base64
from scipy import stats
import matplotlib.pyplot as plt
import flask
import os
import matplotlib.mlab as mlab
from scipy.stats import norm
import statsmodels.api as sm

css_directory = os.getcwd()
stylesheets = ['Style.css']
static_css_route = '/static/'

app = dash.Dash(__name__)

@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                stylesheet
            )
        )
    return flask.send_from_directory(css_directory, stylesheet)


for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})



app.layout=html.Div([
    html.Div([html.H1(children="Probability plot for capability study")], style={"text-align":"center"}),
    html.Div([
    html.H3(children='Upload .csv file'),
    html.H4(children="The input data set must have a csv format"),
    dcc.Upload(id='upload-data',
               children=html.Div([
                   "Select file",
                   html.A()
                   ]),
                   style={
                       'width': '30%',
                        'height': '30px',
                        'lineHeight': '30px',
                        'borderWidth': '1px',
                        'borderStyle': 'solid',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                        }
               ),
    html.H4(children='data-name'),
    dcc.Input(id='data-name', type='text'),
    html.H4(children='Lower limit'),
    dcc.Input(id='LSL', type='number'),
    html.H4(children='Upper limit'),    
    dcc.Input(id='USL', type='number'),
    html.H4(children="Enter the quality level"),
    html.Div([
        dcc.Dropdown(id='drop-down', options=[{'label':'Ppk', 'value':'Ppk'},{'label':'ppm', 'value':'ppm'}], value='Ppk'),
        dcc.Input(id='qual-num', type='number')
        ], style={'display':'inline-block'}),    
    html.H1(children=" "),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.H1(children=" "),
    html.A(id='download', children="Download results", href='/Otsu.jpg', download='/Plots.jpg')
    ], style= {'float':'left', 'border': '1px solid black', 'padding': '10px','width':'auto', 'height':'700px'}),
    html.Div([
    html.Img(id='probability-plot')], style={'float':'right', 'width':'auto'})
    ], style={'display':'table', 'clear':'both'})
app.scripts.config.serve_locally = True 

def usl_critical_pp(dat, usl, Ppk):
    (mu, sigma) = norm.fit(dat)# obtain mean and stdev
    x_mean=0 # 0*10 quantile. in the center, quantile is zero
    y_mean=mu #dat.mean()
    x_max=3*Ppk # enter the distance from center of bell curve, in terms of number of sigmas. (usl-mu)/3Ppk = 
    y_max= usl
    x=[x_mean, x_max]
    y=[y_mean,y_max]
    return x,y
def lsl_critical_pp(dat, lsl, Ppk):
    (mu, sigma) = norm.fit(dat)# obtain mean and stdev
    x_mean=0 # 0*10 quantile. in the center, quantile is zero
    y_mean=mu #dat.mean()
    x_min=-3*Ppk # enter the distance from center of bell curve, in terms of number of sigmas. (usl-mu)/3Ppk = 
    y_min= lsl
    x=[x_min, x_mean]
    y=[y_min, y_mean]
    return x,y

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
    dff=df.iloc[:,0]
    res = stats.probplot(dff, fit=True, plot=None)# res[0][0] and res[0][1] are numpy arraye of size 240.
    quantile=res[0][0]# this is the quantile for the entered data
    dat=res[0][1]# this is the data that was inputed
    bestFit=res[1]# this should be (slope, intercept, r)
    return dat#quantile, dat
def specs(lsl,usl):
    if usl is None:# only lsl is given
        specLimit="LSL: %f"%lsl
    elif lsl is None:# only upper spec is given
        specLimit="USL: %f"%usl
    else:
        specLimit="USL: %(usl).2f and LSL: %(lsl).2f"%{'usl':usl, 'lsl':lsl}
    return specLimit

def plots(contents, lsl, usl, data_name, Ppk):
    dat = parse_contents(contents)
    # create the probability plot
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    pp = sm.ProbPlot(dat, fit=True)
    f=pp.probplot(line='s',ax=ax1, label='Data')
    dots = f.findobj(lambda x: hasattr(x, 'get_color'))
    [d.set_color('black') for d in dots]
    regionPink='Worse than critical distribution'
    regionGreen='Better than critical distribution'
    specLine='Specification limit'
    cNormal='Critical normal'
    if usl is not None:
        x_usl_critical, y_usl_critical = usl_critical_pp(dat, usl, Ppk)
        ax1.plot(x_usl_critical,y_usl_critical, color='blue', label=cNormal)
        ax1.fill_between(x_usl_critical, y_usl_critical[0], y_usl_critical, color='lightgreen', label=regionGreen)
        ax1.fill_between(x_usl_critical, usl, y_usl_critical, color='pink', label=regionPink)
        ax1.axhline(y=usl,color='red', label=specLine)
        regionGreen=''
        regionPink=''
        specLine=''
        cNormal=''
    if lsl is not None:
        x_lsl_critical, y_lsl_critical = lsl_critical_pp(dat, lsl, Ppk)
        ax1.plot(x_lsl_critical,y_lsl_critical, color='blue', label=cNormal)
        ax1.fill_between(x_lsl_critical, lsl, y_lsl_critical, color='pink', label=regionPink)
        ax1.fill_between(x_lsl_critical, y_lsl_critical[1], y_lsl_critical, color='lightgreen', label=regionGreen)
        ax1.axhline(y=lsl,color='red', label=specLine)
    ax1.set_ylabel(data_name)
    ax1.set_xlabel('Percentage')
    ax1.set_title("Probability plot")
    ax1.legend(loc='best')
    specLine='Specification limit'
    cNormal='Critical normal'
    # create the histogram
    n, bins, patches =plt.hist(dat,50, normed=1, orientation=u'horizontal', color='grey')
    # add a 'best fit' line
    (mu, sigma) = norm.fit(dat)# obtain mean and stdev
    y = mlab.normpdf( bins, mu, sigma)# best fit y values for the normal distribution of dat.
    ax2.plot(y, bins, 'black', linewidth=1, label='Fitted normal')
    
    if usl is not None:
        sigma_usl_critical= (usl-mu)/3/Ppk
        bins=[ mu+i*(usl-mu)/20 for i in range(20)]
        y_usl_critical = mlab.normpdf( bins, mu, sigma_usl_critical) 
        ax2.plot(y_usl_critical, bins, 'blue', linewidth=1, label=cNormal)
        ax2.axhline(y=usl, color='red', label=specLine)
        specLine=''
        cNormal=''
    if lsl is not None:
        sigma_lsl_critical= (mu-lsl)/3/Ppk
        bins=[ mu-i*(mu-lsl)/20 for i in range(20)]
        y_lsl_critical = mlab.normpdf( bins, mu, sigma_lsl_critical)
        ax2.plot(y_lsl_critical, bins, 'blue', linewidth=1, label=cNormal)
        ax2.axhline(y=lsl, color='red', label=specLine)
    ax2.set_ylabel(data_name)
    ax2.set_xlabel('Density')
    ax2.set_title("Histogram")
    ax2.legend(loc='best')
    fig.set_size_inches(15, 10)
    fig.suptitle(
        'Sample size: %(samSize)d \n Mean: %(mean).2f\n StDev: %(StDev).2f \n Spefication limits: %(specs)s \n Ppk = %(Ppk).2f'
        %{'samSize':dat.shape[0], 'mean':norm.fit(dat)[0], 'StDev':norm.fit(dat)[1], 'specs':specs(lsl,usl),'Ppk':Ppk }
        )
    plt.savefig('plots.png')
    plt.gcf().clear()
    return 'plots.png'

@app.callback(Output('probability-plot', 'src'),
              [Input('submit-button', 'n_clicks')],              
              [State('upload-data', 'contents'), State('USL', 'value'), State('LSL', 'value'),
               State('data-name', 'value'),State('drop-down', 'value'), State('qual-num', 'value') ]
              )
def plot_graph(n_clicks, contents, usl, lsl, data_name, drop_down, qual_num):
    if drop_down=='ppm':
        sigmas=norm.ppf((1000000-qual_num)/1000000)# this gives you how many sigmas are from mean to usl or lsl
        Ppk=sigmas/3
    else:
        Ppk=qual_num
    image_filename = plots(contents, lsl, usl, data_name, Ppk)# this is an imageof the probability plot
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    src='data:image/png;base64,{}'.format(encoded_image.decode())
    return src
if __name__ == '__main__':
    app.run_server(debug=True)
