import matplotlib.pyplot as plt

color_NH3 = u'#006ba4'
color_He = u'#c85200'

he_model_line = {'label' : r'Anal. He',
        'linestyle' : '-',
        'dashes' : [4,10],
        'color' : color_He}

nh3_model_line = {'label' : r'Anal. NH$_3$',
        'linestyle' : '-',
        'dashes' : [4,10],
        'color' : color_NH3}

he_dsmc_line = {'label' : r'DSMC He',
        'linestyle' : '-',
        'color' : color_He}

nh3_dsmc_line = {'label' : r'DSMC NH$_3$',
        'linestyle' : '-',
        'color' : color_NH3}

def new_figure(**args):
    #fig_width = 3.38
    fig_width = 20 / 2.54  # in cm
    fig_height = 15 / 2.54  # in cm
    plt.rcParams['font.family'] = 'sans-serif' #'serif' for reports
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['lines.linewidth'] = 1.3
    plt.rcParams['legend.columnspacing'] = 0.2
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
    plt.rcParams['lines.dash_joinstyle'] = 'round' # miter|round|bevel
    plt.rcParams['lines.dash_capstyle'] = 'round'          # butt|round|projecting
    plt.rcParams['lines.solid_joinstyle'] = 'round'       # miter|round|bevel
    plt.rcParams['lines.solid_capstyle'] = 'round'   # butt|round|projecting
    plt.rcParams['lines.antialiased'] = True         # render lines in antialised (no jaggies)
    return plt.subplots(figsize=(fig_width, fig_height), dpi=100, **args)
    #return plt.subplots(figsize=(fig_width, fig_width*0.72), dpi=200)


