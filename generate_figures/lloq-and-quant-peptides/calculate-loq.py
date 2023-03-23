import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random
from lmfit import Minimizer, Parameters
plt.style.use('seaborn-whitegrid')

np.random.seed(8888)
random.seed(8888)

# Force warnings (other than FutureWarning) to kill the script; this allows debugging numpy warnings.
#import warnings
#warnings.simplefilter("error")
#warnings.simplefilter("ignore", FutureWarning)

# detect whether the file is Encyclopedia output or Skyline report, then read it in appropriately
def read_input(filename, col_conc_map_file):
    with open(filename, 'r') as f:
        header_line = f.readline()

    # if numFragments is a column, it's an Encyclopedia file
    if 'numFragments' in header_line:
        sys.stdout.write('Input identified as EncyclopeDIA *.elib.peptides.txt filetype.\n')

        df = pd.read_csv(filename, sep=None, engine='python')
        df = df.drop(['numFragments', 'Protein'], 1)  # make a quantitative df with just curve points and peptides
        col_conc_map = pd.read_csv(col_conc_map_file)
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])  # map filenames to concentrations
        df = df.rename(columns={'Peptide': 'peptide'})
        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']
        df_melted = df_melted[df_melted['curvepoint'].isin(col_conc_map['concentration'])]

    # require columns for File Name, Total Area Fragment, Peptide Sequence
    # TODO: option for Total Area Ratio?
    elif all(col in header_line for col in ['Total Area Fragment', 'Peptide Sequence', 'File Name']):
        sys.stdout.write('Input identified as Skyline export filetype. \n')

        df_melted = pd.read_csv(filename)
        df_melted.rename(columns={'File Name': 'filename'}, inplace=True)
        col_conc_map = pd.read_csv(col_conc_map_file)

        # remove any data for which there isn't a map key
        df_melted = df_melted[df_melted['filename'].isin(col_conc_map['filename'])]

        # map filenames to concentrations
        df_melted = pd.merge(df_melted, col_conc_map, on='filename', how='outer')

        # clean up column names to match downstream convention
        df_melted.rename(columns={'Total Area Fragment': 'area',
                                  'Peptide Sequence': 'peptide',
                                  'concentration': 'curvepoint'}, inplace=True)

        # remove points that didn't have a mapping (NA)
        df_melted['curvepoint'].replace('', np.nan, inplace=True)
        df_melted.dropna(subset=['curvepoint'], inplace=True)

        df_melted['area'].fillna(0, inplace=True)  # replace NA with 0

    # dia-nn output
    elif 'Stripped.Sequence' in header_line:
        sys.stdout.write('Input identified as DIA-NN *.pr_matrix.tsv filetype.\n')

        df = pd.read_table(filename, sep=None, engine='python')
        df = df.drop(['Protein.Group',
            'Protein.Ids',
            'Protein.Names',
            'Genes',
            'First.Protein.Description',
            'Proteotypic',
            'Stripped.Sequence',
            'Precursor.Charge',
            'Precursor.Id'], 1)  # make a quantitative df with just curve points and peptides
        col_conc_map = pd.read_csv(col_conc_map_file)
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])  # map filenames to concentrations
        df = df.rename(columns={'Modified.Sequence': 'peptide'})
        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']
        df_melted = df_melted[df_melted['curvepoint'].isin(col_conc_map['concentration'])]

        # remove colons in Unimod description, e.g. "AAVDC(UniMod:4)EC(UniMod:4)EFQNLEHNEK.png"
        df_melted['peptide'] = df_melted['peptide'].str.replace(':', '')
        #print(df_melted.head())

    # convert the curve points to numbers so that they sort correctly
    df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    # replace NaN values with zero
    # TODO: is this appropriate? it's required for lmfit in any case
    df_melted['area'] = df_melted['area'].fillna(0)

    return df_melted


# associates a multiplier value to the curvepoint a la single-point calibration
def associate_multiplier(df, multiplier_file):
    mutliplier_df = pd.read_csv(multiplier_file)

    # merge the multiplier with the data frame
    merged_df = pd.merge(df, mutliplier_df, on='peptide', how='inner')
    merged_df['curvepoint_multiplied'] = merged_df['curvepoint'] * merged_df['multiplier']
    multiplied_df = merged_df[['peptide', 'curvepoint_multiplied', 'area']]
    multiplied_df.columns = ['peptide', 'curvepoint', 'area']

    return multiplied_df


# yang's solve for the piecewise fit using lmfit Minimize function
def fit_by_lmfit_yang(x, y):

    def fcn2min(params, x, data, weight):
        a = params['a'].value
        b = params['b'].value
        c = params['c'].value
        model = np.maximum(c, a*x+b)
        return (model-data) * weight

    # parameter initialization
    def initialize_params(x, y):
        subsetdf = pd.DataFrame({'curvepoint': pd.to_numeric(x), 'area': y})
        mean_y = subsetdf.groupby('curvepoint')['area'].mean()  # find the mean response area for each curve point

        # find the top point, second-top point, and bottom points of the curve data
        conc_list = list(set(x))
        top_point = max(conc_list)
        conc_list.remove(top_point)
        second_top = max(conc_list)
        bottom_point = min(conc_list)

        # using the means, calculate a slope (y1-y2/x1-x2)
        linear_slope = (mean_y[second_top]-mean_y[top_point]) / (second_top-top_point)
        # find the noise intercept using average of bottom three points
        noise_intercept = mean_y[bottom_point]
        # find the linear intercept using linear slope (b = y-mx) and the top point
        linear_intercept = mean_y[top_point] - (linear_slope*top_point)

        # edge case catch?
        if noise_intercept < linear_intercept:
            noise_intercept = linear_intercept * 1.05

        return linear_slope, linear_intercept, noise_intercept

    params = Parameters()
    initial_a, initial_b, initial_c = initialize_params(x,y)
    initial_cminusb = initial_c - initial_b
    params.add('a', value=initial_a, min=0.0, vary=True)  # slope signal
    params.add('b', value=initial_b, vary=True)  # intercept signal
    params.add('c_minus_b', value=initial_cminusb, min=0.0, vary=True)
    params.add('c', expr='b + c_minus_b')

    weights = np.minimum(1 / (np.asarray(np.sqrt(x), dtype=float)+np.finfo(float).eps), 1000)  # inverse weights
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, weights))
    result = minner.minimize()

    return result, minner


# find the intersection of the noise and linear regime
def calculate_lod(model_params, df, std_mult):

    m_noise, b_noise, m_linear, b_linear = model_params

    # calculate the standard deviation for the noise segment
    if (m_noise - m_linear) == 0:
        intersection = np.inf
    else:
        intersection = (b_linear-b_noise) / (m_noise-m_linear)
    std_noise = np.std(df['area'].loc[(df['curvepoint'].astype(float) < intersection)])

    if m_linear <= 0:  # catch edge cases where there is only noise in the curve
        LOD = float('Inf')
    else:
        LOD = (b_noise + (std_mult*std_noise) - b_linear) / m_linear
    lod_results = [LOD, std_noise]

    # LOD edge cases
    curve_points = set(list(df['curvepoint']))
    curve_points.remove(min(curve_points))
    curve_points.remove(max(curve_points))  # now max is 2nd highest point
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        lod_results = [float('Inf'), float('Inf')]
    elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
        lod_results = [float('Inf'), float('Inf')]

    return lod_results


# find the intersection of the noise and linear regime
def calculate_loq(model_params, boot_results, cv_thresh=0.2):

    # initialize the known LOD and a 'blank' LOQ
    LOD = model_params[4]
    LOQ = float('Inf')

    if boot_results.empty:
        LOQ = float('Inf')
    else:
        # subset the bootstrap results for just those values above the LOD
        boot_subset = boot_results[(boot_results['boot_x']>LOD) & (boot_results['boot_cv']< cv_thresh)]

        if boot_subset.empty:
            LOQ = float('Inf')
        else:
            LOQ = boot_subset['boot_x'].min()

            # LOQ edge cases
            if LOQ >= boot_results['boot_x'].max() or LOQ <= 0:
                LOQ = float('Inf')

    return LOQ


# determine prediction interval by bootstrapping
def bootstrap_many(df, new_x, num_bootreps=100):

    def bootstrap_once(df, new_x, iter_num):

        resampled_df = df.sample(n=len(df), replace=True)
        boot_x = np.array(resampled_df['curvepoint'], dtype=float)
        boot_y = np.array(resampled_df['area'], dtype=float)
        fit_result, mini_result = fit_by_lmfit_yang(boot_x, boot_y)
        new_intersection = float('Inf')

        if fit_result.params['a'].value > 0:
            new_intersection = (fit_result.params['b'].value - fit_result.params['c'].value) /\
                               (0. - fit_result.params['a'].value)

            # consider some special edge cases
            if new_intersection > max(boot_x) or new_intersection < 0.:
                new_intersection = float('Inf')

        yresults = []
        for i in new_x:
            if np.isnan(i):
                pred_y = np.nan
            elif i <= new_intersection:  # if the new_x is in the noise,
                pred_y = fit_result.params['c'].value
            elif i > new_intersection:
                pred_y = (fit_result.params['a'].value*i) + fit_result.params['b'].value
            yresults.append(pred_y)
        iter_results = pd.DataFrame(data={'boot_x': new_x, iter_num: yresults})

        return iter_results

    if df.empty or np.isnan(new_x).any():
        boot_summary = pd.DataFrame(columns=['boot_x', 'count', 'mean', 'std', 'min',
                                             '5%', '50%', '95%', 'max', 'boot_cv'])

    else:
        # Bootstrap the data (e.g. resample the data with replacement), eval prediction (new_y) at each new_x
        boot_results = pd.DataFrame(data={'boot_x': new_x})
        for i in range(num_bootreps):
            iteration_results = bootstrap_once(df, new_x, i)

            boot_results = pd.merge(boot_results, iteration_results, on='boot_x')

        # reshape the bootstrap results to be columns=boot_x and rows=boot_y results (each iteration is a row)
        boot_results = boot_results.T
        boot_results.columns = boot_results.iloc[0]
        boot_results = boot_results.drop(['boot_x'], axis='rows')

        # calculate lower and upper 95% PI
        boot_summary = (boot_results.describe(percentiles=[.05, .95])).T
        boot_summary['boot_x'] = boot_summary.index

        # calculate the bootstrapped CV
        boot_summary['boot_cv'] = boot_summary['std']/boot_summary['mean']

    return boot_summary


# plot results
def build_plots(x, y, model_results, boot_results, std_mult):

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(5, 7))
    plt.suptitle(peptide)

    slope_noise, intercept_noise, slope_linear, intercept_linear, LOD, std_noise, LOQ = model_results

    # plot a line given a slope and intercept
    def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
        axes = plt.gca()
        xlims = np.array(axes.get_xlim())
        x_vals = np.arange(xlims[0], xlims[1], ((xlims[1] - xlims[0]) / 100))
        y_vals = intercept + slope * x_vals
        if scale == 'semilogx':
            plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        elif scale == 'loglog':
            plt.loglog(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        else:
            plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

    ###
    ### top plot: linear scale x axis
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    if not boot_results.empty:
        plt.fill_between(boot_results['boot_x'],
                         boot_results['mean']-boot_results['std'], boot_results['mean']+boot_results['std'],
                         color='y', alpha=0.3)
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    add_line_to_plot(slope_noise, (intercept_noise + (std_mult*std_noise)), 'linear', '--', setcolor='0.5')
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=LOD,
                color='m',
                label=('LOD = %.3e' % LOD))

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    plt.ylabel('signal')

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.05)


    ###
    ### bottom plot: bootstrapped CVs for discretized points
    plt.subplot(2, 1, 2)
    plt.plot(boot_results['boot_x'], boot_results['boot_cv'], marker='x', color='k', label='_nolegend_')

    plt.axvline(x=LOD,
                color='m',
                label=('LOD = %.3e' % LOD))

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    # add 20%CV reference line
    plt.axhline(y=0.20, color='r', linestyle='dashed')

    #plt.title(peptide, y=1.08)
    plt.xlabel('quantity')
    plt.ylabel('CV')

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish.
    if len(boot_results['boot_cv']) > 0:
        plt.ylim(ymin=-0.01,
                 ymax=(max(boot_results['boot_cv']*1.05)))

    # save the figure
    # add legend with LOD and LOQ values
    legend = plt.legend(loc=8, bbox_to_anchor=(0, -.75, 1., .102), ncol=2)
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,),
                bbox_inches='tight', pad_inches=0.75)
    #plt.show()
    plt.close()


# usage statement and input descriptions
parser = argparse.ArgumentParser(
    description='A  model for fitting calibration curve data. Takes calibration curve measurements as input, and \
                returns the Limit of Detection (LOD) and Limit of Quantitation (LOQ) for each peptide measured in \
                the calibration curve.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('curve_data', type=str,
                    help='a matrix containing peptides and their quantitative values across each curve point (currently\
                            supporting Encyclopedia *.elib.peptides.txt quant reports and Skyline export reports)')
parser.add_argument('filename_concentration_map', type=str,
                    help='a comma-delimited file containing maps between filenames and the concentration point \
                            they represent (two columns named "filename" and "concentration")')
parser.add_argument('--std_mult', default=2, type=float,
                    help='specify a multiplier of the standard deviation of the noise for determining limit of \
                    detection (LOD)')
parser.add_argument('--cv_thresh', default=0.2, type=float,
                    help='specify a coefficient of variation threshold for determining limit of quantitation (LOQ) \
                            (Note: this should be a decimal, not a percentage, e.g. 20%CV threshold should be input as \
                            0.2)')
parser.add_argument('--bootreps', default=100, type=int,
                    help='specify a number of times to bootstrap the data (Note: this must be an integer, e.g. to \
                            resample the data 100 times, the parameter value should be input as 100')
parser.add_argument('--multiplier_file', type=str,
                    help='use a single-point multiplier associated with the curve data peptides')
parser.add_argument('--output_path', default=os.getcwd(), type=str,
                    help='specify an output path for figures of merit and plots')
parser.add_argument('--plot', default='y', type=str,
                    help='yes/no (y/n) to create individual calibration curve plots for each peptide')
parser.add_argument('--verbose', default='n', type=str,
                    help='output a detailed summary of the bootstrapping step')

# parse arguments from command line
args = parser.parse_args()
raw_file = args.curve_data
col_conc_map_file = args.filename_concentration_map
cv_thresh = args.cv_thresh
std_mult = args.std_mult
bootreps = args.bootreps
multiplier_file = args.multiplier_file
output_dir = args.output_path
plot_or_not = args.plot
verbose = args.verbose

# read in the data
quant_df_melted = read_input(raw_file, col_conc_map_file)

# associate multiplier with the curvepoint ratio (if there is a multiplier provided)
if multiplier_file:
    quant_df_melted = associate_multiplier(quant_df_melted, multiplier_file)

# initialize empty data frame to store figures of merit
peptide_fom = pd.DataFrame(columns=['peptide', 'LOD', 'LOQ',
                                    'slope_linear', 'intercept_linear', 'intercept_noise',
                                    'stndev_noise'])

# and awwaayyyyy we go~
for peptide in tqdm(quant_df_melted['peptide'].unique()):

    subset = quant_df_melted.loc[(quant_df_melted['peptide'] == peptide)]  # subset the dataframe for that peptide

    if subset.empty:  # if the peptide is nan, skip it and move on to the next peptide
        continue

    # sort the dataframe with x values in strictly ascending order
    subset = subset.sort_values(by='curvepoint', ascending=True)

    # create the x and y arrays
    x = np.array(subset['curvepoint'], dtype=float)
    y = np.array(subset['area'], dtype=float)

    # TODO REPLACE WITH .iloc
    subset['curvepoint'] = subset['curvepoint'].astype(str)  # back to string

    # set up the model and the parameters (yang's lmfit minimize function approach)
    result, mini = fit_by_lmfit_yang(x,y)
    slope_noise = 0.0
    slope_linear = result.params['a'].value
    intercept_linear = result.params['b'].value
    intercept_noise = result.params['c'].value

    model_parameters = np.asarray([slope_noise, intercept_noise, slope_linear, intercept_linear])

    lod_vals = calculate_lod(model_parameters, subset, std_mult)
    LOD, std_noise = lod_vals
    model_parameters = np.append(model_parameters, lod_vals)

    # calculate coefficients of variation for discrete bins over the linear range (default bins=100)
    x_i = np.linspace(LOD if np.isfinite(LOD) else min(x), max(x), num=100, dtype=float)

    bootstrap_df = bootstrap_many(subset, new_x=x_i, num_bootreps=bootreps)

    if verbose == 'y':
        boot_summary.to_csv(path_or_buf=os.path.join(output_dir,
                                                     'bootstrapsummary_' + str(list(set(df['peptide']))[0]) + '.csv'),
                            index=True)

    LOQ = calculate_loq(model_parameters, bootstrap_df, cv_thresh)
    model_parameters = np.append(model_parameters, LOQ)

    if plot_or_not == 'y':
        # make a plot of the curve points and the fit, in both linear and log space
        #build_plots(x, y, model_parameters, bootstrap_df, std_mult)
        try:
            build_plots(x, y, model_parameters, bootstrap_df, std_mult)
            #continue
        except ValueError:
            sys.stderr.write('ERROR! Issue with peptide %s. \n' % peptide)


    # make a dataframe row with the peptide and its figures of merit
    new_row = [peptide, LOD, LOQ, slope_linear, intercept_linear, intercept_noise, std_noise]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ',
                                                  'slope_linear', 'intercept_linear', 'intercept_noise',
                                                  'stndev_noise'])
    peptide_fom = peptide_fom.append(new_df_row)

peptide_fom.to_csv(path_or_buf=os.path.join(output_dir, 'figuresofmerit.csv'),
                   index=False)
