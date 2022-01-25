"""Post-processing functions for the ANA/CIFAR-10/VGG-8 experimental design."""

import math
import itertools
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from collections import OrderedDict
import seaborn as sbn
import matplotlib.pyplot as plt
import functools

from typing import List, Tuple, Dict
from typing import Union
from typing import Any


# SUMMARY CREATION

def make_scalar_summary_filename(scalar_tag: str) -> str:
    # works under the assumption that scalar tags are computed as described by:
    #     manager/meter/statistics/lossstatistic.py:L38
    #     manager/meter/statistics/taskstatistic.py:L38
    return '.'.join([scalar_tag.lower().replace('/', '_'), 'xlsx'])


def event_2_df(event_file: str, scalar_tag: str) -> Union[None, pd.DataFrame]:
    """Read the required scalar from a TensorBoard event file."""
    #TODO: monitor TensorBoard's `DataFrame` API for support to easier programmatic reading of TensorBoard event files (https://www.tensorflow.org/tensorboard/dataframe_api)
    ea = EventAccumulator(event_file)
    ea.Reload()
    # assert tag in ea.Tags()
    try:
        _, step_ids, values = zip(*ea.Scalars(scalar_tag))
        df = pd.DataFrame({'Global step': step_ids, scalar_tag: values})
    except KeyError:
        df = None

    return df


def export_exp_scalar_tag(dir_exp: str, scalar_tag: str) -> None:
    """Aggregate a scalar quantity's log across all folds in a given EU."""

    dir_folds = list(filter(lambda f: os.path.isdir(os.path.join(dir_exp, f)) and f.startswith('fold'), os.listdir(dir_exp)))
    dir_folds.sort()

    dfs = OrderedDict()
    for d in dir_folds:

        dir_events = os.path.join(dir_exp, d, 'stats', 'epoch')  #TODO: in this version we do not support "step" statistics
        event_files = list(filter(lambda f: os.path.isfile(os.path.join(dir_events, f)) and f.startswith('event'), os.listdir(dir_events)))
        event_files = list(map(lambda f: os.path.join(dir_events, f), event_files))
        event_files.sort(key=lambda f: os.path.getmtime(f))
        dfs_fold = list(map(lambda f: event_2_df(f, scalar_tag), event_files))
        dfs_fold = list(filter(lambda df: df is not None, dfs_fold))

        assert 1 <= len(dfs_fold)
        df = dfs_fold[0]
        for i in range(1, len(dfs_fold)):
            df = pd.concat([df.loc[~df['Global step'].isin(dfs_fold[i]['Global step'])], dfs_fold[i]], axis=0).reset_index(drop=True)
        dfs[d] = df

    tag_summary_file = os.path.join(dir_exp, 'summary', make_scalar_summary_filename(scalar_tag))
    with pd.ExcelWriter(tag_summary_file, engine='openpyxl') as ew:
        for k, v in dfs.items():
            v.to_excel(ew, sheet_name=k, index=False)


def export_scalar_tag(dir_logs: str, scalar_tag: str) -> None:
    """Summarise a scalar quantity for all the experimental units.

    An indpendent summary for each experimental unit (EU) is created,
    aggregating the quantity's values across all the folds in the unit.
    """
    dir_exps = list(filter(lambda f: os.path.isdir(os.path.join(dir_logs, f)) and f.startswith('exp'), os.listdir(dir_logs)))

    for d in dir_exps:

        dir_exp     = os.path.join(dir_logs, d)
        dir_summary = os.path.join(dir_exp, 'summary')
        if not os.path.isdir(dir_summary):
            os.makedirs(dir_summary, exist_ok=True)

        export_exp_scalar_tag(dir_exp, scalar_tag)


# VISUALISATION

def load_exp_design_log(path_edl: str) -> Tuple[Dict[str, List], pd.DataFrame]:
    """Load the log of an experimental design."""

    sheets    = pd.read_excel(path_edl, sheet_name=None)  # loads all sheets of the Excel book into a dictionary
    exp_dofs  = {k: v['name'].values.tolist() for k, v in sheets.items() if k not in {'CommandLine', 'ExperimentalUnits'}}
    exp_units = sheets['ExperimentalUnits']

    return exp_dofs, exp_units


def select_slice(exp_dofs: Dict[str, List]) -> Tuple[str, Union[str, None], Dict[str, Any]]:
    """Select the degrees of freedom to use when plotting scalar data types.

    Line plots collect several graphs of uni-variate functions. All the
    functions depicted in a line plot must have the same domain and co-domain.
    Different graphs represent different relationships between the domain and
    the co-domain. We draw the graphs together on the same plot because we
    suppose (or we know) that the corresponding relationships are conditioned
    on the value of an underlying latent variable, which we name the
    *conditioning variable*.

    Drawing too many lines on a single line plot canvas makes the plot not
    understandable. We can only draw a small, finite number of lines on the
    same canvas. Therefore, we must choose a finite number of values of the
    conditioning variable to be represented. We have usually two graphical
    aids to help users distinguish graphs according to the value of the
    categorical variables:
     * colours;
     * line styles (continuous, dashed, dotted, dash-dotted, ...).
    The conditioning variable can be continuous or discrete; when discrete,
    it can be either categorical or numerical; continuous variables are
    usually numerical. Numerical variables are intuitively mappable to
    colours: since we can define an order relationship amongst numerical
    spaces, we can assign "cold" colours (e.g., blue) to low values and "hot"
    colours (e.g., red) to high values, with a homogeneous transition for
    intermediate values. Instead, when the variable is categorical we can
    either use non-gradient palettes, or different line styles.

    In particular, in this ANA experiment the degrees of freedom are
    represented by categorical variables. From what is stated in the previous
    paragraph, we can visualise the impact of the combination of up to two
    categorical variables on the same line plot by combining non-gradient
    palettes with different line styles.

    However, in case the domain of a categorical variable is too large, the
    Cartesian product with the domain of a second categorical variable might
    contain too many data points, and plotting their combinations on the same
    plot might yield a messy result. Therefore, the user has the possibility
    of leaving the so-called "secondary conditioning variable" unspecified.

    Consider also that in large experimental designs you might have more than
    two conditioning variables. For the reasons explained above, it is better
    not to try and draw a graph for each combination of the values of the
    conditioning variables. Instead, we pick one (or two) that we desire to
    visualise, and fix the value of the remaining ones. The interpretation of
    the line plot can be explained as follows: imagine that the product
    been fixed to a combination :math:`bar{eta}`; then, the line plot will
    depict the following conditional relationships:
    :math:`f(x | (xi, bar{eta})), eta in Eta`.
    """

    def show_options(options: List[Any]) -> None:
        n_options  = math.ceil(math.log10(len(options)))
        idx_format = "{:" + "{}".format(n_options) + "d}"  # format string to align options
        for idx, o in enumerate(options):
            print((idx_format + " {}").format(idx, o))
        print("\n")

    # select primary DOF
    options = list(exp_dofs.keys())
    while True:
        print("\n")
        print("Select primary degree-of-freedom:")
        show_options(options)
        primary_dof_idx = int(input())
        if primary_dof_idx not in range(0, len(options)):
            print("Invalid choice!")
        else:
            break

    primary_dof = options[primary_dof_idx]

    # select secondary DOF
    options = [k for k in exp_dofs.keys() if k != primary_dof]
    while True:
        print("\n")
        print("Select secondary degree-of-freedom (-1 to skip):")
        show_options(options)
        secondary_dof_idx = int(input())
        if secondary_dof_idx not in range(-1, len(options)):
            print("Invalid choice!")
        else:
            break

    if secondary_dof_idx == -1:
        secondary_dof = None
    else:
        secondary_dof = options[secondary_dof_idx]

    # select combination of remaining factors
    remaining_dofs = [k for k in exp_dofs.keys() if k not in [primary_dof, secondary_dof]]
    options = list(itertools.product(*[exp_dofs[k] for k in remaining_dofs]))  #TODO: filter out those combinations that have not actually been validated during the experiment
    while True:
        print("\n")
        print("Select combination of remaining factors:")
        show_options(options)
        remaining_dofs_idx = int(input())
        if remaining_dofs_idx not in range(0, len(options)):
            print("Invalid choice!")
        else:
            break

    remaining_dofs_values = options[remaining_dofs_idx]
    remaining_dofs_map    = {k: v for k, v in zip(remaining_dofs, remaining_dofs_values)}

    return primary_dof, secondary_dof, remaining_dofs_map


def load_line_plot_data(exp_units:          pd.DataFrame,
                        primary_dof:        str,
                        secondary_dof:      Union[None, str],
                        remaining_dofs_map: Dict[str, Any],
                        dir_logs:           str,
                        scalar_tag:         str) -> pd.DataFrame:
    """Assemble the data set of measurements that are to be plotted."""
    # filter experimental units
    for k, v in remaining_dofs_map.items():
        exp_units = exp_units.loc[exp_units[k] == v]

    # load data
    id_2_df = OrderedDict()
    for id in exp_units['ID'].values:
        dir_exp      = os.path.join(dir_logs, 'exp{:>04d}'.format(id))  #TODO: the format string specifying the experimental unit's folder must be consistent with that defined in quantlab/manager/logbook/logsmanager.py:L41
        summary_file = os.path.join(dir_exp, 'summary', make_scalar_summary_filename(scalar_tag))
        id_2_df[id] = pd.concat(pd.read_excel(summary_file, sheet_name=None))

    # label DataFrames with the values of primary and secondary DOFs: in this way we can use `seaborn`'s functionalities to ease highlighting the impact of categorical variables (see `show_line_plots`)
    for id, df in id_2_df.items():
        df[primary_dof] = df[scalar_tag].map(lambda x: exp_units.loc[exp_units['ID'] == id][primary_dof].values.tolist()[0])  #TODO: `df.assign[primary_dof=...]` will add a column named `primary_dof` (i.e., it does not resolve the variable name before creating the column, hampering the programmatic extension of an existing `DataFrame`)
        if secondary_dof is not None:
            df[secondary_dof] = df[scalar_tag].map(lambda x: exp_units.loc[exp_units['ID'] == id][secondary_dof].values.tolist()[0])

    df = pd.concat(id_2_df)

    return df


def draw_line_plot(exp_dofs:      Dict[str, List],
                   df:            pd.DataFrame,
                   primary_dof:   str,
                   secondary_dof: Union[None, str],
                   scalar_tag:    str,
                   x_min:         Union[None, int, float],
                   x_max:         Union[None, int, float],
                   x_label:       str,
                   y_min:         Union[None, float],
                   y_max:         Union[None, float],
                   y_label:       str,
                   ticks_size:    int,
                   axis_size:     int,
                   legend_size:   int) -> None:
    """Draw a plot highlighting the impact of the conditioning variable(s)."""

    # fix colour and style assignment to DoFs, so that semantics is preserved across figures (https://stackoverflow.com/a/65550131)
    cmap          = sbn.color_palette('Paired')
    dof_2_colours = {dof: {k: v for k, v in zip(dof_values, cmap[1::2])} for dof, dof_values in exp_dofs.items()}
    linestyles       = [(1, 0), (1, 1), (3, 1, 1, 1), (2, 2)]  # `seaborn` seems to use a format similar to `matplotlib`'s (https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html), although without offset
    dof_2_linestyles = {dof: {k: v for k, v in zip(dof_values, linestyles)} for dof, dof_values in exp_dofs.items()}

    if secondary_dof is None:
        ax = sbn.lineplot(data=df, x='Global step', y=scalar_tag, hue=primary_dof, palette=dof_2_colours[primary_dof])  # works in conjunction with event_2_df:L34
    else:
        ax = sbn.lineplot(data=df, x='Global step', y=scalar_tag, hue=primary_dof, palette=dof_2_colours[primary_dof], style=secondary_dof, dashes=dof_2_linestyles[secondary_dof])

    # format axis
    ax.set_xlim(x_min, x_max)
    plt.xticks(fontsize=ticks_size)
    ax.set_xlabel(x_label, fontsize=axis_size)
    ax.set_ylim(y_min, y_max)
    plt.yticks(fontsize=ticks_size)
    ax.set_ylabel(y_label, fontsize=axis_size)

    # format canvas
    plt.grid()

    # format legend
    plt.setp(ax.get_legend().get_title(), fontsize=(legend_size + 2))  # legend title should be slightly larger than legend body
    plt.setp(ax.get_legend().get_texts(), fontsize=legend_size)

    # render
    plt.show()


draw_line_plot_accuracy_valid = functools.partial(draw_line_plot, scalar_tag='Accuracy/Valid', x_min=0, x_max=500, x_label='Epoch', y_min=0.0, y_max=100.0, y_label='Accuracy (%)', ticks_size=20, axis_size=24, legend_size=20)

