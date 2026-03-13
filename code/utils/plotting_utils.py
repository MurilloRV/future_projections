import pandas as pd
import numpy as np
import math
import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rc("axes",labelsize="large")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['text.usetex'] = False

from scipy import interpolate
# import scipy.stats 
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

from iminuit import Minuit
from iminuit.cost import LeastSquares


def scatter_plot_2D(
    x_data,
    y_data,
    color_data,
    x_label={'label':'', 'fontsize':10},
    y_label={'label':'', 'fontsize':10},
    figsize=(4.5, 4.0),
    color_map=mpl.colormaps['tab20c'],
    plot_colorbar=True,
    colorbar_label='',
    upper_right_text='',
    x_range=None,
    y_range=None,
    **scatter_kwargs
):
    """
    Generic function to create a 2D scatter plot with color coding

    Parameters
    ----------
    x_data : array-like
        Data for the x-axis.
    y_data : array-like
        Data for the y-axis.
    color_data : array-like
        Data for color coding the points.
    x_label : dict, optional
        Dictionary with keys 'label' and 'fontsize' for the x-axis label.
    y_label : dict, optional
        Dictionary with keys 'label' and 'fontsize' for the y-axis label.
    figsize : tuple, optional
        Figure size.
    color_map : matplotlib colormap, optional
        Colormap for the scatter plot.
    x_range : tuple, optional
        Range for the x-axis.
    y_range : tuple, optional
        Range for the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    """

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True
    )

    sc = ax.scatter(
        x_data,
        y_data,
        c=color_data,
        cmap=color_map,
        rasterized=True,
        **scatter_kwargs,
    )

    if x_range is not None: ax.set_xlim(*x_range)
    if y_range is not None: ax.set_ylim(*y_range)

    # Check if colorbar has been plotted already
    if plot_colorbar:
        clb = fig.colorbar(
            sc,
            ax=ax,
            label=colorbar_label,
            pad=0,
            fraction=0.1,
            location='right',
            aspect=40,
            extend='max',
        )
        clb.ax.minorticks_on()

    ax.set_xlabel(x_label['label'], fontsize=x_label['fontsize'])
    ax.set_ylabel(y_label['label'], fontsize=y_label['fontsize'])

    ax.text(
        1,
        1,
        upper_right_text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )

    # plt.tight_layout()

    return fig, ax



def read_data_files(
    data_file,
    func_read_file,
    func_read_file_args={},
    BR_constraints=None,
    max_num_bulk_points=None,
):
    print(f"Processing data file: {data_file}")
    df = pd.read_csv(data_file + ".csv")
    df = df.replace("", np.nan)
    df = df.dropna().reset_index(drop=True)

    if max_num_bulk_points is not None:
        df = df[:max_num_bulk_points]
    n_pts = df.shape[0]

    kappas, EWPOs, model_pars = func_read_file(df, **func_read_file_args)

    if not BR_constraints is None:
        satisfy_BR_constraint = [True for i in range(n_pts)]
        for i in range(n_pts):
            for coup in ['uu', 'dd', 'cc', 'ss', 'tt', 'bb', 'ee', 'mumu', 'tautau', 'WW', 'ZZ', 'Zgam', 'gamgam']:
                if np.abs(kappas[coup][i] - 1.) > BR_constraints:
                    satisfy_BR_constraint[i] = False

        for coup in kappas.keys():
            kappas[coup] = np.array(kappas[coup][satisfy_BR_constraint])

        n_pts_BR_constraint = len(kappas['ZZ'])
        print(f"Number of points satisfying the {BR_constraints*100:.3g}% SM constraint on the single higgs couplings: {n_pts_BR_constraint} / {n_pts}")
    else:
        n_pts_BR_constraint = n_pts

    return kappas, EWPOs, model_pars, n_pts, n_pts_BR_constraint



def generate_plot(fig,
                  ax,
                  kappas,
                  plot_colorbar,
                  zoom=False, 
                  zoom_range_x=(2 - 3*0.25, 2 + 3*0.25),
                  zoom_range_y=(0.0, 0.005),
                  ):
    """
    Function called by plot_EffZZH_240_vs_365 to actually plot the 
    scatter (bulk) points in the (k_Zh^240, k_Zh^365) plane
    """
    
    color = np.array(kappas['lam'], dtype=complex).real

    if not isinstance(kappas['ZZ_365'], np.ndarray):
        kappas['ZZ_365'] = kappas['ZZ_365'].to_numpy()
    if not isinstance(kappas['ZZ_240'], np.ndarray):
        kappas['ZZ_240'] = kappas['ZZ_240'].to_numpy()

    x_data = kappas['ZZ_365'] - 1
    y_data = kappas['ZZ_240'] - 1

    color_map = mpl.colormaps['tab20c']

    sc = ax.scatter(
        x_data,
        y_data,
        c=color,
        s=0.4,
        vmin=1.,
        vmax=12.,
        cmap=color_map,
        rasterized=True)

    if zoom==True:
        ax.set_xlim(*zoom_range_x)
        ax.set_ylim(*zoom_range_y)

    # Check if colorbar has been plotted already
    if plot_colorbar:
        clb = fig.colorbar(
            sc,
            ax=ax,
            label=r"$\kappa_\lambda$",
            pad=0,
            fraction=0.1,
            location='right',
            aspect=40,
            extend='max',
        )
        clb.ax.minorticks_on()

        plot_colorbar = False

    return plot_colorbar


def plot_EffZZH_240_vs_365(data_file, 
                           func_read_file,
                           func_read_file_args={},
                           zoom=False, 
                           zoom_range_x=(2 - 3*0.25, 2 + 3*0.25),
                           zoom_range_y=(0.0, 0.005),
                           plot_points=None,
                           point_colors=None,
                           point_leg_columns=2,
                           point_leg_size=6,
                           point_marker_size=10,
                           markeredgecolor="white",
                           markeredgewidth=0.5,
                           point_markers="*",
                           BR_constraints=None,
                           plot_self_consistent_curve=None,
                           model_name="",
                           no_model_text=False,
                           upper_right_text=None,
                           figsize=(4.5, 4.0),
                           legend_loc="upper left",
                           max_num_bulk_points=None,
                           ):
    """
    Function to creates the 2D plot in the (k_Zh^240, k_Zh^365) plane, given a set of model points

    Parameters
    ----------
    data_file : str
        The path to the input data file (without the .csv extension), for the bulk model points.
    func_read_file : callable
        A function to read the csv data file for the model and extract the values for kappa_lambda, 
        the EWPOs, and the model parameters
    func_read_file_args : dict, optional
        Arguments to be passed to func_read_file
    zoom : bool, optional
        Whether to zoom into the plot. Default is False
    zoom_range_x : tuple, optional
        The x-axis limits for the zoomed-in plot
    zoom_range_y : tuple, optional
        The y-axis limits for the zoomed-in plot
    plot_points : list of tuples, optional
        List of points to plot. Each point corresponds to a tuple
        with the following entries:
        - float: x coordinate (k_Zh^365)
        - float: y coordinate (k_Zh^240)
        - float: kappa_lambda for given point
        - str: label for the point, shown in the legend
    point_colors : str or list of str, optional
        List of colors for the plotted points. If None (Default), the 
        tab20b colormap is used. If `lambdas`, then the colors follow
        tab20c according to the value of kappa_lambda
    point_leg_columns : int, optional
        Number of columns for the legend of the points. Default is 2.
    point_leg_size : int or list of int, optional
        Font size for the legend of the points. Default is 6.
    point_marker_size : float or list of float, optional
        Size of the markers for the points. Default is 10.
    markeredgecolor : str or list of str, optional
        Edge color for the markers. Default is "white".
    markeredgewidth : float or list of float, optional
        Edge width for the markers. Default is 0.5.
    point_markers : str or list of str, optional
        Marker style for the points. Default is "*".
    BR_constraints : float, optional
        Branching ratio constraints for the bulk points. Default is None.
    plot_self_consistent_curve : str of list of str, optional
        Whether to plot a curve corresponding to self-consistent fits. 
        Must be either "CH", "CHbox", or a list ["CH", "CHbox"]. Default is None.
    model_name : str, optional
        Name of the model considered, to be show in the model text
    no_model_text : bool, optional
        Whether to exclude the model/lower text. Default is False
    upper_right_text : str, optional
        Text to be shown on the top left of the plot
    figsize : tuple, optional
        Size of the created figure. Default is (4.5, 4.0)
    legend_loc : str, optional
        Location of the legend for the plot. Default is "upper left"
    max_num_bulk_points : int, optional
        Maximum number of bulk points to read and plot. If set to None (default), 
        all points in the data file are considered.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The primary axes of the figure.

    """
    

    fig, ax = plt.subplots(
                figsize=figsize,
                constrained_layout=True)

    plot_colorbar = True

    if isinstance(data_file, str):
        kappas, EWPOs, model_pars, n_pts, n_pts_BR_constraint = read_data_files(
            data_file,
            func_read_file,
            func_read_file_args=func_read_file_args,
            BR_constraints=BR_constraints,
            max_num_bulk_points=max_num_bulk_points,
        )
        n_pts_BR_constraint_total = n_pts_BR_constraint
        n_pts_total = n_pts

        generate_plot(fig=fig,
                      ax=ax,
                      kappas=kappas,
                      zoom=zoom, 
                      zoom_range_x=zoom_range_x,
                      zoom_range_y=zoom_range_y,
                      plot_colorbar=plot_colorbar,
                      )

    elif isinstance(data_file, list) and all(isinstance(item, str) for item in data_file):

        n_pts_total = 0
        n_pts_BR_constraint_total = 0

        for file in data_file:
            kappas, EWPOs, model_pars, n_pts, n_pts_BR_constraint = read_data_files(
                file,
                func_read_file,
                func_read_file_args=func_read_file_args,
                BR_constraints=BR_constraints,
                max_num_bulk_points=max_num_bulk_points,
            )
            n_pts_BR_constraint_total += n_pts_BR_constraint
            n_pts_total += n_pts

            plot_colorbar = generate_plot(fig=fig,
                                          ax=ax,
                                          kappas=kappas,
                                          zoom=zoom, 
                                          zoom_range_x=zoom_range_x,
                                          zoom_range_y=zoom_range_y,
                                          plot_colorbar=plot_colorbar
                                          )

    else:
        raise ValueError("Input must be a string or a list of strings.")
    

    ax.set_xlabel(r'$\kappa_{Zh}^{365} - 1$', fontsize=14)
    ax.set_ylabel(r'$\kappa_{Zh}^{240} - 1$', fontsize=14)

    lower_text = model_name
    if BR_constraints:
        lower_text = lower_text + f"\n{BR_constraints*100:.3g}% SM constraints"
        lower_text = lower_text + f"\n{n_pts_BR_constraint_total} / {n_pts_total} Points"

    else:
        lower_text = lower_text + f'\n{n_pts_total} Points'+'\n'
    
    if not no_model_text:
        ax.text(
            0.97,
            0.05,
            lower_text,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=8
            )

    ax.text(
        1,
        1,
        upper_right_text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )


    if not plot_self_consistent_curve is None:

        legend_loc = "best"
        colors = mpl.colormaps['hsv']

        if isinstance(plot_self_consistent_curve, str):

            leg_handles = []

            lambdas=[]
            k_ZH_240 = {}
            k_ZH_365 = {}

            if plot_self_consistent_curve=='CH':
                file_name = "/cephfs/user/mrebuzzi/phd/HEPfit/HEPfit_snowmass21/Fits_HLLHC_FCCee/large_kappa_lambda_fits/comparison_plots/k_ZH_240_365_predictions.txt"
            elif plot_self_consistent_curve=='CHbox':
                file_name = "/cephfs/user/mrebuzzi/phd/HEPfit/HEPfit_snowmass21/Fits_HLLHC_FCCee/large_kappa_lambda_fits_CHbox/comparison_plots/k_ZH_240_365_predictions.txt"
            else:
                raise ValueError("\'plot_self_consistent_curve\' must be a valid Wilson coefficient!")

            with open(file_name, "r") as self_consistent_results_file:
                lines = self_consistent_results_file.readlines()
                for n, line in enumerate(lines):
                    columns = line.split()

                    if plot_self_consistent_curve=='CH':
                        lmbd = int(columns[0])
                    if plot_self_consistent_curve=='CHbox':
                        lmbd = float(columns[0])

                    lambdas.append(lmbd)
                    if columns[1].startswith("eeZH_FCCee240"):
                        k_ZH_240[lmbd] = np.sqrt(float(columns[2]))
                    elif columns[1].startswith("eeZH_FCCee365"):
                        k_ZH_365[lmbd] = np.sqrt(float(columns[2]))


            lambdas_unique = [lmbd for i, lmbd in enumerate(lambdas) if i%2==0 ]
            
            color = list(colors(np.linspace(0.001, 0.9, len(lambdas_unique))[::-1]))

            ax.plot([(k_ZH_365[lmbd]-1) for lmbd in lambdas_unique],
                    [(k_ZH_240[lmbd]-1) for lmbd in lambdas_unique],
            )

            for i, lmbd in enumerate(lambdas_unique):
                ax.plot((k_ZH_365[lmbd]-1), (k_ZH_240[lmbd]-1), marker="*", ls="none", c=color[i], markersize=10, markeredgecolor='white', markeredgewidth=0.5)

                point_label = rf"$\kappa_\lambda$ = {lmbd}"
                leg_handles.append(Line2D([0], [0], color="k", ls="none", marker="*", c=color[i], markersize=10, markeredgecolor='white', markeredgewidth=0.5, label=point_label))

            hsLegend = ax.legend(
                        handles=leg_handles,
                        loc=legend_loc,
                        frameon=False,
                        prop={'size': 6},
                        ncol=2,
                    )
            ax.add_artist(hsLegend)

        elif isinstance(plot_self_consistent_curve, list) and all(isinstance(item, str) for item in plot_self_consistent_curve):
            if plot_self_consistent_curve==["CH", "CHbox"]:

                lambdas = {}
                k_ZH_240 = {}
                k_ZH_365 = {}
                for WC in plot_self_consistent_curve:

                    leg_handles = []
                    lambdas[WC]=[]
                    k_ZH_240[WC] = {}
                    k_ZH_365[WC] = {}

                    if WC=='CH':
                        file_name = "/cephfs/user/mrebuzzi/phd/HEPfit/HEPfit_snowmass21/Fits_HLLHC_FCCee/large_kappa_lambda_fits/comparison_plots/k_ZH_240_365_predictions.txt"
                        legend_loc='upper left'
                        marker="*"
                        color_line="tab:blue"
                        label_line=r'$C_{H}$'
                        markersize=10
                    elif WC=='CHbox':
                        file_name = "/cephfs/user/mrebuzzi/phd/HEPfit/HEPfit_snowmass21/Fits_HLLHC_FCCee/large_kappa_lambda_fits_CHbox/comparison_plots/k_ZH_240_365_predictions.txt"
                        legend_loc='lower right'
                        marker="s"
                        color_line="tab:orange"
                        label_line='$C_{H\u25A1}$'
                        markersize=5


                    with open(file_name, "r") as self_consistent_results_file:
                        lines = self_consistent_results_file.readlines()
                        for n, line in enumerate(lines):
                            columns = line.split()

                            if WC=='CH':
                                lmbd = int(columns[0])
                            if WC=='CHbox':
                                lmbd = float(columns[0])

                            lambdas[WC].append(lmbd)
                            if columns[1].startswith("eeZH_FCCee240"):
                                k_ZH_240[WC][lmbd] = np.sqrt(float(columns[2]))
                            elif columns[1].startswith("eeZH_FCCee365"):
                                k_ZH_365[WC][lmbd] = np.sqrt(float(columns[2]))


                    lambdas_unique = [lmbd for i, lmbd in enumerate(lambdas[WC]) if i%2==0 ]
                    
                    color = list(colors(np.linspace(0.001, 0.9, len(lambdas_unique))[::-1]))

                    ax.plot([(k_ZH_365[WC][lmbd]-1) for lmbd in lambdas_unique],
                            [(k_ZH_240[WC][lmbd]-1) for lmbd in lambdas_unique],
                            color=color_line,
                    )
                    leg_handles.append(Line2D([0], [0], color="k", ls="-", c=color_line, label=label_line))

                    for i, lmbd in enumerate(lambdas_unique):
                        ax.plot((k_ZH_365[WC][lmbd]-1), (k_ZH_240[WC][lmbd]-1), marker=marker, ls="none", c=color[i], markersize=markersize, markeredgecolor="white", markeredgewidth=0.5)

                        point_label = rf"$\kappa_\lambda$ = {lmbd}"
                        leg_handles.append(Line2D([0], [0], color="k", ls="none", marker=marker, c=color[i], markersize=markersize, markeredgecolor="white", markeredgewidth=0.5, label=point_label))

                    hsLegend = ax.legend(
                        handles=leg_handles,
                        loc=legend_loc,
                        frameon=False,
                        prop={'size': 6},
                        ncol=2,
                    )
                    ax.add_artist(hsLegend)
        


    if not plot_points is None:

        leg_handles = []

        colors = mpl.colormaps['tab20b']
        color = list(colors(np.linspace(0.001, 0.999, len(plot_points)+1)[::-1]))

        if not point_colors is None:
            if point_colors == "lambdas":
                color = np.array(kappas['lam'], dtype=complex).real
                color_map = mpl.colormaps['tab20c']
                norm = Normalize(vmin=1.0, vmax=12.)
                color = [ color_map(norm(pt[3])) for pt in plot_points]
            else:
                color = point_colors


        for arg in [point_marker_size, markeredgecolor, markeredgewidth, point_markers]:
            if isinstance(arg, list) and len(arg) != len(plot_points):
                raise ValueError(f"Length of {arg} must be equal to the number of points in plot_points or a single integer/float.")

        if isinstance(point_marker_size, (int, float)):
            point_marker_size = [point_marker_size for i in range(len(plot_points))]
        if isinstance(markeredgewidth, (int, float)):
            markeredgewidth = [markeredgewidth for i in range(len(plot_points))]
        if isinstance(markeredgecolor, (str)):
            markeredgecolor = [markeredgecolor for i in range(len(plot_points))]
        if isinstance(point_markers, str):
            point_markers = [point_markers for i in range(len(plot_points))]

            
        for i, plot_point in enumerate(plot_points):
            ax.plot(*(plot_point[:2]), marker=point_markers[i], ls="none", c=color[i], markersize=point_marker_size[i], markeredgecolor=markeredgecolor[i], markeredgewidth=markeredgewidth[i])
            point_label = f"{plot_point[3]}"
            # point_label = f"{plot_point[2]}\n($\kappa_{{\lambda}}={plot_point[0]:.3g}$, $\kappa_{{Z}}-1={plot_point[1]:.3g}$)"
            leg_handles.append(Line2D([0], [0], color="k", ls="none", marker=point_markers[i], c=color[i], markersize=point_marker_size[i], markeredgecolor=markeredgecolor[i], markeredgewidth=markeredgewidth[i], label=point_label))
            
            # if zoom: legend_loc = 'center right'

        hsLegend = ax.legend(
            handles=leg_handles,
            loc=legend_loc,
            frameon=False,
            prop={'size': point_leg_size},
            ncol=point_leg_columns,
        )
        ax.add_artist(hsLegend)

    return fig, ax



def plot_self_consistent_curve(curves, 
                               files, 
                               fig, 
                               ax, 
                               labels, 
                               colors_line, 
                               markers, 
                               markersizes, 
                               n_legends=1
):
    """
    Plots sets of SMEFT self-consistent curves in the (k_Zh^240, k_Zh^365) plane

    Parameters
    ----------
    curves : str or list of str
        The name of the Wilson coefficient(s) used to generate the self-consistent curve(s)
    files : str or list of str
        The path(s) to the text file(s) containing the k_Zh values for the self-consistent curve(s)
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The primary axes of the figure.
    labels : str or list of str
        The label(s) for the self-consistent curve(s)
    colors_line : str or list of str
        The color(s) for the line(s) representing the self-consistent curve(s)
    markers : str or list of str
        The marker(s) for the points on the self-consistent curve(s)
    markersizes : int or list of int
        The size(s) of the markers for the points on the self-consistent curve(s)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The updated figure.
    ax : matplotlib.axes.Axes
        The updated primary axes of the figure.

    Notes
    -----
    If any among curves, files, labels, colors_line, markers, and markersizes are lists, then all 
    must be lists of the same length

    """
    if not curves is None:
        
        legend_loc = "best"
        colors = mpl.colormaps['hsv']

        if all([isinstance(arg, str) for arg in (curves, files, labels, colors_line, markers, markersizes)]):

            leg_handles = []

            lambdas=[]
            k_ZH_240 = {}
            k_ZH_365 = {}

            file_name = files

            with open(file_name, "r") as self_consistent_results_file:
                lines = self_consistent_results_file.readlines()
                for n, line in enumerate(lines):
                    columns = line.split()

                    if curves=='CH':
                        lmbd = int(columns[0])
                    if curves=='CHbox':
                        lmbd = float(columns[0])

                    lambdas.append(lmbd)
                    if columns[1].startswith("eeZH_FCCee240"):
                        k_ZH_240[lmbd] = np.sqrt(float(columns[2]))
                    elif columns[1].startswith("eeZH_FCCee365"):
                        k_ZH_365[lmbd] = np.sqrt(float(columns[2]))

            lambdas_unique = [lmbd for i, lmbd in enumerate(lambdas) if i%2==0 ]
            
            color = list(colors(np.linspace(0.001, 0.9, len(lambdas_unique))[::-1]))

            ax.plot([(k_ZH_365[lmbd]-1) for lmbd in lambdas_unique],
                    [(k_ZH_240[lmbd]-1) for lmbd in lambdas_unique],
            )

            for i, lmbd in enumerate(lambdas_unique):
                ax.plot((k_ZH_365[lmbd]-1), (k_ZH_240[lmbd]-1), marker="*", ls="none", c=color[i], markersize=10, markeredgecolor='white', markeredgewidth=0.5)

                point_label = rf"$\kappa_\lambda$ = {lmbd}"
                leg_handles.append(Line2D([0], [0], color="k", ls="none", marker="*", c=color[i], markersize=10, markeredgecolor='white', markeredgewidth=0.5, label=point_label))

            hsLegend = ax.legend(
                        handles=leg_handles,
                        loc=legend_loc,
                        frameon=False,
                        prop={'size': 6},
                        ncol=2,
                    )
            ax.add_artist(hsLegend)


        elif all(isinstance(arg, list) for arg in (curves, files, labels, colors_line, markers, markersizes)) \
            and all( len(curves) == len(arg) for arg in (files, labels, colors_line, markers, markersizes)):
            # and all( isinstance(item, str) for arg in (curves, files, labels, colors_line, markers, markersizes) for item in arg) \

            # if curves==["CH", "CHbox"]:

            lambdas = {}
            k_ZH_240 = {}
            k_ZH_365 = {}
            
            for wc_index, (WC, file_name, label_line, color_line, marker, markersize) in enumerate(zip(curves, files, labels, colors_line, markers, markersizes)):

                if wc_index == 0:
                    leg_handles = []
                    leg_handles_lines = []
                
                lambdas[WC] = []
                k_ZH_240[WC] = {}
                k_ZH_365[WC] = {}

                if WC=='CH':
                    legend_loc='upper left'
                elif WC=='CHbox':
                    legend_loc='lower right'

                legend_loc_lines='lower right'


                with open(file_name, "r") as self_consistent_results_file:
                    lines = self_consistent_results_file.readlines()
                    for n, line in enumerate(lines):
                        columns = line.split()

                        if WC=='CH':
                            lmbd = int(columns[0])
                        if WC=='CHbox':
                            lmbd = float(columns[0])

                        lambdas[WC].append(lmbd)
                        if columns[1].startswith("eeZH_FCCee240"):
                            k_ZH_240[WC][lmbd] = np.sqrt(float(columns[2]))
                        elif columns[1].startswith("eeZH_FCCee365"):
                            k_ZH_365[WC][lmbd] = np.sqrt(float(columns[2]))

                lambdas_unique = [lmbd for i, lmbd in enumerate(lambdas[WC]) if i%2==0 ]
                
                color = list(colors(np.linspace(0.001, 0.9, len(lambdas_unique))[::-1]))

                ax.plot([(k_ZH_365[WC][lmbd]-1) for lmbd in lambdas_unique],
                        [(k_ZH_240[WC][lmbd]-1) for lmbd in lambdas_unique],
                        color=color_line,
                )
                leg_handles_lines.append(Line2D([0], [0], color="k", ls="-", c=color_line, label=label_line))

                for i, lmbd in enumerate(lambdas_unique):
                    ax.plot((k_ZH_365[WC][lmbd]-1), (k_ZH_240[WC][lmbd]-1), marker=marker, ls="none", c=color[i], markersize=markersize, markeredgecolor='white', markeredgewidth=0.5)

                    point_label = rf"$\kappa_\lambda$ = {lmbd}"
                    if not (n_legends == 1 and wc_index != (len(curves) - 1)):
                        leg_handles.append(Line2D([0], [0], color="k", ls="none", marker=marker, c=color[i], markersize=markersize, markeredgecolor='white', markeredgewidth=0.5, label=point_label))


                if not (n_legends == 1 and wc_index != (len(curves) - 1)):
                    hsLegend = ax.legend(
                        handles=leg_handles,
                        loc=legend_loc,
                        frameon=False,
                        prop={'size': 6},
                        ncol=2,
                    )
                    ax.add_artist(hsLegend)

                    hsLegend_lines = ax.legend(
                        handles=leg_handles_lines,
                        loc=legend_loc_lines,
                        frameon=False,
                        prop={'size': 5},
                        ncol=1,
                    )
                    ax.add_artist(hsLegend_lines)

        else:
            raise ValueError("Please specify valid curves / files!")
        
    return fig, ax




def plot_curves (
    ax,
    curves, 
    file_names, 
    data_sets, 
    legend_locs, 
    legend_cols, 
    legend_fontsizes, 
    markers, 
    color_lines, 
    label_lines, 
    markersizes, 
    input_file_layout = "klam",
    klam_in_point_labels=r"$\kappa_\lambda$",
    color_independent=False,
    max_n_markers=None,
):
    """
    Function to plot curves in the (k_Zh^240, k_Zh^365) plane. Input data must
    be given either by a text file (with its path in the file_names list) or
    more directly by a python list (in the data_sets list).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the curves.
    curves : list of str
        The names of the curves to plot.
    file_names : list of str
        The file names containing the data for each curve.
    data_sets : list of list of tuples
        The data sets for each curve, if not using files. Must be complementary 
        to file_names, i.e., for each index i, exactly one of file_names[i] or
        data_sets[i] must be different than None.
        Each entry must be a tuple of the form:
        - float: x coordinate (k_Zh^365)
        - float: y coordinate (k_Zh^240)
        - float: kappa_lambda (or delta_kappa_lambda) for given point
        - str: label for the point, shown in the legend
    legend_locs : list of str
        The location of the legend for each curve. For each entry different
        from None, a new, separate legend is plotted. If an entry is None, this
        curve will share be shown in the most recently created legend. If all 
        entries are None, no legend is plotted.
    legend_cols : list of int
        The number of columns in each legend. Each entry is only relevant if
        the corresponding entry in legend_locs is not None.
    legend_fontsizes : list of int
        The font sizes for the legend text for each curve.
    markers : list of str
        The marker styles for the points forming each curve.
    color_lines : list of str
        The colors for each curve.
    label_lines : list of str
        The labels for each curve.
    markersizes : list of int
        The sizes of the markers for each curve.
    input_file_layout : str, optional
        Layout of the input files. The "klam" layout (default) expects files to be formatted with
        columns corresponding to (kappa_lambda, observable, central_value), where `observable`, 
        a string, is either 'eeZH_FCCee240' or 'eeZH_FCCee365'. The "CH_CHbox" layout
        is for files formatted with (CH, CHbox, observable, central_value).
    klam_in_point_labels : str, optional
        The points are labeled according to their value of kappa_lambda or a related 
        quantity. This argument specifies the latex symbol for this quantity. Default 
        is r"$\kappa_\lambda$". If set to None, the markers are not shown in the legend. 
    color_independent : bool, optional
        The color scheme for the markers follows the `hsv` colormap. The color is assigned
        based on the value of kappa_lambda for each point, normalized within the [-5, 12]
        interval. If the color_independent option is set to True, however, the marker colors 
        will simply be equally spaced within the colormap. Default is False.
    max_n_markers : int, optional
        The maximum number of markers to display for each curve. If set to None (default), all 
        markers are shown.

    Returns
    -------
    k_ZH_240 : dict
        A dictionary containing the k_Zh^240 values for each curve, with the curve names as keys.
    k_ZH_365 : dict
        A dictionary containing the k_Zh^365 values for each curve, with the curve names as keys.

    Notes
    -----
    All list-type arguments must have the same length, matching the number of curves to be plotted.
    If the input layout "CH_CHbox" is used, an error will be raised if the user tries to show the 
    markers in the plot legend for a curve obtained from a .txt file, since no predictions for 
    kappa_lambda are available in these.

    """

    # check all arguments are lists with the same length
    for args in [file_names, 
        data_sets, 
        legend_locs, 
        legend_cols, 
        legend_fontsizes, 
        markers, 
        color_lines, 
        label_lines, 
        markersizes, 
        ]:
        if not isinstance(args, list):
            raise ValueError(f"Expected a list, got {type(args)} instead.")
        elif len(args) != len(curves):
            raise ValueError(f"Length of {args} must be equal to the number of curves ({len(curves)}).")

    if input_file_layout not in ["klam", "CH_CHbox"]:
        raise ValueError(f"Unknown input file layout: {input_file_layout}")

    color_map = mpl.colormaps['hsv']
    norm = Normalize(vmin=-5.0, vmax=12.)

    k_ZH_240 = {}
    k_ZH_365 = {}

    leg_handles = []

    for index, curve in enumerate(curves):

        file_name       = file_names[index]
        data_set        = data_sets[index]
        legend_loc      = legend_locs[index]
        legend_col      = legend_cols[index]
        legend_fontsize = legend_fontsizes[index]
        marker          = markers[index]
        color_line      = color_lines[index]
        label_line      = label_lines[index]
        markersize      = markersizes[index]

        lambdas = []
        k_ZH_240[curve] = []
        k_ZH_365[curve] = []

        if file_name is not None and data_set is None:

            with open(file_name, "r") as input_file:
                lines = input_file.readlines()
                if input_file_layout == "klam":
                    for n, line in enumerate(lines):
                        columns = line.split()

                        if curve=='CH':
                            lmbd = int(columns[0])
                        elif curve=='CHbox':
                            lmbd = float(columns[0])
                        else:
                            lmbd = float(columns[0])

                        lambdas.append(lmbd)
                        if columns[1].startswith("eeZH_FCCee240"):
                            k_ZH_240[curve].append(np.sqrt(float(columns[2])))
                        elif columns[1].startswith("eeZH_FCCee365"):
                            k_ZH_365[curve].append(np.sqrt(float(columns[2])))

                elif input_file_layout == "CH_CHbox":
                    CH_values = []
                    for n, line in enumerate(lines):
                        columns = line.split()
                        CH = float(columns[0])
                        # CHbox = float(columns[1])
                        CH_values.append(CH)
                        # CHbox_values[curve].append(CHbox)

                        # Fill lambdas list for future compatibility, although currently no predictions for kappa_lambda
                        # are available for this input layout
                        lambdas.append(CH)
                        if columns[-2].startswith("eeZH_FCCee240"):
                            k_ZH_240[curve].append(np.sqrt(float(columns[-1])))
                        elif columns[-2].startswith("eeZH_FCCee365"):
                            k_ZH_365[curve].append(np.sqrt(float(columns[-1])))

                    
                    # Remove duplicated
                    CH_values = CH_values[::2]

            # Remove duplicated
            lambdas = [ lmbd for i, lmbd in enumerate(lambdas) if i%2==0 ]

        elif file_name is None and data_set is not None:
            for bp in data_set:
                lmbd = bp[2]
                lambdas.append(lmbd)
                k_ZH_365[curve].append(bp[0] + 1)
                k_ZH_240[curve].append(bp[1] + 1)
        else:
            error_message = f"Please specify valid curves / files!"
            error_message += f"\nCurve: {curve}, File name: {file_name}, Data set: {data_set}"
            error_message += f"legend_loc: {legend_loc}, legend_col: {legend_col}, legend_fontsize: {legend_fontsize}"
            error_message += f"marker: {marker}, color_line: {color_line}, label_line: {label_line}, markersize: {markersize}"
            print(error_message)
            raise ValueError("Please specify valid curves / files!")

        if input_file_layout == "klam":
            colors = list(color_map(norm(np.array(lambdas))))
        elif input_file_layout == "CH_CHbox":
            colors = list(color_map(np.linspace(0.001, 0.9, len(lambdas))[::-1]))
        if color_independent:
            norm = Normalize(vmin=0, vmax=len(lambdas))
            colors = list(color_map(norm(range(len(lambdas)))))

        ax.plot(np.array(k_ZH_365[curve]) - 1,
                np.array(k_ZH_240[curve]) - 1,
                color=color_line,
        )

        if legend_loc is not None:
            leg_handles.append(Line2D([0], [0], color="k", ls="-", c=color_line, label=label_line))
        else:
            leg_handles.insert(index, Line2D([0], [0], color="k", ls="-", c=color_line, label=label_line))

        if marker is not None:
            for i, (k240, k365, lmbd) in enumerate(zip(k_ZH_240[curve], k_ZH_365[curve], lambdas)):

                if max_n_markers is not None and not i % math.ceil(len(lambdas)/max_n_markers)==0:
                    continue

                ax.plot((k365-1), (k240-1), marker=marker, ls="none", c=colors[i], markersize=markersize, markeredgecolor='white', markeredgewidth=0.5)

                if not (legend_loc is None or klam_in_point_labels is None):
                    if input_file_layout == "CH_CHbox" and data_set is None:
                        raise ValueError("For curves with input layout 'CH_CHbox', markers cannot be shown in the plot legend for curves, since no predictions for kappa_lambda are available for these")
                    if isinstance(lmbd, int):
                        point_label = rf"{klam_in_point_labels} = {lmbd}"
                    else:
                        point_label = rf"{klam_in_point_labels} = {lmbd:.6f}"
                
                    leg_handles.append(Line2D([0], [0], color="k", ls="none", marker=marker, c=colors[i], markersize=markersize, markeredgecolor='white', markeredgewidth=0.5, label=point_label))

        if legend_loc is not None:
            hsLegend = ax.legend(
                handles=leg_handles,
                loc=legend_loc,
                frameon=False,
                prop={'size': legend_fontsize},
                ncol=legend_col,
            )
            ax.add_artist(hsLegend)

            leg_handles = []


    return k_ZH_240, k_ZH_365


import inspect

# fitting a quadratic curve to the (klam, max_deviation) points to have a smooth estimate of the uncertainty as a function of klam
def Quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def plot_kZh_uncertainties_and_minuit_fit(
    bsm_model,
    klam,
    curve,
    label,
    color,
    sqrt_s,
    plot_dir,
    plot_name,
    plot_name_suffix="",
    fit_model=Quadratic,
    save_fig=True,
):
    """
    A function to plot the uncertainty estimates for k_Zh as a function of kappa_lambda, 
    and to fit a curve to these points using the Minuit package. The fitted curve is also 
    plotted together with the points.

    Parameters
    ----------
    bsm_model : str
        The BSM model considered, to be shown in the label and legend
    klam : list of float
        The values of kappa_lambda for the points to be plotted
    curve : list of float
        The values of the uncertainty estimates for k_Zh corresponding to the klam points
    label : str
        The label for the points to be plotted, shown in the legend
    color : str
        The color for the points and the fit curve
    sqrt_s : str
        The center of mass energy considered, to be shown in the label and legend
    plot_dir : str
        The directory where to save the plot
    plot_name : str
        The name for the plot file, without extension and without suffixes
    plot_name_suffix : str, optional
        A suffix to be added to the plot name, after the main name and before the extension. 
        Default is an empty string.
    fit_model : function, optional
        The function to be fitted to the points. The first argument must be kappa_lambda.
        Default is a quadratic function.
    save_fig : bool, optional
        Whether to save the figure as a PDF file. Default is True.

    Returns
    -------
    model_arguments : list of str
        The names of the parameters of the fit model function, excluding the first 
        argument (kappa_lambda).
    fitted_parameters : list of float
        The fitted values for the parameters of the fit model function, in the same order 
        as model_arguments
    """

    fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=300)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel(r"$\kappa_\lambda$", fontsize=13)
    ax.set_ylabel(r"$k_{Zh}$ uncertainty estimate", fontsize=12)

    ax.scatter(klam, curve, color=color, label=label+f" ({bsm_model} BPs)")

    least_squares = LeastSquares(klam, curve, yerror=1e-4, model=fit_model)
    model_arguments = list(inspect.signature(fit_model).parameters)[1:]
    initial_values_parameters = { arg : 0 for arg in model_arguments }

    m = Minuit(least_squares, **initial_values_parameters)
    m.migrad()
    m.hesse()

    fitted_parameters = [ m.values[arg] for arg in model_arguments ]
    fitted_parameters_text = ", ".join(f"{arg}={value}" for arg, value in zip(model_arguments, fitted_parameters))
    print(f"Fitted {fit_model.__name__} parameters: {fitted_parameters_text}")

    x_fit = np.linspace(min(klam), max(klam), 100)
    y_fit = fit_model(x_fit, *fitted_parameters)
    ax.plot(x_fit, y_fit, color=color, label=label+f" ({fit_model.__name__} fit)")

    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8.0, loc='best')
    plt.tight_layout()

    if save_fig: fig.savefig(f'{plot_dir}/{plot_name}{plot_name_suffix}_{sqrt_s}_uncertainties_{fit_model.__name__}_fit.pdf')

    return model_arguments, fitted_parameters
