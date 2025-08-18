import pandas as pd
import numpy as np
import os
import subprocess
from pyCollier import db0

# Johannes' formulas
Mh = 125.1
vev = 246.21965
@np.vectorize
def ZZh_hextleg(kala):
    dZh = -(Mh**2*(-9 + 2*np.sqrt(3)*np.pi))/(32*np.pi**2*vev**2)
    return (kala**2-1)*dZh


# SMEFT / HEPfit expressions
M_PI = 3.14159265358979323846
GF = 1.1663787e-5
mHl = 125.1
sqrt = np.sqrt

@np.vectorize
def smeft_sigma_Zh(lmbd, sqrt_s):
    mu = 1

    if sqrt_s == 240:
        C1 = 0.017
    elif sqrt_s == 365:
        C1 = 0.0057
    elif sqrt_s == 500:
        C1 = 0.00099
    else:
        raise ValueError("sqrt_s must be 240, 365, or 500 GeV")

    # Expression for the Higgs self-energy diagram
    dZH = -(9.0/16.0)*( GF*mHl*mHl/sqrt(2.0)/M_PI/M_PI )*( 2.0*M_PI/3.0/sqrt(3.0) - 1.0 )
    
    # Resummations
    dZH1 = dZH / (1.0 - dZH)
    dZH2 = dZH * (1 + 3.0 * dZH) / (1.0 - dZH) / (1.0 - dZH)

    # HEPfit flags
    cLHd6 = 1
    cLH3d62 = 1

    deltaG_hhhRatio = lmbd - 1

    mu = mu + cLHd6*(C1 + 2.0*dZH1)*deltaG_hhhRatio

    mu = mu + cLHd6*cLH3d62*dZH2*deltaG_hhhRatio*deltaG_hhhRatio

    return mu


@np.vectorize
def smeft_sigma_Zh_general_C1(lmbd, sqrt_s, C1_values):
    mu = 1

    if sqrt_s not in C1_values:
        raise ValueError("sqrt_s must be 240, 365, or 500 GeV")
    else:
        C1 = C1_values[sqrt_s]

    # Expression for the Higgs self-energy diagram
    dZH = -(9.0/16.0)*( GF*mHl*mHl/sqrt(2.0)/M_PI/M_PI )*( 2.0*M_PI/3.0/sqrt(3.0) - 1.0 )
    
    # Resummations
    dZH1 = dZH / (1.0 - dZH)
    dZH2 = dZH * (1 + 3.0 * dZH) / (1.0 - dZH) / (1.0 - dZH)

    # HEPfit flags
    cLHd6 = 1
    cLH3d62 = 1

    deltaG_hhhRatio = lmbd - 1

    mu = mu + cLHd6*(C1 + 2.0*dZH1)*deltaG_hhhRatio

    mu = mu + cLHd6*cLH3d62*dZH2*deltaG_hhhRatio*deltaG_hhhRatio

    return mu

# C1 values in the HEPfit formula. See smeft_sigma_Zh
C1_hepfit = {
    240: 0.017, 
    365: 0.0057, 
    500: 0.00099
}

# Function to evaluate the vertex correction to sigma_Zh proportional to (klam-1)
@np.vectorize
def effZZh1Lkala_HEPfit_C1_values(lmdb, sqrt_s, C1_values=C1_hepfit):
    if sqrt_s not in C1_values:
        raise ValueError("sqrt_s must be 240, 365, or 500 GeV")
    else:
        C1 = C1_values[sqrt_s]
    return C1 * (lmdb - 1)



def analytical_formula_CH_CHBox_general_C1(mu_240, mu_365, C1_values):
    """
    Function which inverts the sigma_Zh expressions in HEPfit in order to obtain corresponding
    values for the SMEFT Wilson coefficients (CH, CHbox), as well as kappa_lambda, given values
    for the ratios (mu_240, mu_365), the ratios of the Zh cross-sections over the SM prediction.

    Parameters
    ----------
    mu_240 : float
        The ratio of the e+e- -> Zh cross-section at 240 GeV over the SM prediction.
    mu_365 : float
        The ratio of the e+e- -> Zh cross-section at 365 GeV over the SM prediction.
    C1_values : dict
        The values of the C1 coefficient at different energy scales. Should be a dictionary
        with keys 240 and 365.

    Returns
    -------
    CH : float
        The value of the SMEFT Wilson coefficient CH.
    CHBox : float
        The value of the SMEFT Wilson coefficient CHBox.
    lmbd : float
        The value of the Higgs self-coupling kappa_lambda.
    """

    dmu = mu_365 - mu_240

    LambdaNP2 = 1000.**2

    C1_240 = C1_values[240]
    C1_365 = C1_values[365]
    dC = C1_365 - C1_240

    D1_240 = +121263./LambdaNP2
    D1_365 = +121243./LambdaNP2
    dD = D1_365 - D1_240

    M_PI = 3.14159265358979323846
    GF = 1.1663787e-5
    mHl = 125.1

    dZH = -(9.0/16.0)*( GF*mHl*mHl/sqrt(2.0)/M_PI/M_PI )*( 2.0*M_PI/3.0/sqrt(3.0) - 1.0 )

    dZH1 = dZH / (1.0 - dZH)
    dZH2 = dZH * (1 + 3.0 * dZH) / (1.0 - dZH) / (1.0 - dZH)

    # dZH1 = dZH
    # dZH2 = dZH

    a = dZH2
    b = C1_365 + 2*dZH1 - D1_365*dC/dD 
    c = 1 + D1_365*dmu/dD - mu_365

    dlmbd = (-b-sqrt(b**2 - 4*a*c))/(2*a)
    lmbd = 1 + dlmbd

    CHBox = 1./dD*(dmu - dC*dlmbd)
    CH = -2.1290888208276963*(dlmbd - CHBox/5.498361921343667)

    return CH, CHBox, lmbd


def analytical_formula_CH_CHBox(mu_240, mu_365):
    """
    Function which inverts the sigma_Zh expressions in HEPfit in order to obtain corresponding
    values for the SMEFT Wilson coefficients (CH, CHbox), as well as kappa_lambda, given values
    for the ratios (mu_240, mu_365), the ratios of the Zh cross-sections over the SM prediction.

    Parameters
    ----------
    mu_240 : float
        The ratio of the e+e- -> Zh cross-section at 240 GeV over the SM prediction.
    mu_365 : float
        The ratio of the e+e- -> Zh cross-section at 365 GeV over the SM prediction.

    Returns
    -------
    CH : float
        The value of the SMEFT Wilson coefficient CH.
    CHBox : float
        The value of the SMEFT Wilson coefficient CHBox.
    lmbd : float
        The value of the Higgs self-coupling kappa_lambda.
    """

    return analytical_formula_CH_CHBox_general_C1(mu_240, mu_365, C1_values=C1_hepfit)



# Function to print messages to output and to a file
def print_to_file(message, file):
    print(message)
    print(message, file=file)


def find_benchmark(n_pts,
                   kappas,
                   EWPOs,
                   model_pars,
                   old_df_indices,
                   BP_Names,
                   BP_output_file,
                   max_errors_365,
                   max_errors_240,
                   delta_kappas_z_365,
                   delta_kappas_z_240,
                   BR_constraints=None,
                   ):
    """
    Find benchmark points for a given model. Lists of proposed (k_Zh_365 - 1) and 
    (k_Zh_240 - 1) values are given as input, and the function tries to find actual
    parameter points around these values. The maximum deviation from the proposed values
    should be given in max_errors_365 and max_errors_240.

    Parameters
    ----------
    n_pts : int
        Total number of parameter points considered 
    kappas : dict
        Dictionary with the input data for the coupling modifiers. Each key corresponds 
        to a different coupling (e.g., ZZ_240), and each value is a numpy array or a 
        pd.DataFrame column containting the actual values for each parameter point
    EWPOs : dict
        Dictionary containing the electroweak precision observables.
    model_pars : dict
        Dictionary containing the model parameters.
    old_df_indices : np.ndarray
        Array containing the old indices of the DataFrame.
    BP_Names : list of str
        List of benchmark point names.
    BP_output_file : file
        Output text file where the benchmark point results are written. The results 
        are written as a series of if/else statements in python, which can be directly 
        copied/pasted into the script which sets up the global fits
    max_errors_365 : list of float
        List of maximum allowed deviations from (k_Zh_365 - 1) which the benchmark points 
        must have from the proposed values in delta_kappas_z_365
    max_errors_240 : list of float
        List of maximum allowed deviations from (k_Zh_240 - 1) which the benchmark points 
        must have from the proposed values in delta_kappas_z_240
    delta_kappas_z_365 : list of float
        List of (k_Zh_365 - 1) values around which to find benchmark points
    delta_kappas_z_240 : list of float
        List of (k_Zh_240 - 1) values around which to find benchmark points
    BR_constraints : float, optional
        Branching ratio constraints for BPs. Model parameter points for which the SM Higgs 
        couplings to other SM particles deviate by over BR_constraints*100% are discarded

    Returns
    -------
    bp_kappas : list of dict
        List of results for the coupling modifiers. Entries correspond to benchmark points and are dictionaries 
        with coupling names as keys and their values.
    bp_EWPOs : list of dict
        Results for the Electroweak Precision Observables for the benchmark points.
    bp_model_pars : list of dict
        Results for the model parameters for the benchmark points.
    bp_indices : list of int
        List of indices () where benchmark points were found.
    """

    if not BR_constraints is None:
        satisfy_BR_constraint = [True for i in range(n_pts)]
        for i in range(n_pts):
            for coup in ['uu', 'dd', 'cc', 'ss', 'tt', 'bb', 'ee', 'mumu', 'tautau', 'WW', 'ZZ', 'Zgam', 'gamgam']:
                if np.abs(kappas[coup][i] - 1.) > BR_constraints:
                    satisfy_BR_constraint[i] = False

        for coup in kappas.keys():
            kappas[coup] = np.array(kappas[coup][satisfy_BR_constraint])
        
        for ewpo in EWPOs.keys():
            EWPOs[ewpo] = np.array(EWPOs[ewpo][satisfy_BR_constraint])

        for par in model_pars.keys():
            model_pars[par] = np.array(model_pars[par][satisfy_BR_constraint])

        old_df_indices = old_df_indices[satisfy_BR_constraint]


    if not len(delta_kappas_z_365) == len(delta_kappas_z_240):
        raise ValueError("delta_kappas_z_365 and delta_kappas_z_240 have different lengths!")

    bp_indices = [None for i in range(len(delta_kappas_z_365))]
    bp_kappas = [None for i in range(len(delta_kappas_z_365))]
    bp_EWPOs = [None for i in range(len(delta_kappas_z_365))]
    bp_model_pars = [None for i in range(len(delta_kappas_z_365))]

    for ind, (kZ_365, kZ_240) in enumerate(zip(kappas['ZZ_365'], kappas['ZZ_240'])):
        for BP, (delta_kappa_z_365, delta_kappa_z_240, max_error_365, max_error_240) in enumerate(zip(delta_kappas_z_365, delta_kappas_z_240, max_errors_365, max_errors_240)):
            if abs((kZ_365-1) - delta_kappa_z_365) < max_error_365 and abs((kZ_240-1) - delta_kappa_z_240) < max_error_240:
                bp_indices[BP] = ind


    if any(bp_index is None for bp_index in bp_indices):
        missing_BPs = [i for i, bp_index in enumerate(bp_indices) if bp_index is None]
        raise ValueError(f"The following BPs were not found: {missing_BPs}")


    for BP, bp_index in enumerate(bp_indices):

        print_to_file(f"\nelif BP == \"{BP_Names[BP]}\":", file=BP_output_file)

        # bfp_chisq = chisq[bp_index]
        bp_kappas[BP] = {coup:kps[bp_index] for (coup, kps) in kappas.items()}
        bp_EWPOs[BP] = {obs_name:obs_value[bp_index] for (obs_name, obs_value) in EWPOs.items()}
        bp_model_pars[BP] = {par_name:par_value[bp_index] for (par_name, par_value) in model_pars.items()}

        for coup, kaps in bp_kappas[BP].items():
            print_to_file(f"    kappas['{coup}'] = {kaps}", file=BP_output_file)

        print_to_file(f"    # abs(kappas['ZZ_365'] - kappas['ZZ_240'])/(kappas['ZZ_240'] - 1) = {np.abs((bp_kappas[BP]['ZZ_365'] - bp_kappas[BP]['ZZ_240'])/(bp_kappas[BP]['ZZ_240'] - 1))}", file=BP_output_file)
        print_to_file(f"    # abs(kappas['ZZ_365'] - kappas['ZZ_240']) = {np.abs(bp_kappas[BP]['ZZ_365'] - bp_kappas[BP]['ZZ_240'])}", file=BP_output_file)

        for obs_name, obs_value in bp_EWPOs[BP].items():
            print_to_file(f"    {obs_name} = {obs_value}", file=BP_output_file)

        for par_name, par_value in bp_model_pars[BP].items():
            print_to_file(f"    # {par_name} = {par_value}", file=BP_output_file)

        print_to_file(f"    # Best scan point row: {old_df_indices[bp_index]+2} out of {old_df_indices[-1]+2}", file=BP_output_file)
        
    return bp_kappas, bp_EWPOs, bp_model_pars, bp_indices


### FINISH DOCUMENTATION
def find_benchmark_lambda1(n_pts,
                           kappas,
                           EWPOs,
                           model_pars,
                           old_df_indices,
                           BP_Name,
                           BR_constraints=None,
                           ):
    """
    Function to find the benchmark point that, among the data set, has kappa_lambda
    closest to the Standard Model prediction (kappa_lambda = 1).

    Parameters
    ----------
    n_pts : int
        Total number of parameter points considered 
    kappas : dict
        Dictionary with the input data for the coupling modifiers. Each key corresponds 
        to a different coupling (e.g., ZZ_240), and each value is a numpy array or a 
        pd.DataFrame column containting the actual values for each parameter point
    EWPOs : dict
        Dictionary containing the electroweak precision observables.
    model_pars : dict
        Dictionary containing the model parameters.
    old_df_indices : np.ndarray
        Array containing the old indices of the DataFrame.
    BP_Names : list of str
        Names of the benchmark points to be found.

    BR_constraints : float, optional
        Branching ratio constraints for BPs. Model parameter points for which the SM Higgs 
        couplings to other SM particles deviate by over BR_constraints*100% are discarded

    Returns
    -------
    bp_kappas : dict
        Dictionary containing the results the coupling modifiers, for the benchmark point found. 
        The dictionary keys are the coupling names, and predictions are stored in the dictionary
        values.
    bp_EWPOs : dict
        Results for the Electroweak Precision Observables for the benchmark point.
    bp_model_pars : dict
        Results for the model parameters for the benchmark point.
    bp_index : int
        List of indices () where benchmark point was found.
    """

    if not BR_constraints is None:
        satisfy_BR_constraint = [True for i in range(n_pts)]
        for i in range(n_pts):
            for coup in ['uu', 'dd', 'cc', 'ss', 'tt', 'bb', 'ee', 'mumu', 'tautau', 'WW', 'ZZ', 'Zgam', 'gamgam']:
                if np.abs(kappas[coup][i] - 1.) > BR_constraints:
                    satisfy_BR_constraint[i] = False

        for coup in kappas.keys():
            kappas[coup] = np.array(kappas[coup][satisfy_BR_constraint])
        
        for ewpo in EWPOs.keys():
            EWPOs[ewpo] = np.array(EWPOs[ewpo][satisfy_BR_constraint])

        for par in model_pars.keys():
            model_pars[par] = np.array(model_pars[par][satisfy_BR_constraint])

        old_df_indices = old_df_indices[satisfy_BR_constraint]

    bp_index = None
    bp_kappas = None
    bp_EWPOs = None
    bp_model_pars = None

    # for ind, lmbd in enumerate(kappas['lam']):
    #     if abs(lmbd-1) < max_delta_lambda:
    #         bp_index = ind

    bp_index = np.argmin(kappas['lam'])

    if bp_index is None:
        raise ValueError(f"Could not find such BP!")

    print(f"\nelif BP == \"{BP_Name}\":")

    bp_kappas = {coup:kps[bp_index] for (coup, kps) in kappas.items()}
    bp_EWPOs = {obs_name:obs_value[bp_index] for (obs_name, obs_value) in EWPOs.items()}
    bp_model_pars = {par_name:par_value[bp_index] for (par_name, par_value) in model_pars.items()}

    for coup, kaps in bp_kappas.items():
        print(f"    kappas['{coup}'] = {kaps}")

    for obs_name, obs_value in bp_EWPOs.items():
        print(f"    {obs_name} = {obs_value}")

    for par_name, par_value in bp_model_pars.items():
        print(f"    # {par_name} = {par_value}")

    print(f"    # Best scan point row: {old_df_indices[bp_index]+2} out of {old_df_indices[-1]+2}")

    return bp_kappas, bp_EWPOs, bp_model_pars, bp_index


