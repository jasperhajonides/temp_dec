from scipy.optimize import leastsq
import numpy as np


def cosine_least_squares_fit(tuning):
    """Fit a cosine to tuning curve data to obtain model fit Parameters

    Parameters
    ----------
    tuning : ndarray
            1d values of tuning curve sampled from [-pi-x to pi]
            or
            2d array of tuning curve bins by repeats
    t     : ndarray
          
    Returns
    --------
    dictionary:
        data_fit : ndarray
                estimated cosine fit to data
        amplitude: ndarray
                highest value - mean of the cosine
        phase: ndarray
                phase of cosine. negative value means peak is shifted left,
                postitive value means peak is shifted right
        mean: ndarray
                mean value of cosine
    """
    n_steps = tuning.shape[0]
    
#   vector of tuning curve phase ranging from [-pi-x to pi]
#         where x is the the stepsize
    t = np.linspace(-np.pi+(0.5*2*np.pi/n_steps),np.pi-(0.5*2*np.pi/n_steps),n_steps)
    
    guess_mean = 1/12
    guess_std = np.std(tuning)#3*np.std(0.0033)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.cos(t+guess_phase) + guess_mean

    # if input is 2d array we loop over the second dimension
    if len(tuning.shape) == 2:
        print('Fitting tuning curves, each tp (2d)')

        repeats = tuning.shape[1] 
        #initialise parameter estimates
        est_amp = np.zeros(repeats)
        est_phase = np.zeros(repeats)
        est_mean = np.zeros(repeats)
        data_fit = np.zeros((n_steps,repeats))

        for rep in range(0,tuning.shape[1]):
            optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning[:,rep]
            ea, ep, em = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]
            est_amp[rep] = ea * np.sign(ea) # if max amplitude is negative we also multiply phase shift by -1
            est_phase[rep] = ep * np.sign(ea)
            est_mean[rep] = em

            # recreate the fitted curve using the optimized parameters
            data_fit[:,rep] = ea*np.cos(t+ep) + em
    elif len(tuning.shape)==3:
        print('Fitting tuning curves, each tp, each distractor-target bin (3d)')
        repeats = tuning.shape[1] 
        distr_bins = tuning.shape[2]
        #initialise parameter estimates
        est_amp = np.zeros((repeats,distr_bins))
        est_phase = np.zeros((repeats,distr_bins))
        est_mean = np.zeros((repeats,distr_bins))
        data_fit = np.zeros((n_steps,repeats,distr_bins))

        for rep in range(0,repeats):
            for b in range(0,distr_bins):
                optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning[:,rep,b]
                ea, ep, em = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]
##                
#                if (ea < 0) & (ep<0):
#                    ea = ea * np.sign(ea)
#                    ep =  ep + np.pi
#                elif (ea > 0) & (ep>0):
#                    ea = ea * np.sign(ea)
#                    ep = ep - np.pi
#                else: #phew all normal
#                    ea = ea* np.sign(ea)
#                    ep = ep* np.sign(ea)
#                
                
                est_amp[rep,b] = ea * np.sign(ea) # if max amplitude is negative we also multiply phase shift by -1
                est_phase[rep,b] = ep * np.sign(ea)
                est_mean[rep,b] = em

                # recreate the fitted curve using the optimized parameters
                data_fit[:,rep,b] = ea*np.cos(t+ep) + em
            
    else:
        print('Fitting tuning curves, 1d')

        optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning
        est_amp, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]

        # if max amplitude should not be negative, otherwise we'll shift phase accordingly
        if (est_amp < 0) & (est_phase<0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.pi + est_phase
        elif (est_amp < 0) & (est_phase>0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.pi - est_phase
        else: #phew all normal
            est_amp = est_amp* np.sign(est_amp)
            est_phase = est_phase* np.sign(est_amp)

        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp*np.cos(est_freq*t+est_phase) + est_mean


    output = {
    "data_fit": data_fit,
    "amplitude": est_amp,
    "phase": -est_phase,
    "mean" : est_mean
    }

    return output


def TC_least_squares_fit(tuning):
    """Fit a cosine to tuning curve data to obtain model fit Parameters

    Parameters
    ----------
    tuning : ndarray
            1d values of tuning curve sampled from [-pi-x to pi]
            or
            2d array of tuning curve bins by repeats
    t     : ndarray
          
    Returns
    --------
    dictionary:
        data_fit : ndarray
                estimated cosine fit to data
        amplitude: ndarray
                highest value - mean of the cosine
        phase: ndarray
                phase of cosine. negative value means peak is shifted left,
                postitive value means peak is shifted right
        mean: ndarray
                mean value of cosine
    """
    
    with open ('/Users/jasperhajonides/Documents/EXP8_UpdProtec/MEG/analyses/tuning_curve', 'rb') as fp:
            [TC] = pickle.load(fp)
            
            
    n_steps = tuning.shape[0]
    
    
    guess_std = np.std(tuning)#3*np.std(0.0033)/(2**0.5)/(2**0.5)
    guess_amp = 1



    # if input is 2d array we loop over the second dimension
    if len(tuning.shape) == 2:
        # print('Fitting tuning curves, each tp (2d)')

        repeats = tuning.shape[1] 
        #initialise parameter estimates
        est_amp = np.zeros(repeats)
        data_fit = np.zeros((n_steps,repeats))

        for rep in range(0,tuning.shape[1]):
            optimize_func = lambda x: x[0]*(TC-1/n_steps) - tuning[:,rep]
            ea = leastsq(optimize_func, [guess_amp])[0]
            est_amp[rep] = ea 

            # recreate the fitted curve using the optimized parameters
            data_fit[:,rep] = ea*(TC-1/n_steps)
    elif len(tuning.shape)==3:
        # print('Fitting tuning curves, each tp, each distractor-target bin (3d)')
        repeats = tuning.shape[1] 
        distr_bins = tuning.shape[2]
        #initialise parameter estimates
        est_amp = np.zeros((repeats,distr_bins))
        data_fit = np.zeros((n_steps,repeats,distr_bins))

        for rep in range(0,repeats):
            for b in range(0,distr_bins):
                optimize_func = lambda x: x[0]*(TC-1/n_steps)  - tuning[:,rep,b]
                ea = leastsq(optimize_func, [guess_amp])[0]

                est_amp[rep,b] = ea 

                # recreate the fitted curve using the optimized parameters
                data_fit[:,rep,b] = ea*(TC-1/n_steps)
            
    else:
        # print('Fitting tuning curves, 1d')

        optimize_func = lambda x: x[0]*(TC-1/n_steps)- tuning
        est_amp = leastsq(optimize_func, [guess_amp])[0]

        # if max amplitude should not be negative, otherwise we'll shift phase accordingly
        
        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp*(TC-1/n_steps)


    output = {
    "data_fit": data_fit,
    "amplitude": est_amp,
    }

    return output

