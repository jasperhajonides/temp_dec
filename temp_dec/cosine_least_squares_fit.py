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
            vector of tuning curve phase ranging from [-pi-x to pi]
            where x is the the stepsize
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
    t = np.linspace(-np.pi+(0.5*2*np.pi/n_steps),np.pi-(0.5*2*np.pi/n_steps),n_steps)
    guess_mean = 1/9
    guess_std = np.std(tuning)#3*np.std(0.0033)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.cos(t+guess_phase) + guess_mean

    # if input is 2d array we loop over the second dimension
    if len(tuning.shape) > 1:
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
    else:
        optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning
        est_amp, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]

        # if max amplitude should not be negative, otherwise we'll shift phase accordingly
        if (est_amp < 0) & (est_phase<0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.pi + est_phase
        elif (est_amp < 0) & (est_phase>0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.pi _ est_phase
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
