#!/usr/bin/env python
"""
Helper functions for computations for the BIO-482 Miniproject.
"""

# Imports
import numpy as np
from scipy.signal import find_peaks, stft


def Function_Detect_APs(MembranePotential, SR_Vm, Vm_Deriv_Thrs):
    # Parameters
    AP_Param_Output = []
    AP_Win = 0.0015  # time (s) to search for AP's peak
    Min_AP_Amp = 0.005  # Minimum AP amplitude (V)
    AP_length = np.round(AP_Win * SR_Vm)
    Vm_Deriv = np.diff(MembranePotential) * SR_Vm
    Vm_Deriv_Corr=Vm_Deriv - Vm_Deriv_Thrs
    Vm_Deriv_Corr[Vm_Deriv_Corr==0]=0.000001
    AP_Thrs_Onset = np.diff(np.divide(Vm_Deriv_Corr, np.abs(Vm_Deriv_Corr)))
    # AP_Thrs_Onset = np.diff(np.divide(Vm_Deriv - Vm_Deriv_Thrs, np.abs(Vm_Deriv - Vm_Deriv_Thrs)))
    Vm_Med = np.median(MembranePotential)
    AP_Thrs_Index, Peaks = find_peaks(AP_Thrs_Onset, height=0.1, prominence=0.5, distance=SR_Vm * 0.001)

    AP_Param_Output = np.empty([len(AP_Thrs_Index), 6])
    AP_Param_Output[:] = np.nan

    if (AP_Thrs_Index.any()):
        AP_cnt = 0
        for i in range(len(AP_Thrs_Index)):
            pt1 = int(AP_Thrs_Index[i])
            pt2 = int(AP_Thrs_Index[i] + AP_length)

            if (pt2 < len(MembranePotential) - 1):  # remove -1?
                AP_Seg = MembranePotential[pt1:pt2]
                Ind = np.argmax(AP_Seg)
                if (Ind.any()):
                    AP_Index = pt1 + Ind - 1  # remove -1?
                    AP_Amp = MembranePotential[AP_Index] - MembranePotential[pt1]
                    if ((AP_Amp > Min_AP_Amp) & (MembranePotential[AP_Index] > Vm_Med)):
                        AP_Param_Output[AP_cnt, 0] = AP_Thrs_Index[i] / SR_Vm  # Thrs Time
                        AP_Param_Output[AP_cnt, 1] = MembranePotential[AP_Thrs_Index[i]]  # Thrs Vm
                        AP_Param_Output[AP_cnt, 2] = AP_Index / SR_Vm  # Peak Time
                        AP_Param_Output[AP_cnt, 3] = MembranePotential[AP_Index]  # Peak Vm
                        AP_Param_Output[AP_cnt, 4] = AP_Amp  # Peak Amp
                        Ind = AP_Thrs_Index[i]
                        pt2 = AP_Index
                        # we define a time window for the AP ...
                        pt1 = int(AP_Thrs_Index[i])  # ... AP threshold index ...
                        pt3 = int(pt2 + (0.005 * SR_Vm))  # ... and 3 ms after the peak.
                        if (pt1 > 0 & pt3 < len(MembranePotential)):
                            Vm_HalfAmp = MembranePotential[Ind] + AP_Amp / 2  # Vm at half amplitude
                            sAP_Seg = MembranePotential[pt1:pt3]  # cut a segment of the VM that contains the AP
                            sAP_Seg = sAP_Seg - Vm_HalfAmp  # substract the Vm at half-amplitude
                            sAP_OnOff = np.diff(np.divide(sAP_Seg, np.abs(sAP_Seg)))  # compute the binary signal
                            sAP_Indmax = np.argmax(sAP_OnOff)  # identify index begening AP at half amplitude
                            sAP_Indmin = np.argmin(sAP_OnOff)  # identify index end AP at half amplitude
                            
                            if np.min(sAP_OnOff)>-1:
                                sAP_Indmin=len(sAP_Seg)

                            AP_Param_Output[AP_cnt, 5] = ((sAP_Indmin - sAP_Indmax) / SR_Vm) * 1000  # compute duration at half-amplitude
                        else:
                            AP_Param_Output[AP_cnt, 5] = np.nan
                        AP_cnt = AP_cnt + 1

    if (AP_Param_Output.any()):
        Amp_Lim_Inf = min([np.median(AP_Param_Output[:, 4]) - (5 * np.std(AP_Param_Output[:, 4])), 0.03])
        Peak_Lim_Inf = np.min([np.median(AP_Param_Output[:, 3]) - 5 * np.std(AP_Param_Output[:, 3]), -0.02])
        cnt_Max = len(AP_Param_Output)
        for i in range(cnt_Max):
            cnt = cnt_Max - i - 1
            if ((AP_Param_Output[cnt, 4] < Amp_Lim_Inf) & (AP_Param_Output[cnt, 3] < Peak_Lim_Inf)):
                # AP_Param[cnt,:]=np.empty([1, 6])
                AP_Param_Output = np.delete(AP_Param_Output, cnt, 0)

    AP_Param_Output = AP_Param_Output[~np.isnan(AP_Param_Output).all(axis=1), :]

    return AP_Param_Output

def extract_continuous_trace(data, Mouse, Cell_Counter, Trial_Counter, key='Trial_MembranePotential'):
    """Extracts membrane potential from data structure

    Parameters
    ----------
    data : dict
        Dictionary with the imported datastructure
    Mouse : str
        Name of the mouse we want to process
    Cell_Counter : int
        Count of neurons per mouse
    Trial_Counter : int
        Trial number that we want to process

    Returns
    -------
    selected_trial_vm : np.ndarray
        membrane potential for selected mouse, cell, trial
    """
    # Find correct trial with mouse, cell, and trial constraint.
    # This line is a little bit weird because of the types and how the vectors
    #     are represented.
    selected_trial = (Mouse == data['Mouse_Name']) & (data['Cell_Counter'] == Cell_Counter) & (data['Trial_Counter'] == Trial_Counter)
    # extract the membrane potential
    continuous_trace = data[key][np.where(selected_trial)[0][0]]

    return continuous_trace

def extract_continuous_trace_time(data, Mouse, Cell_Counter, Trial_Counter, key='Trial_MembranePotential'):
    """Extracts membrane potential from data structure

    Parameters
    ----------
    data : dict
        Dictionary with the imported datastructure
    Mouse : str
        Name of the mouse we want to process
    Cell_Counter : int
        Count of neurons per mouse
    Trial_Counter : int
        Trial number that we want to process

    Returns
    -------
    selected_trial_vm : np.ndarray
        membrane potential for selected mouse, cell, trial
    """
    # Find correct trial with mouse, cell, and trial constraint.
    # This line is a little bit weird because of the types and how the vectors
    #     are represented.
    selected_trial = (Mouse == data['Mouse_Name']) & (data['Cell_Counter'] == Cell_Counter) & (data['Trial_Counter'] == Trial_Counter)
    # extract the membrane potential
    times = data[key][np.where(selected_trial)[0]][0]
    start_time = 0
    sr = data['Trial_WhiskerAngle_C2_right_SamplingRate'][selected_trial][0]
    end_time = data['Trial_WhiskerAngle_C2_right'][selected_trial][0].shape[0] / sr
    samples = int((end_time - start_time) * sr)
    continuous_trace = np.zeros(samples)

    if type(times) == np.ndarray:
        for time in times:
            continuous_trace[int(time[0]*sr):int(time[1]*sr)] = 1

    return continuous_trace


def Function_AP_Features(membrane_potential, sr_vm, ap_peak_index, ap_peak_vm, vm_deriv_thrs):
    """
    In this function we find the action potential threshold and duration.
    We define the threshold as the point where the derivative is the maximum before the AP
    and duration as the time difference from half max amplitude to half max amplitude.
    """
    vm_deriv = np.diff(membrane_potential) * 0.001 * sr_vm  # we calculate the derivative
    # we find the points that have deriveative
    ap_thr_onset = np.diff((vm_deriv - vm_deriv_thrs) / np.abs(vm_deriv - vm_deriv_thrs))

    # initialize arrays
    ap_thrs_ind = np.empty_like(ap_peak_vm)
    ap_thrs_vm = np.empty_like(ap_peak_vm)
    ap_duration = np.empty_like(ap_peak_vm)

    # In order to find the duration of each AP more efficiently we don't search the whole time around the peak but only
    # time_search around the peak
    time_search = int(0.002 * sr_vm)
    for i, ap_ind in enumerate(ap_peak_index):
        peak_ind = ap_ind
        before_peak_ind = peak_ind - time_search
        after_peak_ind = peak_ind + time_search
        # we need a sanity check first that we don't go over the threshold of the array
        if before_peak_ind > 0 and after_peak_ind < membrane_potential.shape[0]:
            ind = np.argmax(ap_thr_onset[before_peak_ind:after_peak_ind])
            ap_thrs_ind[i] = before_peak_ind + ind
            ap_thrs_vm[i] = membrane_potential[int(ap_thrs_ind[i])]

            ap_amp = membrane_potential[ap_ind] - ap_thrs_vm[i]
            vm_half_amp = ap_thrs_vm[i] + ap_amp / 2

            sAP_Seg = membrane_potential[before_peak_ind: after_peak_ind] - vm_half_amp
            sAP_OnOff = np.diff(sAP_Seg / np.abs(sAP_Seg))
            sAP_Indmax = np.argmax(sAP_OnOff)
            sAP_Indmin = np.argmin(sAP_OnOff)
            ap_duration[i] = ((sAP_Indmin - sAP_Indmax) / sr_vm) * 1000  # return time in ms
        else:
            ap_thrs_ind[i] = np.nan
            ap_thrs_vm[i] = np.nan
            ap_duration[i] = np.nan

    return ap_thrs_ind, ap_thrs_vm, ap_duration


def Function_Compute_FFTs(membrane_potential, sr_vm, TimeWindow):
    """ Compute the short time fourrier transform of the membrane potential, i.e.
    every TimeWindow * sr_vm (step) samples we calculate the fft of the signal. At
    the same time we detrend the singal (remove its mean) and multiply it with a
    hanning window (it is in the default of stft). About the detrending we don't want
    the signal lay ~-60mV giving us a strong bias term which probably is not usuful.
    About the window, we use a window function because if we don't we may get some
    high frequency artifacts.
    """
    step = int(TimeWindow*sr_vm)
    nfft = int(2**np.ceil(np.log(step)/np.log(2)))
    FFT =  np.abs(stft(membrane_potential, sr_vm, nperseg=step, noverlap=0, detrend='constant', nfft=nfft, padded=False, boundary=None)[2])
    FFT[0,:] /= 2
    return FFT


def Function_CutAPs(Vm, SR_Vm, AP_Peak_Times, AP_Thrs_Times, AP_End_Wind=0.015):
    from scipy.signal import savgol_filter

    AP_Thrs_Index = np.round(AP_Thrs_Times * SR_Vm)
    AP_Peak_Index = np.round(AP_Peak_Times * SR_Vm)
    Vm_Sub = Vm.copy()
    for Ind in range(len(AP_Peak_Index)):

        pt1 = int(AP_Thrs_Index[Ind])
        pt2 = int(AP_Peak_Index[Ind])

        pt4 = int(min(len(Vm) - 1, pt2 + np.round(AP_End_Wind * SR_Vm)))
        Vm_Thrs = Vm[pt1]
        AP_seg = savgol_filter(Vm[pt1:pt4], 5, 3)

        cond = 0
        n2 = pt2 - pt1 + 20
        while (cond == 0):
            n2 += 1
            if (n2 > len(AP_seg)):
                pt_end = pt1 + n2
                cond = 1
            else:
                D_Vm1 = AP_seg[n2 - 2] - Vm_Thrs
                D_Vm2 = AP_seg[n2 - 1] - Vm_Thrs
                if D_Vm2 < 0:
                    pt_end = pt1 + n2
                    cond = 1
                if (D_Vm2 > D_Vm1):
                    pt_end = pt1 + n2
                    cond = 1

        pt3 = min(len(Vm) - 1, pt_end)
        Delta_Vm = Vm[pt3] - Vm[pt1]
        In = range(pt3 - pt1)
        In = np.divide(In, pt3 - pt1) * Delta_Vm
        In = In + Vm[pt1]
        Vm_Sub[pt1:pt3] = In

    return Vm_Sub

def Function_SubThrsVm(membrane_potential, sampling_rate, TimeWindow):
    Numb_Wind = np.floor(membrane_potential.shape[0] / sampling_rate / TimeWindow)
    mean_vm = []
    std_vm = []
    for window in range(int(Numb_Wind)):
        pt1 = int(TimeWindow * sampling_rate * window)
        pt2 = int(TimeWindow * sampling_rate * (window + 1))
        mean_vm += [np.mean(membrane_potential[pt1:pt2])]
        std_vm += [np.std(membrane_potential[pt1:pt2])]
    return np.array(mean_vm), np.array(std_vm)


def Function_Event_Triggered_Avg(membrane_potential, sampling_rate, event_times,
                                 pre_window, post_window, min_event_dur, min_iti):
    if type(event_times[0][0]) == np.ndarray:
        event_times = event_times[0]

    time_length = int(np.floor((pre_window + post_window) * sampling_rate))
    vm_out = []
    for i, event in enumerate(event_times):

        event_dur = event[1] - event[0]
        if i == 0:
            iti = event[0]
        else:
            iti = event_times[i][0] - event_times[i - 1][1]

        if event_dur > min_event_dur and iti > min_iti:
            pt1 = int(np.floor((event[0] - pre_window) * sampling_rate)) + 2
            pt2 = pt1 + time_length + 2
            if pt1 > 0 and pt2 < len(membrane_potential) - 1:
                vm_out += [membrane_potential[pt1:pt2]]

    return np.vstack(vm_out).T


def Function_Event_Triggered_AP_Numb(ap_index, sampling_rate, event_times, pre_window,
                                     post_window, min_event_dur, min_iti):
    if type(event_times[0][0]) == np.ndarray:
        event_times = event_times[0]

    pre_ind = -int(np.floor(pre_window * sampling_rate))
    post_ind = int(np.floor(post_window * sampling_rate))

    aps_out = []
    for i, event in enumerate(event_times):

        event_dur = event[1] - event[0]
        if i == 0:
            iti = event[0]
        else:
            iti = event_times[i][0] - event_times[i - 1][1]

        if event_dur > min_event_dur and iti > min_iti:
            pt0 = int(np.floor((event[0]) * sampling_rate)) - 1
            pt1 = int(np.floor((event[0] - pre_window) * sampling_rate)) - 1
            if pt1 > 0:
                ap_index_t0 = ap_index - pt0
                select = ap_index_t0 > pre_ind
                select = np.logical_and(select, ap_index_t0 < post_ind)
                aps_out += [ap_index_t0[select]]

    return aps_out


def Function_Times2Vect(AP_times, SR_Vm, Length_Vect):
    AP_vect = np.zeros([Length_Vect, 1])
    if ~np.isnan(AP_times).any():
        for t in range(len(AP_times)):
            AP_pt = np.round(AP_times[t] * SR_Vm) - 1
            AP_vect[int(AP_pt), 0] = 1
    return AP_vect


def Function_Event_Triggered_Signal(Signal, sampling_rate, event_times,
                                    pre_window, post_window, min_event_dur, min_iti):
    if len(event_times.shape) == 1:
        event_times = event_times[None]
    if type(event_times[0][0]) == np.ndarray:
        event_times = event_times[0]

    time_length = int(np.floor((pre_window + post_window) * sampling_rate))
    Sign_out = []
    for i, event in enumerate(event_times):

        event_dur = event[1] - event[0]
        if i == 0:
            iti = event[0]
        else:
            iti = event_times[i][0] - event_times[i - 1][1]

        if event_dur > min_event_dur and iti > min_iti:
            pt1 = int(np.floor((event[0] - pre_window) * sampling_rate))
            pt2 = pt1 + time_length
            if pt1 > 0 and pt2 < len(Signal) - 1:
                Sign_out += [Signal[pt1:pt2]]
    if Sign_out == []:
        return -1
    return np.vstack(Sign_out).T


def Function_PSTH(AP_avg, SR_Vm, Pre_Window, Post_Window, bin_size):
  
    """     This function generates a peri-stimulus time histogram (PSTH) based on the averaged AP signal around event time 

     INPUTS:
     AP_avg = vector of the averaged AP signal around event onset time
     SR_Vm = Sampling rate of the membrane potential (sample / s)
     Pre_Window = time before event onset time (s)
     Post_Window = time after event onset time (s)
     bin_size = size of the bins to compute the PSTH (s)

     OUPUT:
     AP_PSTH = matrix containing the time vector (column 1) (s) and the firing
     rate (column 2) (Hz) for each time bin."""


    AP_PSTH=[]
    AP_PSTH_pre=[]
    AP_PSTH_post=[]

    bin_pt=np.round(bin_size*SR_Vm)

    pt0=int(np.floor(Pre_Window*SR_Vm))
    pt2=pt0
    pt_min=int(pt0-np.floor(pt0/bin_pt)*bin_pt+1)

    cnt=0

    while(pt2>pt_min):

        pt1=int(pt0-(bin_pt*(cnt)))
        pt2=int(pt1-bin_pt+1)
        AP_PSTH_pre.append([-1*bin_size*(cnt+1),np.sum(AP_avg[pt2:pt1])/bin_size])
        cnt+=1



    pt0=int(np.floor(Pre_Window*SR_Vm))
    pt2=pt0
    pt_max=int(pt0+np.floor((Post_Window*SR_Vm)/bin_pt)*bin_pt)

    cnt=0

    while(pt2<pt_max):

        pt1=int(pt0+(bin_pt*cnt))
        pt2=int(pt1+bin_pt)
        AP_PSTH_post.append([bin_size*(cnt),np.sum(AP_avg[pt1:pt2])/bin_size])
        cnt+=1

    AP_PSTH=np.concatenate((AP_PSTH_pre, AP_PSTH_post), axis=0)
    # AP_PSTH=np.sort(AP_PSTH,axis=0)
    AP_PSTH=AP_PSTH[AP_PSTH[:, 0].argsort(),:]
    return AP_PSTH