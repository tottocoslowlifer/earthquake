import os
import numpy as np
from obspy import Stream, Trace, UTCDateTime
import csv


def convert_ndarry_stream(data, time_str, station_name, sampling_rate=100):
    """
    Convert a 2D NumPy array of waveform data into an ObsPy Stream object.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (3, N), where N is the number of samples
        for each component (UD, NS, EW).
    time_str : str
        Start time string in the format 'yymmdd_HHMMSS'
        (e.g., '150323_141948').
    station_name : str
        Station name to be assigned to each trace.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 100 Hz.

    Returns
    -------
    stream : obspy.Stream
        ObsPy Stream object containing three Traces with appropriate metadata.
    """

    year = 2000 + int(time_str[:2])
    month, day = int(time_str[2:4]), int(time_str[4:6])
    hour, minute, second = (
        int(time_str[7:9]),
        int(time_str[9:11]),
        int(time_str[11:13])
    )
    utc_time = UTCDateTime(year, month, day, hour, minute, second)

    channels = ["UD", "NS", "EW"]
    stream = Stream()
    for i, ch in enumerate(channels):
        trace = Trace(data=data[i, :])
        trace.stats.update({
            "sampling_rate": sampling_rate,
            "starttime": utc_time,
            "network": "MeSO-net",
            "station": station_name,
            "location": "",
            "channel": ch,
        })
        stream.append(trace)
    return stream


def convert_stream_to_ndarray(stream, channel_order=["UD", "NS", "EW"]):
    """
    Convert ObsPy Stream to ndarray of shape (samples, channels).

    Parameters:
        stream (obspy.Stream): Stream object containing 3 components.
        channel_order (list): Order of channels to extract,
        default is ["UD", "NS", "EW"].

    Returns:
        np.ndarray: Array of shape (3, n_samples)
    """
    traces = []
    for ch in channel_order:
        tr = stream.select(channel=ch)
        if len(tr) == 0:
            raise ValueError(f"Channel {ch} not found in the stream.")
        traces.append(tr[0].data)

    # Stack and transpose to shape (samples, channels)
    data = np.stack(traces)
    return data


def calc_snr(signal, noise):
    """
    Calculate signal-to-noise ratio (SNR) in decibels (dB).

    Parameters:
    ----------
    signal : np.ndarray
        Array containing the signal portion.
    noise : np.ndarray
        Array containing the noise portion.

    Returns:
    -------
    float
        SNR value in dB.
    """
    return 10 * np.log10(np.std(signal) / np.std(noise))


def calc_cc(a, b):
    """
    Calculate the Pearson correlation coefficient between two signals.

    Parameters:
    ----------
    a : np.ndarray
        First signal.
    b : np.ndarray
        Second signal.

    Returns:
    -------
    float
        Correlation coefficient between a and b (range: -1 to 1).
    """
    return np.corrcoef(a, b)[0, 1]


def zscore(data):
    """
    Normalize each channel using z-score normalization
    (zero mean, unit variance).

    Parameters:
    ----------
    data : np.ndarray
        2D array of shape (channels, time), e.g., (3, N).

    Returns:
    -------
    np.ndarray
        Z-score normalized data with the same shape.
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data


def calc_loss(data1, data2, p_onset, s_onset, sf=100):

    '''
    Calculate a loss value based on signal-to-noise ratio (SNR)
    and correlation coefficients (CC)
    between original and denoised seismic waveform data
    for P-wave, S-wave, and noise segments.

    Parameters:
    ----------
    data1 (Original wave) : np.ndarray
        Original waveform data of shape (3, N),
        where N is the number of time steps.
    data2 (Denoised wave) : np.ndarray
        Denoised waveform data of shape (3, N), corresponding to data1.

    Returns:
    -------
    loss : float
        Averaged loss across 3 channels, combining SNR and CC values.
    P_SNR : float
        Averaged P-wave SNR after denoising.
    S_SNR : float
        Averaged S-wave SNR after denoising.
    P_CC : float
        Averaged correlation coefficient between original and
        denoised P-wave signals.
    s_cc : float
        Averaged correlation coefficient between original and
        denoised S-wave signals.
    n_cc : float
        Averaged correlation coefficient between original and
        denoised noise segments.
    '''

    p_snrs = []
    s_snrs = []
    p_ccs = []
    s_ccs = []
    n_ccs = []

    loss = []

    for ch in [0, 1, 2]:
        orig_data = data1[ch, :]
        den_data = data2[ch, :]

        signal_p = orig_data[p_onset: p_onset+sf*5]
        noise_p = orig_data[p_onset-sf*5: p_onset]

        signal_p_deno = den_data[p_onset: p_onset+sf*5]
        noise_p_deno = den_data[p_onset-sf*5: p_onset]

        signal_s = orig_data[s_onset: s_onset+sf*5]
        signal_s_deno = den_data[s_onset: s_onset+sf*5]

        # snr_p_orig = calc_snr(signal_p, noise_p)
        # snr_s_orig = calc_snr(signal_s, noise_p)

        snr_p_deno = calc_snr(signal_p_deno, noise_p_deno)
        snr_s_deno = calc_snr(signal_s_deno, noise_p_deno)

        cc_n = calc_cc(noise_p, noise_p_deno)
        cc_p = calc_cc(signal_p, signal_p_deno)
        cc_s = calc_cc(signal_s, signal_s_deno)

        loss_ch = (snr_p_deno + snr_s_deno) * cc_p * cc_s * cc_n

        p_snrs.append(snr_p_deno)
        s_snrs.append(snr_s_deno)

        p_ccs.append(cc_p)
        s_ccs.append(cc_s)
        n_ccs.append(cc_n)

        loss.append(loss_ch)

    return np.mean(loss), p_snrs, s_snrs, p_ccs, s_ccs, n_ccs


def evaluate(raw_dir, w_dir, model, model_name):
    total_loss = 0

    # Load pretrained denoising model
    model = model

    # Prepare data
    raw_dir = raw_dir
    files = sorted(os.listdir(raw_dir))
    # print(len(files))

    filename = w_dir+'/'+model_name+'.csv'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # headder
        writer.writerow(['FileName', 'LOSS', 'UD_P_SNR', 'UD_S_SNR',
                         'UD_P_CC', 'UD_S_CC', 'UD_N_CC', 'NS_P_SNR',
                         'NS_S_SNR', 'NS_P_CC', 'NS_S_CC', 'NS_N_CC',
                         'EW_P_SNR', 'EW_S_SNR', 'EW_P_CC', 'EW_S_CC',
                         'EW_N_CC'])

        c = 0

        for fn in files:

            if c % 100 == 0:
                print(f'{c}/{len(files)}')

            data = np.load('../data/Learning'+'/'+fn, allow_pickle=True)
            wave, p_onset, s_onset = data['wave'], data['pidx'], data['sidx']
            time_str, station_name = os.path.basename(fn).replace(
                '.npz', ''
                ).split('_')

            original_stream = convert_ndarry_stream(
                wave-np.mean(wave, axis=1, keepdims=True), time_str,
                station_name)
            denoised_stream = model.annotate(original_stream)

            original = convert_stream_to_ndarray(
                original_stream, channel_order=["UD", "NS", "EW"])
            denoised = convert_stream_to_ndarray(
                denoised_stream,
                channel_order=[
                    model_name+"_UD",
                    model_name+"_NS",
                    model_name+"_EW",
                ])

            loss, p_snrs, s_snrs, p_ccs, s_ccs, n_ccs = calc_loss(
                original, denoised, p_onset, s_onset
            )

            writer.writerow([
                os.path.basename(fn),
                loss,
                p_snrs[0], s_snrs[0], p_ccs[0], s_ccs[0], n_ccs[0],
                p_snrs[1], s_snrs[1], p_ccs[1], s_ccs[1], n_ccs[1],
                p_snrs[2], s_snrs[2], p_ccs[2], s_ccs[2], n_ccs[2],
            ])

            # print(os.path.basename(fn), loss, p_snr, s_snr, P_CC, s_cc, n_cc)

            total_loss += loss

            c += 1

    return total_loss/(len(files))


def experiment(model, model_name):
    raw_dir = '../data/Learning'
    w_dir = '../data/results'

    for dir in [raw_dir, w_dir]:
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)

    print(evaluate(raw_dir, w_dir, model, model_name))
