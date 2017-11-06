from scipy.signal import butter, lfilter


def butter_pass(cutoff, fs, type, order=5):
    nyq = 0.5 * fs
    freq = cutoff / nyq    
    b, a = butter(order, freq , btype=type)
    return b, a


def butter_filter(data, cutoff, fs, type, order=5):
    b, a = butter_pass(cutoff, fs, type, order=order)
    y = lfilter(b, a, data)
    return y


