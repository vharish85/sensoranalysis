from scipy.signal import butter, lfilter


def butter_pass(cutoff, fs, type, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=type, analog=False)
    return b, a

def butter_filter(data, cutoff, fs, type, order=5):
    b, a = butter_pass(cutoff, fs, type,order=order)
    y = lfilter(b, a, data)
    return y


