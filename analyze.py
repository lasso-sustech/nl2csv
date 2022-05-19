#!/usr/bin/env python3
import csv
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

HT20_MAP = {
    '15 short GI': 144.4,
    '14 short GI': 130.0,
    '13 short GI': 115.6,
    '12 short GI': 86.7,
    '11 short GI': 57.8,
    '10 short GI': 43.3,
    '9 short GI' : 28.9,
    #
    '8 short GI' : 14.4,
    '7 short GI' : 72.2,
    '6 short GI' : 65.0,
    '5 short GI' : 57.8,
    '4 short GI' : 43.3,
    '3 short GI' : 28.9,
    '2 short GI' : 21.7,
    '1 short GI' : 14.4,
    '0 short GI' : 7.2,

    '15': 130.0,
    '14': 117.0,
    '13': 104.0,
    '12': 78.0,
    '11': 52.0,
    '10': 39.0,
    '9' : 26.0,
    '8' : 13.0,
    #
    '7' : 65.0,
    '6' : 58.5,
    '5' : 52.0,
    '4' : 39.0,
    '3' : 26.0,
    '2' : 19.5,
    '1' : 13.0,
    '0' : 6.5,

    ''  : 0.0
}

def get_cdf(y):
    pmf_x = np.linspace( 0, np.max(y)*1.001, num=len(y) )
    pmf_y = np.zeros(len(y))
    #
    for i in range(1,len(y)):
        pmf_y[i] = np.logical_and( y>=pmf_x[i-1], y<pmf_x[i] ).sum()
    cdf_y = np.cumsum(pmf_y) / len(y)
    return (pmf_x, cdf_y)

def moving_average(x, w):
    ret = np.convolve(x, np.ones(w), 'valid') / w
    return np.concatenate(( ret, ret[-w:] ))

def block_average(x, w):
    _fix = len(x) % w
    if _fix:
        x = np.concatenate(( x,x[-(w-_fix):] ))
    ret = np.mean( np.reshape(x, (w,-1)), axis=0 )
    ret = np.repeat(ret, w)
    if _fix:
        ret = ret[:-(w-_fix)]
    return np.append(ret, ret[-1])

def block_max(x, w):
    _fix = len(x) % w
    if _fix:
        x = np.concatenate(( x,x[-(w-_fix):] ))
    ret = np.max( np.reshape(x, (w,-1)), axis=0 )
    ret = np.repeat(ret, w)
    if _fix:
        ret = ret[:-(w-_fix)]
    return ret.append(ret, ret[-1])

def analyze_thru_vs_mcs(timestamp, rx_bytes, rx_packets, mcs_thru):
    acc_thru = np.diff(rx_bytes) / np.diff(timestamp) *(8/1E6)
    time_part = np.clip( acc_thru / mcs_thru[1:], 0.0, 1.0 )
    _pkt_thru = np.diff(rx_packets) / np.diff(timestamp)
    est_time_part = _pkt_thru / _pkt_thru.max()
    est_thru  = est_time_part * mcs_thru[1:]
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    ## plot throughput comparison
    ax1.plot(timestamp, mcs_thru, color='darkorange')
    ax1.plot(timestamp, [acc_thru[0], *acc_thru], color='blue')
    ax1.plot(timestamp, [est_thru[0], *est_thru], color='green')
    ax1.set_xlabel('Timestamp (second)')
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.legend(['MCS-based', 'Real-time', 'Estimated'])

    ## plot average accumulated throughput
    # _w = 2
    # smooth_thru = block_average(acc_thru, _w)
    # plt.plot(timestamp, smooth_thru, color='green')

    ## plot access-time driven throughput

    ## plot real access time
    # ax1_2 = ax1.twinx()
    # ax1_2.plot(timestamp, [time_part[0], *time_part], color='green')
    # ax1_2.set_ylabel('Access Time Percentage')

    ## plot cdf
    ax2.plot( *get_cdf(est_time_part), color='blue' )
    ax2.plot( *get_cdf(time_part), color='green' )
    ax2.set_xlabel('Access Time')
    ax2.set_ylabel('CDF')
    ax2.legend(['Real-time / MCS-based', 'Packet-based Estimation'])
    ax2.set_ylabel('CDF')
    #
    # ax2_1 = ax2.twiny()
    # ax2_1.plot( *get_cdf(mcs_thru), color='darkorange' )
    # ax2_1.plot( *get_cdf(acc_thru), color='blue' )
    # ax2_1.legend(['MCS-based', 'Real-time'])

    plt.show()
    pass

def analyze_mcs_vs_rssi(timestamp, mcs_thru, mcs_idx, sig_a, sig_b):
    sig_min = [ min(*x) for x in zip(sig_a, sig_b) ]
    sig_max = [ max(*x) for x in zip(sig_a, sig_b) ]

    fig, ax = plt.subplots()
    ax.plot(timestamp, mcs_thru, color='blue')
    #
    axp = ax.twinx()
    axp.plot(timestamp, sig_max, color='red')
    axp.plot(timestamp, sig_min, color='green')

    plt.show()
    pass

def main(filename):
    with open(filename) as fh:
        reader = csv.reader(fh, delimiter=',')
        _title = next(reader)
        
        result = []
        for item in reader:
            item = [
                float(item[0]), #timestamp,
                int(item[1]),   # RX_BYTES
                int(item[2]),   # RX_PACKETS
                HT20_MAP[item[3]],       # throughput
                HT20_MAP.index(item[3]), # throughput index
                int(item[4].strip('|')), # last RSSI of chain A 
                int(item[5].strip('|')), # last RSSI of chain B
            ]
            result.append( item )
        timestamp, rx_bytes, rx_packets, mcs_thru, mcs_idx, sig_a, sig_b = list( zip(*result) )

        # analyze_thru_vs_mcs(timestamp, rx_bytes, rx_packets, mcs_thru)
        analyze_mcs_vs_rssi(timestamp, mcs_thru, mcs_idx, sig_a, sig_b)
    pass

if __name__=='__main__':
    try:
        if len(argv)<2:
            print('Usage: ./analyze.py <csv_file>')
        else:
            main(argv[1])
    except Exception as e:
        raise e
    finally:
        pass
