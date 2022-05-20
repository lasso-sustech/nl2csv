#!/usr/bin/env python3
import csv
from tqdm import tqdm
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

CUDA = lambda x: x.cuda()

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
HT20_MAP_KEYS = list(HT20_MAP.keys())

W_SIZE   = 25

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs) - W_SIZE + 1
    
    def __getitem__(self, idx):
        _input = torch.FloatTensor( self.inputs[idx:idx+W_SIZE] )
        _label = self.labels[idx+W_SIZE-1]
        return _input, _label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(2*W_SIZE, 2970),
            nn.ReLU(),
            #
            nn.Linear(2970, 990),
            nn.ReLU(),
            nn.Linear(990, 330),
            nn.ReLU(),
            nn.Linear(330, 66),
            nn.ReLU(),
            nn.Linear(66, 33),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc_block(x)
    
    pass


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

def rssi_mcs_fitting(inputs, labels, training=True):
    dataset = CustomDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    net = Net()
    try:
        net.load_state_dict( torch.load('logs/latest.pth') )
        net.eval()
        print('Latest model loaded.')
    except:
        pass
    finally:
        net = CUDA(net)
    
    ## training
    if training:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        #
        pbar = tqdm(range(500))
        for epoch in pbar:
            current_loss = 0.0
            for i, (_inputs,_labels) in enumerate(dataloader, 0):
                _inputs, _labels = CUDA(_inputs), CUDA(_labels)
                optimizer.zero_grad()
                _outputs = net(_inputs)
                #
                _loss = loss_fn(_outputs, _labels)
                _loss.backward()
                optimizer.step()
                #
                current_loss += _loss.item()
            pbar.set_description( f'Loss: {current_loss:.6f}' )
        try:
            torch.save(net.state_dict(), 'logs/latest.pth')
            print('Latest model saved.')
        except:
            print('Model not saved.')
        pass
    ##
    return net, dataset

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
    # smooth_thru = block_average(acc_thru, W_SIZE)
    # plt.plot(timestamp, smooth_thru, color='green')

    ## plot access-time driven throughput

    ## plot real access time
    # ax1_2 = ax1.twinx()
    # ax1_2.plot(timestamp, [time_part[0], *time_part], color='green')
    # ax1_2.set_ylabel('Access Time Percentage')

    ## plot cdf
    ax2.plot( *get_cdf(est_time_part), color='blue' )
    ax2.plot( *get_cdf(time_part), color='green' )
    ax2.set_xlabel('Access Time Portion')
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
    _inputs = list(zip(sig_a, sig_b))
    _labels = mcs_idx

    
    net, dataset = rssi_mcs_fitting(_inputs, _labels, training=True)    
    ## testing
    # with torch.no_grad():
    #     x, _ = dataset[0]
    #     print( net( CUDA(x) ).cpu() )
    #     _preds = [ np.argmax( net(CUDA(x)).cpu() ).item() for x,_ in dataset ]
    #     _values = [ HT20_MAP[HT20_MAP_KEYS[x]] for x in _preds ]

    fig, ax = plt.subplots()
    # ax.plot(timestamp, mcs_thru, color='blue')
    # ax.plot(timestamp, _values, color='green')
    ax.set_xlabel('Timestamp (second)')
    ax.set_ylabel('Throughput (Mbps)')
    #
    # axp = ax.twinx()
    # axp.plot(timestamp, sig_max, color='red')
    # axp.plot(timestamp, sig_min, color='green')
    # axp.set_ylabel('RSSI (dBm)')

    # plt.show()
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
                HT20_MAP_KEYS.index(item[3]), # throughput index
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
