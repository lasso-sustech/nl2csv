#!/usr/bin/env python3
import re, time, csv
import subprocess as sp
from sys import argv
from pathlib import Path
from halo import Halo

SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)

FILTER = {
    'RX_BYTES'   : re.compile('rx bytes:\s*(\d*)'),
    'RX_PACKETS' : re.compile('rx packets:\s*(\d*)'),
    'RX_BITRATE' : re.compile('rx bitrate:\s*.*MCS\s*(.*)'),
    'SIGNAL_AVG' : re.compile('signal avg:\s*(-\d*)\s*dBm')
}

def fetch_statistics(dev, mac_addr):
    ret = SHELL_RUN( f'iw dev {dev} station get {mac_addr}' )
    _output = ret.stdout.decode()
    
    result = {}
    for _name, _filter in FILTER.items():
        _ret = _filter.findall(_output)
        _ret = _ret[0] if len(_ret)>0 else ''
        result[_name] = _ret
        pass
    return result

def main(dev, mac_addr):
    postfix = time.strftime('%Y_%m_%d_%H%M%S')
    filename = f'logs/{dev}_{postfix}.csv'
    Path('logs').mkdir(exist_ok=True)

    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=',', quotechar='|')
        writer.writerow(['timestamp', *list(FILTER.keys())])
        #
        with Halo('Collecting ...') as spinner:
            counter, start_time = 0, time.time()
            while(True):
                timestamp = time.time()
                result = fetch_statistics(dev, mac_addr)
                writer.writerow([timestamp, *list(result.values())])
                time.sleep(0.01)
                #
                counter += 1
                time_delta = time.time() - start_time
                spinner.text = f'Time Elapsed: {time_delta:7.2f} s; {counter} Collected.'
        pass
    
    pass

if __name__=='__main__':
    try:
        if len(argv)!=3:
            print('Usage: ./nl2csv <NET_DEV> <STA_MAC>')
        else:
            main( dev=argv[1], mac_addr=argv[2] )
    except Exception as e:
        raise e
    finally:
        pass
