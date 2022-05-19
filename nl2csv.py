#!/usr/bin/env python3
import re, time, csv
import subprocess as sp
from sys import argv
from pathlib import Path
from argparse import ArgumentParser
from halo import Halo

SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)

FILTER = {
    'RX_BYTES'   : re.compile('rx bytes:\s*(\d*)'),
    'RX_PACKETS' : re.compile('rx packets:\s*(\d*)'),
    'RX_BITRATE' : re.compile('rx bitrate:\s*.*MCS\s*(.*)'),
    'SIGNAL_CHAIN' : re.compile('signal:\s*-\d+\s*\[(-\d+,\s+-\d+)\]\s+dBm')
    #re.compile('signal avg:\s*(-\d*)\s*dBm')
}

def fetch_statistics(net_dev, sta_mac):
    ret = SHELL_RUN( f'iw dev {net_dev} station get {sta_mac}' )
    _output = ret.stdout.decode()
    
    result = {}
    for _name, _filter in FILTER.items():
        _ret = _filter.findall(_output)
        _ret = _ret[0] if len(_ret)>0 else ''
        result[_name] = _ret
        pass
    return result

def main(args):
    postfix = time.strftime('%Y_%m_%d_%H%M%S')
    filename = f'logs/{args.net_dev}_{postfix}.csv'
    Path('logs').mkdir(exist_ok=True)

    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=',', quotechar='|')
        writer.writerow(['timestamp', *list(FILTER.keys())])
        #
        with Halo('Collecting ...') as spinner:
            try:
                counter, start_time = 0, time.time()
                while(True):
                    timestamp  = time.time()
                    delta_time = time.time() - start_time
                    if delta_time >= args.omit:
                        result = fetch_statistics(args.net_dev, args.sta_mac)
                        writer.writerow([timestamp, *list(result.values())])
                        counter += 1
                    #
                    elapsed_time = delta_time - args.omit
                    spinner.text = f'Time Elapsed: {elapsed_time:7.2f} s; {counter} Collected.'
                    #
                    time.sleep(0.01)
            except Exception as e:
                spinner.warn()
                raise e
        pass
    
    pass

if __name__=='__main__':
    try:
        parser = ArgumentParser(description='Nl80211 to CSV, Python version.')
        parser.add_argument('net_dev', metavar='NET_DEV', help='e.g., wlp1s0.')
        parser.add_argument('sta_mac', metavar='STA_MAC', help='MAC address of the associated STA.')
        parser.add_argument('-O', '--omit', type=float, default=0.0,
                            help='Omit the beginning period for logging (Unit: second).')
        #
        args = parser.parse_args()
        main(args)
    except Exception as e:
        raise e
    finally:
        pass
