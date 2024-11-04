from devicelib.device import Device
from devicelib.stream import stream_process
import os
import time

path_now = os.path.expanduser('~/')

# create device object
device_dataControl = Device()

# perform pwd to check directory

GLOBAL_STREAM_IP = '103.193.179.252'
# procedure to get stream key

def write_key(stream_key):
    file_path = f'{path_now}/key.txt'
    print(f'Stream Key preview for admin stored in : {file_path}')
    with open(file_path, 'w') as file:
        file.write(stream_key)

def device_get_stream_key():
    # system clear screen in python
    while True:
        print('\033c')
        status, stream_key = device_dataControl.run_process()
        print(f'[STREAM_KEY] : Response Status : {status}')
        if stream_key:
            print(f'[STREAM_KEY] : {stream_key}')
            write_key(stream_key)
            time.sleep(2)
            return stream_key
        else:
            print('[STREAM_KEY] : Stream Key not granted')
            print('[STREAM_KEY] : Restarting the process until stream key is granted')
            # sleep for 5 seconds
            time.sleep(5)

    return stream_key

if __name__ == '__main__':
    while True:
        try:
            strkey = device_get_stream_key()
            print(f'[STREAM_KEY] : {strkey}')
            if strkey is None:
                continue
            while True:
                stream_process(stream_ip=GLOBAL_STREAM_IP, stream_key=strkey)
        except Exception as e:
            print(f'Error with pesan\t: {e}')
        time.sleep(5)
