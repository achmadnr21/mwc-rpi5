from devicelib.device import Device
from devicelib.stream import stream_process
import os
import time
import logging
import traceback
import sys
from datetime import datetime
import zoneinfo

_WIB = zoneinfo.ZoneInfo('Asia/Jakarta')

# ─── Logging Setup ──────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    """Configure root logger to write to both stdout and a rotating log file."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir,
        f"mwc_service_{datetime.now(_WIB).strftime('%Y%m%d')}.log"
    )

    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    formatter.converter = lambda *args: datetime.now(_WIB).timetuple()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler — always visible in journalctl / terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — persistent log on disk
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f'Log file: {log_filename}')
    except Exception as e:
        root_logger.warning(f'Could not open log file {log_filename}: {e}')

    return root_logger


logger = _setup_logger()

# ─── Config ─────────────────────────────────────────────────────────────────

path_now = os.path.expanduser('~/')

# create device object
device_dataControl = Device()

GLOBAL_STREAM_IP = '46.250.229.47'

# Set this to a real stream key to bypass the API (for local testing when API is down).
# Leave as None to use the normal API registration flow.
LOCAL_TEST_STREAM_KEY = None

# ─── Helpers ────────────────────────────────────────────────────────────────

def write_key(stream_key: str):
    file_path = f'{path_now}/key.txt'
    logger.info(f'[STREAM_KEY] Stream Key preview stored at: {file_path}')
    try:
        with open(file_path, 'w') as file:
            file.write(stream_key)
        logger.debug('[STREAM_KEY] Key file written successfully')
    except Exception as e:
        logger.error(f'[STREAM_KEY] Failed to write key file: {e}')


def device_get_stream_key() -> str:
    """Loop until the API grants a stream key."""
    while True:
        print('\033c')
        logger.info('[STREAM_KEY] Requesting stream key from API…')
        try:
            status, stream_key = device_dataControl.run_process()
            logger.info(f'[STREAM_KEY] API response status: {status}')
        except Exception as e:
            logger.error(f'[STREAM_KEY] Exception during run_process: {e}')
            logger.debug(traceback.format_exc())
            time.sleep(5)
            continue

        if stream_key:
            logger.info(f'[STREAM_KEY] Stream key granted: {stream_key}')
            write_key(stream_key)
            time.sleep(2)
            return stream_key
        else:
            logger.warning('[STREAM_KEY] Stream key not granted — retrying in 5 s…')
            time.sleep(5)


# ─── Main loop ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('MarkasWalet Stream Service starting up')
    logger.info(f'Working directory : {os.getcwd()}')
    logger.info(f'Script directory  : {os.path.dirname(os.path.abspath(__file__))}')
    logger.info(f'Python executable : {sys.executable}')
    logger.info('=' * 60)

    loop_count = 0
    while True:
        loop_count += 1
        logger.info(f'[MAIN] ── Main loop iteration #{loop_count} ──')
        try:
            if LOCAL_TEST_STREAM_KEY:
                strkey = LOCAL_TEST_STREAM_KEY
                device_identity_name = device_dataControl.getDeviceName() or 'Local Test Device'
                logger.info(f'[MAIN] Using local test key: {strkey}')
            else:
                strkey = device_get_stream_key()
                device_identity_name = device_dataControl.getDeviceName()

            logger.info(f'[MAIN] Stream key : {strkey}')
            logger.info(f'[MAIN] Device name: {device_identity_name}')

            if strkey is None:
                logger.warning('[MAIN] Stream key is None — restarting key acquisition')
                continue

            inner_loop = 0
            while True:
                inner_loop += 1
                logger.info(f'[MAIN] Starting stream_process (attempt #{inner_loop})')
                try:
                    stream_process(
                        stream_ip=GLOBAL_STREAM_IP,
                        stream_key=strkey,
                        device_name=device_identity_name,
                    )
                except Exception as inner_e:
                    logger.error(f'[MAIN] stream_process raised an exception: {inner_e}')
                    logger.error(traceback.format_exc())
                    logger.info('[MAIN] Waiting 5 s before restarting stream_process…')
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info('[MAIN] KeyboardInterrupt received — shutting down gracefully')
            sys.exit(0)
        except Exception as e:
            logger.critical(f'[MAIN] Unhandled exception in main loop: {e}')
            logger.critical(traceback.format_exc())
            logger.info('[MAIN] Restarting main loop in 5 s…')
            time.sleep(5)
