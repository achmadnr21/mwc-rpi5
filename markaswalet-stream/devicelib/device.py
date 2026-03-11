import sqlite3 as sq
import random
import string
import requests
import os
import logging
import traceback

logger = logging.getLogger(__name__)

class Device:
    def __init__(self, API_URL='https://markaswalet-iot.techiro.co.id/api/camera/'):
        self.API_URL = API_URL
        self.database_dir = os.path.expanduser('~/localdb')
        self.database_name = 'device.db'
        self.create_table_if_not_exist()
    def local_database_connection(self):
        db_dir = self.database_dir
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            self.create_table_if_not_exist()

        conn = sq.connect(f'{db_dir}/{self.database_name}')
        return conn
        
    def create_table_if_not_exist(self):
        # buat folder localdb jika belum ada
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_data (
            ID INTEGER PRIMARY KEY NOT NULL,
            name VARCHAR(255) NULL,
            location VARCHAR(255) NULL,
            password VARCHAR(64) NOT NULL,
            stream_key VARCHAR(255) NULL
            )
        ''')
        conn.commit()
        conn.close()

    def generate_password(self, length=64):
        characters = string.ascii_letters + string.digits + '#@%-!+'
        password = ''.join(random.choice(characters) for _ in range(length))
        return password

    def generate_streamKey(self, length=64):
        characters = string.ascii_letters + string.digits
        password = ''.join(random.choice(characters) for _ in range(length))
        return password
    
    def getDeviceName(self):
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name FROM device_data limit 1
        ''')
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            return None
    
    def is_check_local_data_exist(self):
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM device_data
        ''')
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def get_data(self):
        logger.info('[GET DATA] Getting data from API…')
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ID, password, name, location, stream_key FROM device_data limit 1
        ''')
        rows = cursor.fetchall()
        if rows:
            row = rows[0]
            local_device_id = row[0]
            local_device_password = row[1]
            local_device_name = row[2]
            local_device_location = row[3]
            local_device_stream_key = row[4]
        else:
            conn.close()
            return 404, None  
        params = {
        'id': local_device_id,
        'password': local_device_password
        }

        #lakukan request ke API
        response = requests.get(self.API_URL+'get-device-data', params=params)
        logger.info(f'[GET DATA] API response: {response.status_code}')

        granted_stream_key = None
        if response.status_code == 200:
            data = response.json()
            remote_device_id = data['id']
            remote_device_name = data['name']
            remote_device_location = data['location']
            remote_device_stream_key = data['stream_key']

            # update local db if data is different
            if local_device_name != remote_device_name or local_device_location != remote_device_location or local_device_stream_key != remote_device_stream_key:
                cursor.execute('''
                    UPDATE device_data SET name = ?, location = ?, stream_key = ? WHERE ID = ?
                ''', (remote_device_name, remote_device_location, remote_device_stream_key, remote_device_id))
                conn.commit()
            granted_stream_key = remote_device_stream_key
        else:
            granted_stream_key = None
        # close connection
        conn.close()
        return response.status_code, granted_stream_key

    def regist(self):
        # check if local data exist
        if self.is_check_local_data_exist():
            return self.get_data()
        
        logger.info('[REGIST] Registering device…')

        name = "Techiro Device"
        location = "Unset"
        generated_password = self.generate_password(32)

        data = {
            'name': name,
            'location': location,
            'password': generated_password
        }

        logger.debug(f'[REGIST] Registration payload: {data}')
        response = requests.post(self.API_URL + 'regist', json=data)


        granted_stream_key = None
        if response.status_code == 201 or response.status_code == 200:
            logger.info('[REGIST] Device registered successfully')
            stream_key = response.json()['stream_key']
            d_id = response.json()['id']

            conn = self.local_database_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO device_data (id, name, location, password, stream_key) VALUES (?, ?, ?, ?, ?)
            ''', ( d_id ,name, location, generated_password, stream_key))
            conn.commit()
            conn.close()
            granted_stream_key = stream_key

        else:
            logger.error(f'[REGIST] Registration failed — HTTP {response.status_code}: {response.text[:200]}')
            granted_stream_key = None

        return response.status_code, granted_stream_key

    def get_config(self):
        """Fetch device config from API using stored id+password.
        Returns a dict (config payload) or None if unavailable."""
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT ID, password FROM device_data LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        device_id, local_password = row
        params = {'id': device_id, 'password': local_password}
        try:
            response = requests.get(self.API_URL + 'device-config', params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('config')
        except Exception as e:
            logger.error(f'[GET_CONFIG] Error fetching config: {e}')
            logger.debug(traceback.format_exc())
        return None

    def report_bird_count(self, count_in: int, count_out: int, crossing_count: int = 0) -> bool:
        """POST accumulated bird count delta + current crossing_count to API. Returns True on success."""
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT ID, password FROM device_data LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if not row:
            return False
        device_id, local_password = row
        data = {
            'id': device_id,
            'password': local_password,
            'count_in': count_in,
            'count_out': count_out,
            'crossing_count': crossing_count,
        }
        try:
            response = requests.post(self.API_URL + 'bird-count', json=data, timeout=10)
            return response.status_code == 201
        except Exception as e:
            logger.error(f'[BIRD_COUNT] Error reporting count: {e}')
            logger.debug(traceback.format_exc())
        return False

    def get_today_count(self) -> dict:
        """Fetch today's accumulated counts + last crossing_count from API.
        Returns dict with total_in, total_out, crossing_count (all int, default 0)."""
        conn = self.local_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT ID, password FROM device_data LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {'total_in': 0, 'total_out': 0, 'crossing_count': 0}
        device_id, local_password = row
        try:
            response = requests.get(
                self.API_URL + 'bird-count-today',
                params={'id': device_id, 'password': local_password},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'total_in':       int(data.get('total_in', 0)),
                    'total_out':      int(data.get('total_out', 0)),
                    'crossing_count': int(data.get('crossing_count', 0)),
                }
        except Exception as e:
            logger.error(f'[GET_TODAY_COUNT] Error: {e}')
            logger.debug(traceback.format_exc())
        return {'total_in': 0, 'total_out': 0, 'crossing_count': 0}

    def run_process(self):
        status, stream_key = self.regist()
        return status, stream_key


    def print_status(self, status):
        if status == 200:
            print('Success')
        elif status == 201:
            print('Created')
        elif status == 400:
            print('Bad Request')
        elif status == 401:
            print('Unauthorized')
        elif status == 403:
            print('Forbidden')
        elif status == 404:
            print('Not Found')
        elif status == 405:
            print('Method Not Allowed')
        elif status == 409:
            print('Conflict')
        elif status == 500:
            print('Internal Server Error')
        else:
            print('Unknown Status Code')
