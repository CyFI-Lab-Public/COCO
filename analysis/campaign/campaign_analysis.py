from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import psycopg2
from psycopg2.extras import DictCursor

# read conn_auth and CACHE from the system environment
import os
conn_auth = os.environ['CONN_AUTH'].split(',')
CACHE = Path(os.environ['CACHE'])

# Make the cache folder if it doesn't exist
if not CACHE.exists():
    CACHE.mkdir(parents=True)

# Turn the job into batches
def batchnize_job(input_list, func, batch_size=10):
    # Split the input list into batches
    batches = [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    results = []

    # Give batches to func asynchronically and collect returns
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(func, batches), total=len(batches)))

    return results

# Load from cache
def load_cache(cache, input_list, file_type):
    # Turn input_list into lower case
    input_list = [item.lower() for item in input_list]

    # Check each item in input_list and load the cache using multiprocessing
    def load_item(args):
        item, file_type = args
        file_path = cache / item
        if file_path.exists():
            if file_type == 'json':
                with open(file_path, 'r') as f:
                    return [item, json.load(f)]
            elif file_type == 'pickle':
                with open(file_path, 'rb') as f:
                    return [item, pickle.load(f)]
            else:
                raise Exception('Invalid file type')

        return None

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(load_item, [(item, file_type) for item in input_list]), total=len(input_list)))

    return results

def query_sender_of_contracts(contracts):
    # Query the databse and return the raw database query result that could be used to see the sender
    
    contract_list = ','.join(f"'{contract.lower()}'" for contract in contracts)

    sql = f"SELECT * FROM traces WHERE to_address IN ({contract_list})"
    conn = psycopg2.connect(database=conn_auth[0], user=conn_auth[1], password=conn_auth[2], host=conn_auth[3], port=conn_auth[4])
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Build a map from contract to a txn set
    contract_txn_map = {}

    for row in rows:
        contract = row['to_address']
        if contract not in contract_txn_map:
            contract_txn_map[contract] = set()
        contract_txn_map[contract].add(row['transaction_hash'])

    # get all transactions and query the traces table where transaction_hash in (txns)
    txns = set()
    for contract in contract_txn_map:
        txns = txns.union(contract_txn_map[contract])

    txn_list = ','.join(f"'{txn}'" for txn in txns)

    sql = f"SELECT * FROM traces WHERE transaction_hash IN ({txn_list})"
    conn = psycopg2.connect(database=conn_auth[0], user=conn_auth[1], password=conn_auth[2], host=conn_auth[3], port=conn_auth[4])
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Build a map from txn to traces list
    txn_traces_map = {}
    
    for row in rows:
        txn = row['transaction_hash']
        if txn not in txn_traces_map:
            txn_traces_map[txn] = []
        txn_traces_map[txn].append(row)

    # Now connect contract_txn_map and txn_traces_map together
    contract_txn_traces_map = {}

    for contract in contract_txn_map:
        contract_txn_traces_map[contract] = []
        for txn in contract_txn_map[contract]:
            contract_txn_traces_map[contract].append(txn_traces_map[txn])

    # Caceh contract_txn_traces_map using pickle
    for contract in contract_txn_traces_map:
        with open(CACHE / 'sender' / contract, 'wb') as f:
            pickle.dump(contract_txn_traces_map[contract], f)

    return contract_txn_traces_map

def get_sender_of_contracts(contracts, batch_size=10):
    ret = {}

    # Turn contracts into lower case
    contracts = [contract.lower() for contract in contracts]

    # Try to check the cache first
    sender_cache = CACHE / 'sender'

    if not sender_cache.exists():
        sender_cache.mkdir(parents=True)

    # try to load from cache
    cache_results = load_cache(sender_cache, contracts, 'pickle')

    # Remove the contracts that already have a cache file
    contracts = list(set(contracts) - set([item[0] for item in cache_results if item is not None]))

    # Batchnize the job
    db_results = batchnize_job(contracts, query_sender_of_contracts, batch_size=batch_size)
