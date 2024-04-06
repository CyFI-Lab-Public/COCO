from pathlib import Path
import multiprocessing as mp
import gzip
import json
from tqdm import tqdm
import os

# Path to the traces gz folder
traces_gz_folder = Path(os.environ.get("TRACES_GZ_FOLDER", "/tmp"))

# Path to the transactions gz folder
txn_gz_folder = Path(os.environ.get("TXN_GZ_FOLDER", "/tmp"))

# Path to the contract gz folder
contracts_gz_folder = Path(os.environ.get("CONTRACTS_GZ_FOLDER", "/tmp"))

# Path to the cache folder
cache_folder = Path(os.environ.get("CACHE_FOLDER", "/tmp"))

def load_json_gz_file(file_path):
    objects = []

    with gzip.open(file_path, 'rt') as f:  # 'rt' for reading text
        for line in f:
            obj = json.loads(line)
            objects.append(obj)

    return objects

def mp_save_data(json_path, data):
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to save the json first, if it fails, save the binary
    try:
        with json_path.open("w") as f:
            json.dump(data, f)
    except:
        with json_path.open("wb") as f:
            f.write(data)

def mp_load_data(args):
    key, json_path = args
    if json_path.exists():
        # try to load the json first, if it fails, load the binary
        try:
            with json_path.open("r") as f:
                return [key, json.load(f)]
        except:
            with json_path.open("rb") as f:
                return [key, f.read()]


def mp_get_all_txn_given_hash_list(args):
    ret = {}

    hash_list, gz_file = args

    hash_list = set([hash.lower() for hash in hash_list])

    # Load the gz file
    txns = load_json_gz_file(gz_file)

    # Go through each txn record
    for txn in txns:
        if txn['hash'].lower() in hash_list:
            ret[txn['hash'].lower()] = txn

    return ret

def get_all_txns_given_hash_list(txn_hash_list):
    # Get all raw transactions data from a list of transaction hashes

    # Turn the hash list into a set and to lower case
    txn_hash_set = set([txn_hash.lower() for txn_hash in txn_hash_list])

    # Try to load from the cache first so that we dont need to go thorugh this again
    txns = {}

    # Use mp to load the cache
    args = [(txn_hash, cache_folder / "transactions" / "hash" / txn_hash) for txn_hash in txn_hash_set]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                txns[result[0]] = result[1]

    # Get the list of txn_hash that we need to query
    txn_hash_list_to_query = [txn_hash for txn_hash in txn_hash_set if txn_hash not in txns]

    if not txn_hash_list_to_query:
        return txns

    # Get all gz files first
    all_gz_files = list(txns_gz_folder.iterdir())

    # Build the argument list with the txn_hash
    args = [(txn_hash_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_all_txn_given_hash_list, args)

        for result in tqdm(results, total=len(args)):
            txns.update(result)

    # Save the cache and return
    # Make sure the folder exists
    # Create a mp Pool
    # We wonly want to cache the ones that we have not cached
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async() to run the function in parallel
        results = [pool.apply_async(mp_save_data, (cache_folder / "transactions" / "hash" / txn_hash, txns[txn_hash])) for txn_hash in txns if txn_hash in txn_hash_list_to_query]

        # wait for all tasks to complete
        for result in results:
            result.get()

    return txns

def mp_get_all_txn_from_to_address_list(args):
    ret = {}

    address_list, gz_file = args

    address_list = set([address.lower() for address in address_list])

    if len(address_list) == 0:
        return ret

    # Load the gz files
    txns = load_json_gz_file(gz_file)

    # Go through each txn record
    for txn in txns:
        if 'from_address' in txn and txn['from_address'].lower() in address_list:
            if txn['from_address'].lower() not in ret:
                ret[txn['from_address'].lower()] = {}

            if 'from' not in ret[txn['from_address'].lower()]:
                ret[txn['from_address'].lower()]['from'] = []

            ret[txn['from_address'].lower()]['from'].append(txn)

        if 'to_address' in txn and txn['to_address'].lower() in address_list:
            if txn['to_address'].lower() not in ret:
                ret[txn['to_address'].lower()] = {}

            if 'to' not in ret[txn['to_address'].lower()]:
                ret[txn['to_address'].lower()]['to'] = []

            ret[txn['to_address'].lower()]['to'].append(txn)

    return ret

def mp_get_all_traces_from_to_address_list(args):
    ret = {}

    address_list, gz_file = args

    address_list = set([address.lower() for address in address_list])

    if len(address_list) == 0:
        return ret

    # Load the gz files
    traces = load_json_gz_file(gz_file)

    # Go through each txn record
    for trace in traces:
        if 'from_address' in trace and trace['from_address'].lower() in address_list:
            if trace['from_address'].lower() not in ret:
                ret[trace['from_address'].lower()] = {}

            if 'from' not in ret[trace['from_address'].lower()]:
                ret[trace['from_address'].lower()]['from'] = []

            ret[trace['from_address'].lower()]['from'].append(trace)

        if 'to_address' in trace and trace['to_address'].lower() in address_list:
            if trace['to_address'].lower() not in ret:
                ret[trace['to_address'].lower()] = {}

            if 'to' not in ret[trace['to_address'].lower()]:
                ret[trace['to_address'].lower()]['to'] = []

            ret[trace['to_address'].lower()]['to'].append(trace)

    return ret

def get_all_traces_from_to_address_list(address_list):
    # Get all the traces from and to a list of addresses
    # We try to comine as much as possible since it is a very expensie function to run
    address_list = set([address.lower() for address in address_list])


    address_traces = {}

    # Use mp to load from the cache first so that we dont need to go through this again
    args = [(address, cache_folder / "traces" / "address" / address) for address in address_list]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                address_traces[result[0]] = result[1]

    # Get the list of address that we need to query
    address_list_to_query = [address for address in address_list if address not in address_traces]

    if not address_list_to_query:
        return address_traces

    # Get all gz files first
    all_gz_files = list(traces_gz_folder.iterdir())

    # Build the argument list with the address
    args = [(address_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_all_traces_from_to_address_list, args)

        for result in tqdm(results, total=len(args)):
            # We need to merge the result in address_traces
            for address, trace_info in result.items():
                if not address in address_traces:
                    address_traces[address] = trace_info
                else:
                    for direction, traces in trace_info.items():
                        if direction not in address_traces[address]:
                            address_traces[address][direction] = []

                        address_traces[address][direction].extend(traces)

    # Save the cache and return
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async() to run the function in parallel
        results = [pool.apply_async(mp_save_data, (cache_folder / "traces" / "address" / address, address_traces[address])) for address in address_traces if address in address_list_to_query]

        # wait for all tasks to complete
        for result in results:
            result.get()

    return address_traces

def get_all_txn_from_to_address_list(address_list):
    # Get all the transactions from and to a list of addresses
    # We try to comine as much as possible since it is a very expensie function to run
    address_list = set([address.lower() for address in address_list])

    address_txns = {}

    # Use mp to load from the cache first so that we dont need to go through this again
    args = [(address, cache_folder / "transactions" / address) for address in address_list]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                address_txns[result[0]] = result[1]

    # Get the list of address that we need to query
    address_list_to_query = [address for address in address_list if address not in address_txns]

    if not address_list_to_query:
        return address_txns

    # Get all gz files first
    all_gz_files = list(txns_gz_folder.iterdir())

    # Build the argument list with the address
    args = [(address_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_all_txn_from_to_address_list, args)

        for result in tqdm(results, total=len(args)):
            # We need to merge the result in address_txn
            for addr, txn_info in result.items():
                if not addr in address_txns:
                    address_txns[addr] = txn_info

                else:
                    for direction, txns in txn_info.items():
                        if direction not in address_txns[addr]:
                            address_txns[addr][direction] = []

                        address_txns[addr][direction].extend(txns)

    # Save the cache and return
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async() to run the function in parallel
        results = [pool.apply_async(mp_save_data, (cache_folder / "transactions" / address, address_txns[address])) for address in address_txns if address in address_list_to_query]

        # wait for all tasks to complete
        for result in results:
            result.get()

    return address_txns


def mp_get_all_traces_given_hash_list(args):
    ret = {}

    txn_hash_list, gz_file = args

    txn_hash_list = set(txn_hash_list)

    # Early return if txn_hash_list is empty
    if len(txn_hash_list) == 0:
        return ret

    # Load the gz file
    traces = load_json_gz_file(gz_file)

    # Go through each trace record
    for trace in traces:
        if 'transaction_hash' in trace and trace['transaction_hash'].lower() in txn_hash_list:
            if trace['transaction_hash'] not in ret:
                ret[trace['transaction_hash']] = []
            ret[trace['transaction_hash']].append(trace)

    return ret

def get_all_traces_given_txn_list(txn_hash_list):

    # Turn each hash into lower case
    txn_hash_list = [txn_hash.lower() for txn_hash in txn_hash_list]

    # Get all the traces associted with a txn_hash_list
    traces = {}

    # Use mp to load the cache
    args = [(txn_hash, cache_folder / "traces" / "transactions" / txn_hash) for txn_hash in txn_hash_list]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                traces[result[0]] = result[1]

    # Get the list of txn_hash that we need to query
    txn_hash_list_to_query = [txn_hash for txn_hash in txn_hash_list if txn_hash not in traces]

    # Get all gz filse first
    all_gz_files = list(traces_gz_folder.iterdir())

    # Build the argument list with the txn_hash
    args = [(txn_hash_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_all_traces_given_hash_list, args)

        for result in tqdm(results, total=len(args)):
            for txn_hash in result:
                if txn_hash not in traces:
                    traces[txn_hash] = []
                traces[txn_hash].extend(result[txn_hash])

    # Save the cache and return
    # Make sure the folder exists
    # Create a multiprocessing Pool
    # We only wan to cache the ones that we have not cached
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async to call process_trace with arguments
        results = [pool.apply_async(mp_save_data, 
                                    (cache_folder / "traces" / "transactions" / txn_hash, traces[txn_hash])) 
                   for txn_hash in traces if txn_hash in txn_hash_list_to_query]

        # Wait for all tasks to complete
        for result in results:
            result.get()

    return traces

def mp_get_creation_txn_given_contract(args):
    ret = {}

    contract_list, gz_file = args

    contract_list = set(contract_list)

    # Load the gz file
    traces = load_json_gz_file(gz_file)

    # Go through each trace record
    for trace in traces:
        if 'trace_type' in trace and trace['trace_type']== 'create' and 'to_address' in trace and trace['to_address'].lower() in contract_list:
            ret[trace['to_address']] = trace['transaction_hash']

    return ret

def get_creation_txn_given_contract(contract_list):
    # Get the creation transaction of a contract

    # Turn each hash into lower case set
    contract_list = set([contract.lower() for contract in contract_list])

    # Try to load from the cache first so that we dont need to go through this again
    contract_creation_txn = {}

    # Use mp to load the cache
    args = [(contract, cache_folder / "creation_txn" / contract) for contract in contract_list]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                contract_creation_txn[result[0]] = result[1]

    # Get the list of contract that need to query db
    contract_list_to_query = tuple([contract for contract in contract_list if contract not in contract_creation_txn])

    if not contract_list_to_query:
        return contract_creation_txn

    # Get all gz files first
    all_gz_files = list(traces_gz_folder.iterdir())

    # Build the argument list with contract_list
    args = [(contract_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_creation_txn_given_contract, args)

        for result in tqdm(results, total=len(args)):
            contract_creation_txn.update(result)

    # Save the cache and return
    # Make sure the folder exists
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async
        results = [pool.apply_async(mp_save_data, (cache_folder / "creation_txn" / contract, contract_creation_txn[contract])) for contract in contract_creation_txn if contract in contract_list_to_query]

        # wait for all tasks to complete
        for result in results:
            result.get()

    return contract_creation_txn

def mp_get_all_code_from_address(args):
    ret = {}

    address_list, gz_file = args

    address_list = set(address_list)

    # Early return if address_list is empty
    if len(address_list) == 0:
        return ret

    # Load the gz file
    contracts = load_json_gz_file(gz_file)

    # Go through each contract record
    for contract in contracts:
        if contract['address'].lower() in address_list:
            ret[contract['address'].lower()] = contract['bytecode']

    return ret

def get_contracts_code(address_list):
    # Get codes of all contracts in address_list
    # Return format:
    # {
    #   "address1": "code1",
    #   "address2": "code2",
    #   ...
    # }

    address_list = [address.lower() for address in address_list]

    contract_code = {}

    # use mp to load the cache
    args = [(address, cache_folder / "contracts" / "code" / address) for address in address_list]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_load_data, args)

        for result in tqdm(results, total=len(args)):
            if result is not None:
                contract_code[result[0]] = result[1]

    # Get the list of address that we need to query
    address_list_to_query = tuple([address for address in address_list if address not in contract_code])

    # Get all gz filse first
    all_gz_files = list(contracts_gz_folder.iterdir())

    # Build the argument list with the address
    args = [(address_list_to_query, gz_file) for gz_file in all_gz_files]

    # Use multiprocessing to speed it up
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_get_all_code_from_address, args)

        for result in tqdm(results, total=len(args)):
            for address in result:
                if address not in contract_code:
                    contract_code[address] = result[address]

    # Save the cache and return
    # Make sure the folder exists
    # Create a multiprocessing Pool
    # We only wan to cache the ones that we have not cached
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use pool.apply_async to call process_trace with arguments
        results = [pool.apply_async(mp_save_data, 
                                    (cache_folder / "contracts" / "code" / address, contract_code[address])) 
                   for address in contract_code if address in address_list_to_query]

        # Wait for all tasks to complete
        for result in results:
            result.get()

    return contract_code
