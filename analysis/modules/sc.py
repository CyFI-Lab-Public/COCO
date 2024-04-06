# Smart Contract Related Modules
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from dateutil.tz import tzutc
from dateutil.parser import parse
from pathlib import Path
import time
import random
from typing import Callable
import json
from web3 import Web3, HTTPProvider
import re
import networkx as nx
import os
import pickle
import multiprocessing as mp
import numpy as np
import psycopg2


cache_path = ""

to_sleep = True

def keep_awake():
    global to_sleep
    to_sleep = False

def need_sleep():
    global to_sleep
    to_sleep = True

def is_to_sleep():
    return to_sleep

def query_database(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

# Down below are some analysis code
def match_eth_account(text):
    # Ethereum addresses start with '0x' and are followed by 40 hexadecimal characters.
    regex = r'\b0x[a-fA-F0-9]{40}\b'

    # Find all Ethereum addresses in the text.
    eth_addresses = re.findall(regex, text)

    return eth_addresses

# Down below are network related Ethereum scarping or query function
def set_cache_path(path):
    global cache_path
    cache_path = path

def get_today_date():
    return datetime.today().strftime('%Y-%m-%d')

def cache_response(response_text, key, data_type, cache_type, service="", convert_to_list=False):
    # data_type includes: txn, address, block, eoa

    file_path = Path(cache_path) / service / data_type / get_today_date() / f"{key}.{cache_type}"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # if file_path already exists and cache_type is json, try to merge the json
    if cache_type == "json":
        if file_path.exists():
            with open(str(file_path), "r") as f:
                old_response_text = f.read()
                # The old text should be a list while the response_text is a dict
                old_list = json.loads(old_response_text)
                new_json = json.loads(response_text)
                # If the old_list is actually a dictionary, we overwrite it
                if isinstance(old_list, dict):
                    old_list = new_json
                else:
                    old_list.append(new_json)

                response_text = json.dumps(old_list)

        elif convert_to_list:
            new_json = json.loads(response_text)
            response_text = json.dumps([new_json])

    with open(str(file_path), "w") as f:
        f.write(response_text)

def load_cache(key, data_type, random_cache, cache_type, service="", integrate_cache=False):
    """
    Load the cache from the cache folder
    @param key: the stem of the file name
    @param data_type: the type of the data, will be reflected in the cache path
    @param random_cache: if True, will randomly pick a date from the cache folder and load the cache
    @param cache_type: the type of the cache, can be html or json
    @param integrate_cache: if True, will load the cache from all the folders in the cache path. ONLY APPLIED TO JSON
    @return: the content of the cache, or None if no cache is found
    """

    if random_cache:
        # find a file named {key}.html in all folders in data_type
        # if there are multiple files, pick the first one we saw
        # if there's no file, return None

        # Keep track of the empty files, which will be removed
        empty_path = set()
        for path in (Path(cache_path)/service).glob(f"{data_type}/*/{key}.{cache_type}"):
            # We try to go through each file until we find the one whose content is not empty
            with open(str(path), "r") as f:
                content = f.read()
                if content:
                    return content
                
                else:
                    empty_path.add(path)

        # Remove the empty files
        for path in empty_path:
            path.unlink()

    if integrate_cache and cache_type == "json":
        # find all the files named {key}.json in all folders in data_type
        # if there are multiple files, combine them into a list
        # if there's no file, return None
        result = []
        for path in (Path(cache_path)/service).glob(f"{data_type}/*/{key}.{cache_type}"):
            with open(str(path), "r") as f:
                res_json = json.loads(f.read())
                if isinstance(res_json, list):
                    result.extend(res_json)
                else:
                    result.append(res_json)

        # Merge the json results together
        if result:
            return json.dumps(result)
        else:
            return None

    else:
        # use today's date to get the cache
        file_path = Path(cache_path) / service / data_type / get_today_date() / f"{key}.{cache_type}"
        
        if file_path.exists():
            with open(str(file_path), "r") as f:
                return f.read()

    return None

def easy_exponential_backoff(request_function: Callable, max_retries: int, maximum_backoff: int = 64):
    retries = 0
    backoff = 1

    while retries < max_retries:
        result = request_function()

        if result is None:
            # Calculate the wait time
            random_number_milliseconds = random.randint(0, 1000) / 1000
            wait_time = min((2 ** backoff) + random_number_milliseconds, maximum_backoff)
            
            # Wait and retry the request
            time.sleep(wait_time)
            retries += 1

        else:
            return result
    
    # If the maximum number of retries has been reached, return None
    return None

def exponential_backoff(request_function: Callable, max_retries: int, maximum_backoff: int = 64):
    retries = 0
    backoff = 1

    while retries < max_retries:
        status_code, result = request_function()
        
        if status_code == 429 or status_code is None:  # If the request is successful, return the result.

            # Calculate the wait time
            random_number_milliseconds = random.randint(0, 1000) / 1000
            wait_time = min((2 ** backoff) + random_number_milliseconds, maximum_backoff)
            
            # Wait and retry the request
            
            #print(f"Sleeping for {wait_time}s")
            
            time.sleep(wait_time)
            retries += 1

        else:
            return [status_code, result]
    
    # If the maximum number of retries has been reached, return None
    return [None, None]

def general_request(method: str, url: str, headers: dict = {}, params: dict = {}, json: dict = {}):
    # This function returns the status_code and json format of the response
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, json=json)
        elif method == "POST":
            response = requests.post(url, headers=headers, params=params, json=json)
        else:
            response = requests.get(url, headers=headers, params=params, json=json)
        status_code = response.status_code
        return [status_code, response.json()]

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return [None, None]


def infura_get_contract_hex_code(api_key, contract_address):
    """
    Get the bytecode of the contract using the infura
    """
    key = f"{contract_address}"

    hex_code = load_cache(key, "bytecode", True, "txt", "infura")

    if hex_code:
        return hex_code

    def query_contract_code(api_key, contract_address):
        # If we cant find the cache, query infura API
        w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{api_key}"))
        try:
            byte_code = w3.eth.get_code(w3.to_checksum_address(contract_address))
            hex_code = byte_code.hex()
            cache_response(hex_code, key, "bytecode", "txt", "infura")
            return hex_code
        except Exception as e:
            print(f"Error when querying infura: {e}")
            return None

    request_function = lambda: query_contract_code(api_key, contract_address)

    result = easy_exponential_backoff(request_function, 5)

    if result:
        return result

def alchemy_get_address_block_range(api_key, address):
    """
    Get the txn block range given a address
    """

    address = address.lower()

    first_block = "0x0"
    last_block = "0x105eb60"

    from_range = alchemy_get_txn_range_given_address(api_key, from_address=address)

    # first txn json
    first_txn_json = from_range[0][0]

    # Last txn json
    last_txn_json = from_range[1][0]

    if first_txn_json:
        first_block = first_txn_json["result"]["transfers"][0]["blockNum"]

    if last_txn_json:
        last_block = last_txn_json["result"]["transfers"][0]["blockNum"]

    return (first_block, last_block)

def alchemy_get_txn_count(api_key, address):
    """
    Get the txn count given a address
    """
    address = address.lower()
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    headers = {
        "content-type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getTransactionCount",
        "params": [address, "latest"],
        "id": 1,
    }

    # Wrap the request to alchemy in lambda
    # Print the payload in the command line
    request_function = lambda: general_request("POST", url, headers, {}, payload)

    status_code, result = exponential_backoff(request_function, 5)

    if status_code == 200 and result:
        if result.get('result'):
            return int(result.get('result'), 16)

    return None

def alchemy_get_address_time_range(api_key, address):

    address = address.lower()
    # Check txn from address
    from_range = alchemy_get_txn_range_given_address(api_key, from_address=address)
    to_range = alchemy_get_txn_range_given_address(api_key, to_address=address)

    from_range_time = (from_range[0][-1], from_range[1][-1])
    to_range_time = (to_range[0][-1], to_range[1][-1])

    # Now we pick the earliest and latest time, its possbile the time is None
    earliest_time = None
    latest_time = None

    if from_range_time[0] is not None and to_range_time[0] is not None:
        earliest_time = min(parse(from_range_time[0]), parse(to_range_time[0]))
    elif from_range_time[0] is not None:
        earliest_time = parse(from_range_time[0])

    elif to_range_time[0] is not None:
        earliest_time = parse(to_range_time[0])

    if from_range_time[1] is not None and to_range_time[1] is not None:
        latest_time = max(parse(from_range_time[1]), parse(to_range_time[1]))
    elif from_range_time[1] is not None:
        latest_time = parse(from_range_time[1])
    elif to_range_time[1] is not None:
        latest_time = parse(to_range_time[1])

    if earliest_time:
        earliest_time = str(int(earliest_time.replace(tzinfo=timezone.utc).timestamp()))

    if latest_time:
        latest_time = str(int(latest_time.replace(tzinfo=timezone.utc).timestamp()))


    return (earliest_time, latest_time)
        
def alchemy_get_txn_range_given_address(api_key, from_address="", to_address="", maxCount=1, pageKey="", contract_address=[], category=["external", "internal", "erc20", "erc721", "erc1155", "specialnft"], withMetadata=True, exclude_zero_value=False, file_key=None):
    """
    Query the transaction related to an address using Alcademy API and get the txn range
    @still_query: if True, will query the API even if the cache is available
    """

    first_txn_json = None
    last_txn_json = None

    # Convert all possible address to lower case
    from_address = from_address.lower()
    to_address = to_address.lower()
    contract_address = [address.lower() for address in contract_address]
    from_block = "0x0"
    to_block = "0x105eb60"

    # We always use the cache for the first txn
    key = f"{from_address}_{to_address}_{contract_address}_{category}_{withMetadata}_{exclude_zero_value}_{from_block}_{to_block}_asc"
    # Get rid of the "'" and "[]" in the key
    key = key.replace("'", "").replace("[", "").replace("]", "")

    # Replace ", " with "_" in the key
    key = key.replace(", ", "_")

    cache_text = load_cache(key, "address", True, "json", "alchemy")
    first_txn_json = json.loads(cache_text) if cache_text else {}
    if not first_txn_json:
        url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        headers = {
            "content-type": "application/json",
            "accept": "application/json"
        }

        payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "category": category,
                    "withMetadata": withMetadata,
                    "excludeZeroValue": exclude_zero_value,
                    "maxCount": hex(maxCount),
                    "order": "asc"
                    }]
                }

        if from_address:
            payload["params"][0]["fromAddress"] = from_address

        if to_address:
            payload["params"][0]["toAddress"] = to_address

        if contract_address:
            payload["params"][0]["contractAddresses"] = contract_address

        if pageKey:
            payload["params"][0]["pageKey"] = pageKey

        # Wrap the request to alchemy in lambda
        # Print the payload in the command line
        request_function = lambda: general_request("POST", url, headers, {}, payload)

        status_code, result = exponential_backoff(request_function, 5)

        if status_code == 200 and result and result.get("result", None) and result["result"].get("transfers", None):
            # Cache the result
            if not file_key:
                key = f"{from_address}_{to_address}_{contract_address}_{category}_{withMetadata}_{exclude_zero_value}_{from_block}_{to_block}_asc"
                # Get rid of the "'" and "[" and "]" in the key
                key = key.replace("'", "").replace("[", "").replace("]", "")

                # Replace ", " with "_" in the key
                key = key.replace(", ", "_")
            else:
                key = file_key

            # Each cached file should be a list of queried response
            cache_response(json.dumps(result), key, "address", "json", "alchemy")
            first_txn_json = result

    # We always want to query the last txn again
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    headers = {
        "content-type": "application/json",
        "accept": "application/json"
    }

    payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{
                "fromBlock": from_block,
                "toBlock": to_block,
                "category": category,
                "withMetadata": withMetadata,
                "excludeZeroValue": exclude_zero_value,
                "maxCount": hex(maxCount),
                "order": "desc"
                }]
            }

    if from_address:
        payload["params"][0]["fromAddress"] = from_address

    if to_address:
        payload["params"][0]["toAddress"] = to_address

    if contract_address:
        payload["params"][0]["contractAddresses"] = contract_address

    if pageKey:
        payload["params"][0]["pageKey"] = pageKey

    # Wrap the request to alchemy in lambda
    # Print the payload in the command line
    request_function = lambda: general_request("POST", url, headers, {}, payload)

    status_code, result = exponential_backoff(request_function, 5)

    if status_code == 200 and result and result.get("result", None) and result["result"].get("transfers", None):
        # Cache the result
        if not file_key:
            key = f"{from_address}_{to_address}_{contract_address}_{category}_{withMetadata}_{exclude_zero_value}_{from_block}_{to_block}_desc"
            # Get rid of the "'" and "[" and "]" in the key
            key = key.replace("'", "").replace("[", "").replace("]", "")

            # Replace ", " with "_" in the key
            key = key.replace(", ", "_")
        else:
            key = file_key

        # Each cached file should be a list of queried response
        cache_response(json.dumps(result), key, "address", "json", "alchemy")
        last_txn_json = result

    first_txn_hash = None
    first_txn_time = None

    last_txn_hash = None
    last_txn_time = None

    if first_txn_json and first_txn_json.get('result') and first_txn_json.get('result').get('transfers'):
        txn = first_txn_json['result']['transfers'][0]
        first_txn_hash = txn['hash']
        first_txn_time = txn['metadata']['blockTimestamp']

    if last_txn_json and last_txn_json.get('result') and last_txn_json.get('result').get('transfers'):
        txn = last_txn_json['result']['transfers'][0]
        last_txn_hash = txn['hash']
        last_txn_time = txn['metadata']['blockTimestamp']

    return [(first_txn_json, first_txn_hash, first_txn_time), (last_txn_json, last_txn_hash, last_txn_time)]



def alchemy_query_transaction_from_address(api_key, from_address="", to_adderss="", maxCount=1000, pageKey="", response_text=None, use_cache=True, random_cache=False, integrate_cache=True, still_query=False, contract_address=[], category=["external", "internal", "erc20", "erc721", "erc1155", "specialnft"], withMetadata=True, exclude_zero_value=False, from_block="0x0", to_block="latest", order="asc", start_time=None, end_time=None, file_key=None):
    """
    Query the transaction related to an address using Alcademy API
    @still_query: if True, will query the API even if the cache is available

    return will be like this:
    [
        one reply from alchemy -> dict,
        another reply from alchemy,
    ]
    """

    # Convert all possible address to lower case
    from_address = from_address.lower()
    to_adderss = to_adderss.lower()
    contract_address = [address.lower() for address in contract_address]
    from_block = from_block.lower()
    to_block = to_block.lower()

    old_from_block = from_block
    old_to_block = to_block

    def filter_by_time(response_json, start_time, end_time):
        # Parse start and end times, if provided
        start_time = parse(start_time).replace(tzinfo=tzutc()) if start_time else None
        end_time = parse(end_time).replace(tzinfo=tzutc()) if end_time else None

        filtered_response_json = []
        
        #do binary search to find start time
        start = 0
        end = len(response_json)
        
        while start < end:
            mid = (start + end) // 2
            if parse(response_json[mid]['result']['transfers'][0]['metadata']['blockTimestamp']) < start_time:
                start = mid+1
            else:
                end = mid
        if start < len(response_json):
            if parse(response_json[mid]['result']['transfers'][0]['metadata']['blockTimestamp']) != start_time:
                start = start - 1
                if start < 0:
                    start = 0
        else:
            start = 0
        
        for index in range(start, len(response_json)):
            # if end start time of response greater than end time stop
            if parse(response_json[index]['result']['transfers'][0]['metadata']['blockTimestamp']) > end_time:
                break
            filtered_transfers = None
            # if start time and end time of alchemy response within bound add it automatically
            if (parse(response_json[index]['result']['transfers'][0]['metadata']['blockTimestamp']) >= start_time and 
                parse(response_json[index]['result']['transfers'][-1]['metadata']['blockTimestamp']) <= end_time):
                filtered_transfers = response_json[index]['result']['transfers']
            else:
                filtered_transfers = [
                    transfer for transfer in response_json[index]['result']['transfers']
                    if (start_time <= parse(transfer['metadata']['blockTimestamp']) if start_time else True) and
                       (parse(transfer['metadata']['blockTimestamp']) <= end_time if end_time else True)
                ]
            
            # If there are any filtered transfers, add them to the item and then to the filtered response
            if filtered_transfers:
                filtered_item = response_json[index].copy()
                filtered_item['result']['transfers'] = filtered_transfers
                filtered_response_json.append(filtered_item)

        return filtered_response_json

    # if both from_address and to_address are empty, raise an error
    if not from_address and not to_adderss:
        raise ValueError("Either from_address or to_address must be provided")

    # if both from_block and to_block are empty, raise an error
    if from_block == 0 and to_block == 0:
        raise ValueError("Either from_block or to_block must be provided")

    loaded_from_cache = False

    response_json = []

    if use_cache:
        # Try to load the cache
        # Compose the file name based on all the parameters
        key = f"{from_address}_{to_adderss}_{contract_address}_{category}_{withMetadata}_{exclude_zero_value}_{old_from_block}_{old_to_block}"
        # Get rid of the "'" and "[]" in the key
        key = key.replace("'", "").replace("[", "").replace("]", "")

        # Replace ", " with "_" in the key
        key = key.replace(", ", "_")

        cache_text = load_cache(key, "address", random_cache, "json", "alchemy", integrate_cache)
        cached_json = json.loads(cache_text) if cache_text else []
        if cached_json:
            response_json.extend(cached_json)
            loaded_from_cache = True

    if (not response_json) or still_query:
        # If we load something from the cache already, try to get the maximum block number
        if loaded_from_cache:
            # We traverse the response_json in the reverse order until we find a blockNum
            for item in response_json[::-1]:
                if item['result']['transfers']:
                    from_block = item['result']['transfers'][-1]['blockNum']
                    from_block = hex(int(from_block, 16) + 1)
                    break

            # if int(response_json[-1]['result']['transfers'][-1]["blockNum"], 16) >= int(from_block, 16):
            #     from_block = response_json[-1]['result']['transfers'][-1]["blockNum"]
                # We want to plus 1
                # from_block = hex(int(from_block, 16) + 1)

        # Before really query alchemy, lets see if the current records are already in the range of the start_time and end_time specifide in parameters
        just_return = False
        if end_time and response_json:
            try:
                maxtime = parse(response_json[-1]['result']['transfers'][-1]['metadata']['blockTimestamp'])
                if maxtime and maxtime > parse(end_time).replace(tzinfo=tzutc()):
                    just_return = True

            except:
                pass

        # If we already have enough records according to the end_time, we will filter the records based on the start time and end time
        if just_return:
            return filter_by_time(response_json, start_time, end_time)

        # Wrap the request to alchemy in lambda
        url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        headers = {
            "content-type": "application/json",
            "accept": "application/json"
        }

        payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "category": category,
                    "withMetadata": withMetadata,
                    "excludeZeroValue": exclude_zero_value,
                    "maxCount": hex(maxCount),
                    "order": order
                    }]
                }

        if from_address:
            payload["params"][0]["fromAddress"] = from_address

        if to_adderss:
            payload["params"][0]["toAddress"] = to_adderss

        if contract_address:
            payload["params"][0]["contractAddresses"] = contract_address

        if pageKey:
            payload["params"][0]["pageKey"] = pageKey

        # Wrap the request to alchemy in lambda
        # Print the payload in the command line
        request_function = lambda: general_request("POST", url, headers, {}, payload)

        status_code, result = exponential_backoff(request_function, 5)

        if status_code == 200 and result and result.get("result", None) and result["result"].get("transfers", None):
            # Cache the result
            if not file_key:
                key = f"{from_address}_{to_adderss}_{contract_address}_{category}_{withMetadata}_{exclude_zero_value}_{old_from_block}_{old_to_block}"
                # Get rid of the "'" and "[" and "]" in the key
                key = key.replace("'", "").replace("[", "").replace("]", "")

                # Replace ", " with "_" in the key
                key = key.replace(", ", "_")
            else:
                key = file_key

            # Each cached file should be a list of queried response
            cache_response(json.dumps(result), key, "address", "json", "alchemy", convert_to_list=True)

            response_json.append(result)

            # Before recursive call, lets see if the current records are already in the range of the start_time and end_time specifide in parameters
            just_return = False
            if end_time and response_json:
                try:
                    maxtime = parse(response_json[-1]['result']['transfers'][-1]['metadata']['blockTimestamp'])
                    if maxtime and maxtime > parse(end_time).replace(tzinfo=tzutc()):
                        just_return = True

                    if maxtime:
                        #print("max time:", maxtime)
                        pass

                except:
                    pass

            # If we already have enough records according to the end_time, we will filter the records based on the start time and end time
            if just_return:
                return filter_by_time(response_json, start_time, end_time)

            else:
                # If the response has a next page, recursively call the function
                if "pageKey" in result["result"].keys() and result["result"]["pageKey"]:
                    try:
                        new_response_json = alchemy_query_transaction_from_address(api_key, from_address, to_adderss, maxCount, result["result"]["pageKey"], None, False, False, False, still_query, contract_address, category, withMetadata, exclude_zero_value, from_block, to_block, order, start_time=start_time, end_time=end_time, file_key=key)
                        response_json.extend(new_response_json)
                    except Exception as e:
                        print(f"Error fetching next page: {e}")
                        pass

        elif status_code != 200:
            raise ValueError("Error fetching txn data from Alcademy:", status_code)

    return response_json

def infura_query_transaction_count(api_key, address):
    '''
    This is a simple function to use infura web3 api to get the transaction count sent from the address. No cache yet
    '''
    w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{api_key}"))

    # Apparently we need to convert the address to checksum address first
    address = w3.to_checksum_address(address)

    return w3.eth.get_transaction_count(address)

def alchemy_query_txn(api_key, txn_hash, use_cache=True, random_cache=True, still_query=False) -> dict:
    loaded_from_cache = False

    response_json = {}

    if use_cache:
        # Try to load the cache
        # Compose the file name based on all the parameters
        key = f"{txn_hash}"

        cache_text = load_cache(key, 'transaction', random_cache, 'json', 'alchemy')
        response_json = json.loads(cache_text) if cache_text else {}
        if response_json:
            loaded_from_cache = True

    if (not response_json) or still_query:
        url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        headers = {
            "content-type": "application/json",
            "accept": "application/json"
        }

        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "trace_transaction",
            "params": [txn_hash]
        }

        # Wrap the request to alchemy in lambda
        request_function = lambda: general_request("POST", url, headers, {}, payload)
        status_code, result = exponential_backoff(request_function, 5)

        if status_code == 200 and result:
            # Cache the result
            # If we are not loading from the cache, the result should be a dict already
            if not loaded_from_cache:
                key = f"{txn_hash}"
                cache_response(json.dumps(result), key, "transaction", "json", "alchemy")

            response_json = result

    if response_json:
        return response_json
    else:
        raise ValueError("Error fetching txn information from Alcademy")

def scrape_is_source_code_available(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Check whether the source code of the sc is available
    """
    address = address.lower()

    # Try to see if theres source code tag
    tags, response_text = scrape_address_tag(address, response_text, proxy_url, proxy_key, use_cache, random_cache, additional_headers)

    if tags:
        for tag in tags:
            if 'source code' in tag.lower():
                return True

    if response_text:
        soup = BeautifulSoup(response_text, 'html.parser')

        # Try to check if we can find the check mark on the page
        if soup.find_all('i', class_='fas fa-check-circle text-success position-absolute'):
            return True

    return False


def scrape_txn_time(txn_hash, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the txn information
    """

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(txn_hash, "txn", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True

    if not response_text:

        url = f"https://etherscan.io/tx/{txn_hash}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/tx/{txn_hash}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, txn_hash, "txn", "html", "etherscan")

        else:
            raise ValueError("Error fetching txn data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    time_element = soup.find('span', id='showUtcLocalDate')

    if time_element:
        timestamp = time_element.get('data-timestamp')
    else:
        timestamp = None

    return timestamp


def scrape_txn_creator(txn_hash, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the txn information
    """

    def get_from_to_element(soup):
        """
        Get the from and to element on the page
        """

        # The transactin page could have one more row called status, which is not always there
        delta = 1

        # Use the length of all div with class 'row' as the upper bound
        upper_bound = len(soup.find_all("div", {"class": "row"}))

        while delta < upper_bound:

            # Iterate each row until we find the 'From:' row
            row_elements = soup.select(f"div.row:nth-child({delta}) > div:nth-child(1)")

            if not row_elements:
                delta += 1
                continue

            row_category = row_elements[0].text.strip()

            if row_category == "From:":
                break

            delta += 1

        # Get the from element
        from_elements = soup.select(f"div.row:nth-child({delta}) > div:nth-child(2) > div:nth-child(1) > span:nth-child(1)")

        # Get the to element
        to_elements = soup.select(f"div.row:nth-child({delta+1}) > div:nth-child(2)")

        return from_elements, to_elements

    txn_info = {
            "creator": None,
            "txn_hash": None,
            "parent_contract": None,
            "contract": None,
            "block": None
            }

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(txn_hash, "txn", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True

    if not response_text:

        url = f"https://etherscan.io/tx/{txn_hash}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/tx/{txn_hash}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, txn_hash, "txn", "html", "etherscan")

        else:
            raise ValueError("Error fetching txn data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    # Get the txn hash
    txn_hash_element = soup.find("span", {"id":"spanTxHash"})

    if txn_hash_element:
        txn_hash = txn_hash_element.text.strip()
        txn_info["txn_hash"] = txn_hash
    else:
        raise ValueError("Error fetching txn hash from Etherscan: {txn_hash}")

    # Get the block number
    blokc_element = soup.select('span.gap-1 > a:nth-child(2)')

    if blokc_element:
        block = blokc_element[0].text.strip()
        txn_info["block"] = block
    else:
        raise ValueError("Error fetching block number from Etherscan: {txn_hash}")

    creator_element, to_element = get_from_to_element(soup)

    if creator_element:
        all_a_elements = creator_element[0].find_all("a")
        # Go through each element until we find the one with href and href composed as /address/
        for element in all_a_elements:
            if element.has_attr("href") and "/address/" in element["href"]:
                # Extract the address from the href
                creator = element["href"].split("/")[-1].strip()
                txn_info["creator"] = creator

    if not txn_info["creator"]:
        raise ValueError("Error fetching txn creator from Etherscan: {txn_hash}")

    if to_element:
        to_element = to_element[0]

        # If there's only one 'div' inside this to_element, it is eoa creating contract direcly
        if len(to_element.find_all("div", recursive=False)) == 1:
            contract = to_element.find_all('div')[0].find('span').find('a').text
            txn_info["contract"] = contract

        elif len(to_element.find_all('div', recursive=False)) == 2:
            # If there are two 'div' inside this to_element, it is contract creating another contract
            # The first 'div' is the parent contract, the second 'div' is the contract created by the parent contract
            parent_contract = to_element.find_all('div')[0].find('span').find('a').text
            contract_ele = to_element.find_all('div')[1].find('a', {'class': 'hash-tag text-truncate'})
            # it could be truncated, read the contractor from the href
            if contract_ele:
                contract = contract_ele.get('href').split('/')[-1]
                txn_info["contract"] = contract
                txn_info["parent_contract"] = parent_contract

        else:
            raise ValueError("Error fetching contract address from Etherscan: {txn_hash}")

    else:
        raise ValueError("Error fetching contract address from Etherscan: {txn_hash}")

    return [txn_info, response_text]

def get_root_nodes(graph):
    return [x for x in graph.nodes if graph.in_degree(x) == 0 and graph.out_degree(x) > 0]

def scrape_contract_creator(address, alchemy_key, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the contract creator address and the txn hash.
    Note that: when the creator is another contract, creator would be the eoa made the transaction to that contract.
    """

    print(f"Getting contract creator for {address}")

    # All convert to lower case
    address = address.lower()

    # Define the return data structure
    creator_info = {
        "creator": None,
        "txn_hash": None,
        "contract": address,
        'timestamp': None
        }

    # Get txn hash given the address
    txn_hash = scrape_contract_creation_txn(address, response_text, proxy_url, proxy_key, use_cache, random_cache, additional_headers=additional_headers)

    creator_info['txn_hash'] = txn_hash

    # Get txn timestmap
    txn_time = scrape_txn_time(creator_info['txn_hash'])

    if txn_time:
        creator_info['timestamp'] = txn_time

    else:
        raise ValueError(f"Error fetching txn timestamp from Etherscan: {address}")

    # Now we have the txn hash, build the graph from the txn
    creation_graph = get_contract_creation_graph_from_txn(alchemy_key, creator_info["txn_hash"])

    # make sure the address is in the graph
    if not creation_graph or address not in creation_graph:
        raise ValueError(f"Error fetching contract creation graph from Alchemy: {address}")

    # We get all edges to the address node and find the edge with edge_type equal to 'create'
    address_in_edges = creation_graph.in_edges(address, data=True)

    # Go through each edge and find the one with edge_type equal to 'create', then get the 'traceAddress' from edge
    traceAddress = None
    for edge in address_in_edges:
        if edge[2]["txn_type"] == "create":
            traceAddress = edge[2]["traceAddress"]

    # Get the root
    root = get_root_nodes(creation_graph)

    assert len(root) == 1
    
    path = []

    # If we cant find create, it is a directyl created sc
    if not traceAddress:
        path.append(root[0])

    else:

        # We need a visited nodes set cuz it is possible to have loop transaction
        visited = set()

        node = root[0]

        while node != address:

            if node in visited:
                continue
            else:
                visited.add(node)

            # Get out edges from the node
            node_out_edges = creation_graph.out_edges(node, data=True)

            # Go through each edge and get all tra
            all_trace_addrs = {}
            for edge in node_out_edges:
                # if traceAddress is empty, we are eterating the out edge of root nodes
                if not edge[2]['traceAddress']:
                    assert edge[0] == root[0]
                    path.append(edge[0])
                    node = edge[1]
                    break

                # Otherwise, we need to make sure that the traceAddress on this edge is the start of the target traceAddress
                txn_trace_addr = edge[2]['traceAddress']
                all_trace_addrs[tuple(txn_trace_addr)] = edge

            # Now we sort the all_trace_addrs with length of each key from long to short and iterate it
            sorted_trace_addrs = sorted(all_trace_addrs.items(), key=lambda x: len(x[0]), reverse=True)
            for each_sort_trace_addr in sorted_trace_addrs:
                # Check if each_sort_trace_addr is the start of traceAddress list
                if traceAddress[:len(each_sort_trace_addr[0])] == list(each_sort_trace_addr[0]):
                    # If it is, we add the edge[0] to the path, update node to edge[1] and update traceAddress
                    path.append(each_sort_trace_addr[1][0])
                    node = each_sort_trace_addr[1][1]
                    break

        path.append(address)

    creator_info["creator"] = path

    # create a stack to do dfs
    return [creator_info, response_text]

def scrape_address_tag(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}, timeout=None):
    """
    Scrape the address tag from the address
    """

    address = address.lower()

    tags = set()

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(address, "address", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True

    if not response_text:

        url = f"https://etherscan.io/address/{address}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/address/{address}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload, timeout=timeout)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, address, "address", "html", "etherscan")

        else:
            raise ValueError("Error fetching data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    # Get the tag
    tag_elements = soup.find_all("span", {"class": "hash-tag text-truncate"})

    # Get the tag badge row
    badge_row = soup.find_all("div", {"class": "d-flex flex-wrap align-items-center gap-1"})

    if badge_row:
        badge_row = badge_row[0]
        badge_elements = badge_row.find_all("span", {"class": "badge"})

        for badge in badge_elements:
            tags.add(badge.text.strip())

    for tag_element in tag_elements:
        if not tag_element.has_attr("data-bs-toggle"):
            tags.add(tag_element.text.strip())

    return [tags, response_text]

def scrape_address_txn_time(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the first txn and last txn time of an account
    """

    first_txn_time = last_txn_time = None

    first_txn, last_txn = scrape_address_txn_range(address, response_text, proxy_url, proxy_key, use_cache, random_cache, additional_headers=additional_headers)

    if first_txn:
        first_txn_time = scrape_txn_time(first_txn, response_text, proxy_url, proxy_key, use_cache, random_cache, additional_headers=additional_headers)

    if last_txn:
        last_txn_time = scrape_txn_time(last_txn, response_text, proxy_url, proxy_key, use_cache, random_cache, additional_headers=additional_headers)

    return first_txn_time, last_txn_time

def scrape_address_txn_range(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the first txn and the last txn of an account
    """

    first_txn = last_txn = None

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(address, "address", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True
            keep_awake()

    if not response_text:

        url = f"https://etherscan.io/address/{address}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)
        need_sleep()

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/address/{address}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, address, "address", "html", 'etherscan')

        else:
            raise ValueError("Error fetching data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    first_txn_h4 = soup.find('h4', string='First Txn Sent')

    last_txn_h4 = soup.find('h4', string='Last Txn Sent')

    if first_txn_h4:
        href_txn = first_txn_h4.find_next_sibling('div').find('a').get('href')
        if href_txn:
            first_txn = href_txn.split('/')[-1]

    if last_txn_h4:
        href_txn = last_txn_h4.find_next_sibling('div').find('a').get('href')
        if href_txn:
            last_txn = href_txn.split('/')[-1]

    return first_txn, last_txn


def scrape_address_type(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the address type tag from the address
    """

    addr_type = None

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(address, "address", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True

    if not response_text:

        url = f"https://etherscan.io/address/{address}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/address/{address}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, address, "address", "html", 'etherscan')

        else:
            raise ValueError("Error fetching data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')
    
    # Get the type whether it is a contract or not
    type_elements = soup.find_all("h1", {"class": "h5 mb-0"})
    for type_element in type_elements:
        if type_element.text.strip() == "Contract":
            addr_type = "Contract"
        else:
            addr_type = "Address"

    if addr_type:
        return [addr_type, response_text]

    else:
        raise ValueError("Error fetching data from Etherscan")

def get_contract_creation_graph_from_txn(alchemy_api, txn_hash, use_cache=True, random_cache=True, still_query=False):
    """
    Get the contract creation graph from the txn
    """

    graph = nx.MultiDiGraph()

    # Get the txn info
    txn_info = alchemy_query_txn(alchemy_api, txn_hash, use_cache, random_cache, still_query)

    if not txn_info.get('result'):
        raise ValueError("Error fetching txn info from Alchemy API")

    for txn in txn_info['result']:
        # Get 'from' and 'to' accounts
        # If the txn is 'suiside', from address will be refund address
        if txn['type'] == 'suicide':
            from_account = txn['action']['refundAddress']
            to_account = txn['action']['address']

        else:
            from_account = txn['action']['from'].lower()
            
            # Attempt to get 'to' address
            to_account = None
            if txn['action'].get('to'):
                to_account = txn['action']['to'].lower()

            elif txn.get('result') and txn.get('result').get('address'):
                to_account = txn['result']['address'].lower()
            else:
                # We need manual intervention here
                # its possible that the contract creation failed due to the lack of gas
                continue

        graph.add_node(from_account)
        graph.add_node(to_account)
        graph.add_edge(from_account, to_account, txn=txn, txn_type=txn['type'], traceAddress=txn['traceAddress'])

    return graph

def scrape_contract_creation_txn(address, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the contract creation txn
    """

    # Convert address to lower case
    address = address.lower()
    
    loaded_from_cache = False

    txn_hash = None

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(address, "address", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True
            keep_awake()

    if not response_text:

        url = f"https://etherscan.io/address/{address}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)
        need_sleep()

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/address/{address}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, address, "address", "html", "etherscan")

        else:
            raise ValueError("Error fetching data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    # Get the txn hash that creates this contract
    txn_hash_elements = soup.find_all("a", {"class": "hash-tag text-truncate", "data-bs-toggle": "tooltip"})

    found_txn_hash = False

    if txn_hash_elements:
        # Go through each a element until we find the one with href with /tx/ in it
        for txn_hash_element in txn_hash_elements:
            if "/tx/" in txn_hash_element.get("href"):
                # get the txn hash from the href
                txn_hash = txn_hash_element.get("href").split("/")[-1]
                found_txn_hash = True

    if not found_txn_hash:
        # Raise the exception that it cannot find the txn
        raise ValueError(f"Unable to find the txn hash that creates this contract {address}")

    return txn_hash

def pull_contract_from_txn(alchemy_api, txn_hash):
    # Get the creation graph first
    creation_graph = get_contract_creation_graph_from_txn(alchemy_api, txn_hash)

    # result should be a map from the deployed contract address to the tuple with init_code and resulted code
    res = {}

    # Go through each edig in the graph to find the onew ith txn_type='create'
    for edge in creation_graph.edges(data=True):
        if edge[2].get('txn_type') == 'create':
            txn = edge[2].get('txn')

            # Get the created contract address
            contract = edge[1]

            # Get the init codo from the txn
            init_code = txn['action']['init']

            # Get the deployed code from txn
            result_code = txn['result']['code']

            res[contract.lower()] = [init_code, result_code]

    return res

def pull_contract_related_code(contract_address, alchemy_api, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Pull the creation code and the result code given a smart contract address
    """
    # We could get the creation graph already, which will take care of (1) find txn and (2) pull the code used in txn

    # Get the txn for etherscan
    txn_hash = scrape_contract_creation_txn(contract_address, response_text=None, proxy_url=proxy_url, proxy_key=proxy_key, use_cache=use_cache, random_cache=random_cache, additional_headers=additional_headers)

    # Get the created contract information from the txn
    create_contract_info = pull_contract_from_txn(alchemy_api, txn_hash)

    if contract_address.lower() in create_contract_info:
        return create_contract_info[contract_address.lower()]
    else:
        # If we cant find any code, raise an exception
        raise ValueError(f"Unable to find the code for contract {contract_address}")

def get_deployed_sc_from_txn(alchemy_api, txn_hash):
    # Get all deployed SC address given txn_hash
    # Get the creation graph given the txn_hash
    creation_graph = get_contract_creation_graph_from_txn(alchemy_api, txn_hash)

    deployed_sc = set()

    # Go through each edge
    # put the edge[2] into deployed_sc if the edge has txn_type equial to 'create'
    for edge in creation_graph.edges(data=True):
        if edge[2].get('txn_type') == 'create':
            deployed_sc.add(edge[1])

    return list(deployed_sc)

#use breadth first search to create graph
def graph_create(apikey, address, start_time = None, end_time = None, dropNone=False, use_cache=True, overwrite=False):

    address = address.lower()

    G = nx.MultiDiGraph()

    # Before doing anything, check if we have the cache of the graph yet
    if use_cache:
        # Target cached graph path
        key = Path(cache_path) / "scf" / "txn_graph" / f"{address}.gpickle"

        if key.exists():
            G = nx.read_gml(str(key.absolute()))

            if not overwrite:
                return G
    #queue of addresses
    q = []
    q.append(address)
    if not dropNone:
        G.add_node("None")
    visited = set()
    while q:
        size = len(q)
        for i in range(size):
            addr = q.pop(0)
            if addr in visited:
                continue
            visited.add(addr)
            print(f"Visited: {len(visited)}  Quering: {addr}")
            queries = alchemy_query_transaction_from_address(apikey, 
                                             from_address= addr,
                                             start_time = start_time,
                                             end_time = end_time, 
                                             still_query = True)
            for query in queries:
                for transfer in query['result']['transfers']:
                    # Get the txn hash
                    txn_hash = transfer['hash']
                    # Get the contract_creation_graph
                    contract_creation_graph = get_contract_creation_graph_from_txn(apikey, txn_hash)
                    # Go throug each edge in contract_creation_graph
                    for edge in contract_creation_graph.edges(data=True):
                        from_address = edge[0]
                        to_address = edge[1]

                        if not from_address in G:
                            G.add_node(from_address)

                        if not to_address in G:
                            G.add_node(to_address)

                        if not from_address in visited:
                            q.append(from_address)

                        if not to_address in visited:
                            q.append(to_address)

                        G.add_edge(from_address, to_address, transfer=transfer, value = transfer['value'], asset = transfer['asset'], txn = edge[2].get('txn'), txn_type = edge[2].get('txn_type'))
    
    # Now we want to cache this graph so that we dnt need to all these trackings again
    key = Path(cache_path) / "scf" / "txn_graph" / f"{address}.gpickle"

    nx.write_gml(G, str(key.absolute()))
    return G

#use breadth first search to get all reachable addresses
def extract_subgraph(graph, init_node):
    #queue of addresses
    q = []
    q.append(init_node)
    visited = set()
    while q:
        size = len(q)
        for i in range(size):
            addr = q.pop(0)
            if addr in visited:
                continue
            visited.add(addr)
            print(addr)
            for node in graph.successors(addr):
                if node not in visited:
                    q.append(node)
    #create subgraph by considering the visited set of nodes using networkx
    G = graph.subgraph(visited)
    return G

def scrape_token_from_txn(txn_hash, response_text=None, proxy_url=None, proxy_key=None, use_cache=True, random_cache=True, additional_headers={}):
    """
    Scrape the txn information
    """

    def get_erc20_element(soup):
        """
        Get the from and to element on the page
        """

        # The transactin page could have one more row called status, which is not always there
        delta = 1

        # Use the length of all div with class 'row' as the upper bound
        upper_bound = len(soup.find_all("div", {"class": "row"}))

        while delta < upper_bound:

            # Iterate each row until we find the 'From:' row
            row_elements = soup.select(f"div.row:nth-child({delta}) > div:nth-child(1)")

            if not row_elements:
                delta += 1
                continue

            row_category = row_elements[0].text.strip()

            if row_category.startswith("ERC-20 Tokens Transferred:"):
                break

            delta += 1
        
        # Get the ERC-20 Tokens Transferred:
        erc20_element = soup.select(f"div.row:nth-child({delta}) > div:nth-child(2)")

        return erc20_element

    txn_info = {
            "erc20": [],
            "input": 
                {
                    "function": None,
                    "method": None,
                    "params": None
                }
            }

    loaded_from_cache = False

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(txn_hash, "txn", random_cache, "html", "etherscan")
        if response_text:
            loaded_from_cache = True

    if not response_text:

        url = f"https://etherscan.io/tx/{txn_hash}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": f"https://etherscan.io/tx/{txn_hash}"}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text

            # cache the response if the response is not from cache
            if not loaded_from_cache:
                cache_response(response_text, txn_hash, "txn", "html", "etherscan")

        else:
            raise ValueError("Error fetching txn data from Etherscan:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')

    erc20_element = get_erc20_element(soup)
    
    input_element = soup.find("textarea", {"id":"inputdata"})
    
    if input_element:
        try:
            element = [i for i in input_element.text.replace("\r", "").split("\n") if i != '']
            txn_info["input"]["function"] = element[0].split(":")[1].strip()
            txn_info["input"]["method"] = element[1].split(":")[1].strip()
            params = []
            for i in range(2, len(element)):
                params.append(element[i].split(":")[1].strip())
            txn_info["input"]["params"] = params
        except:
            txn_info["input"] = input_element.text
    
    if erc20_element:
        all_div_elements = erc20_element[0].find_all("div", {"class": "row-count d-flex align-items-baseline"})
        for div in all_div_elements:
            a_elements = div.find_all("a")
            temp_dict = {"from": None, "to": None, "for": None, "token": None}
            temp_dict["from"] = a_elements[0]["href"].split("=")[1]
            temp_dict["to"] = a_elements[1]["href"].split("=")[1]
            temp_dict["for"] = div.find_all("span", {"class": "me-1"})[-1].text
            temp_dict["token"] = a_elements[2]["href"].split("/")[2]
            txn_info["erc20"].append(temp_dict)

    return [txn_info, response_text]

def load_csv(path):
    exchange = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                time = row[1].split(' ')
                time = time[0] + 'T' + time[1].split('+')[0] + '.000Z'
                row[1] = parse(time)
            except:
                pass
            exchange.append(row)
    return exchange
    
def add_usd_value(graph, exchange):
    def binsearch(exchange, datetime):
        i = 0
        j = len(exchange)
        while i < j:
            mid = (i + j) // 2
            if exchange[mid][1] < datetime:
                i = mid + 1
            else:
                j = mid
        return i
    for edge in graph.edges:
        index = binsearch(exchange, parse(graph.edges[edge]['data']['metadata']['blockTimestamp']))
        graph.edges[edge]['usd_value'] = float(exchange[index][-2]) * graph.edges[edge]['data']['value']

## API for Postgresql Connection
#####IMPORTANT: remember to close connection when finished
#####IMPORTANT: remember to make a new connection for each new process in Multiprocessing
################postgres connections cannot be pickled so make a wrapper function

def query_from_database(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    ret = cursor.fetchall()
    return ret

def query_transactions(conn, from_address="", to_address=""):
    if not from_address and not to_address:
        return []
    query = ("SELECT hash, nonce, transaction_index, from_address, to_address, value, "+
                 "gas, gas_price, input text, receipt_cumulative_gas_used, receipt_gas_used, "+
                 "receipt_contract_address, receipt_root, receipt_status, block_timestamp, "+
                 "block_number, block_hash, max_fee_per_gas, max_priority_fee_per_gas, "+ 
                 "transaction_type, receipt_effective_gas_price FROM transactions WHERE ")
    if from_address and to_address:
        from_address = from_address.lower()
        to_address = to_address.lower()
        query += "from_address = '" + from_address + "' AND to_address = '" + to_address + "'"
    if from_address:
        from_address = from_address.lower()
        query += "from_address = '" + from_address + "'"
    if to_address:
        to_address = to_address.lower()
        query += "to_address = '" + to_address + "'"
    ret = query_from_database(conn, query)
    return ret

def query_traces(conn, from_address="", to_address=""):
    if not from_address and not to_address:
        return []
    query = ("SELECT transaction_hash, transaction_index, from_address, to_address, value, input, "+
             "output, trace_type, call_type, reward_type, gas, gas_used, subtraces, trace_address, "+
             "error, status, block_timestamp, block_number, block_hash, trace_id FROM traces WHERE ")
    if from_address and to_address:
        from_address = from_address.lower()
        to_address = to_address.lower()
        query += "from_address = '" + from_address + "' AND to_address = '" + to_address + "'"
    if from_address:
        from_address = from_address.lower()
        query += "from_address = '" + from_address + "'"
    if to_address:
        to_address = to_address.lower()
        query += "to_address = '" + to_address + "'"
    ret = query_from_database(conn, query)
    return ret

def get_transactions_from_txn_hash(conn, txn_hash=""):
    if not txn_hash:
        return []
    txn_hash = txn_hash.lower()
    query = ("SELECT hash, nonce, transaction_index, from_address, to_address, value, "+
             "gas, gas_price, input text, receipt_cumulative_gas_used, receipt_gas_used, "+
             "receipt_contract_address, receipt_root, receipt_status, block_timestamp, "+
             "block_number, block_hash, max_fee_per_gas, max_priority_fee_per_gas, "+ 
             "transaction_type, receipt_effective_gas_price FROM transactions WHERE "+ 
             "hash = '" + txn_hash + "'")
    ret = query_from_database(conn, query)
    return ret

def get_traces_from_txn_hash(conn, txn_hash=""):
    if not txn_hash:
        return []
    txn_hash = txn_hash.lower()
    query = ("SELECT transaction_hash, transaction_index, from_address, to_address, value, input, "+
             "output, trace_type, call_type, reward_type, gas, gas_used, subtraces, trace_address, "+
             "error, status, block_timestamp, block_number, block_hash, trace_id FROM traces WHERE "+\
             "transaction_hash = '" + txn_hash + "'")
    ret = query_from_database(conn, query)
    return ret

def get_contract_from_addr(conn, address=""):
    if not address:
        return []
    address = address.lower()
    query = ("SELECT address, bytecode, function_sighashes, is_erc20, is_erc721, block_timestamp, "+
             "block_number, block_hash from contracts WHERE address = '" + address + "'")
    ret = query_from_database(conn, query)
    return ret

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

def get_source_tag(addr, proxy_url=None,proxy_key=None, additional_headers= {}):
    addr = addr.lower()
    tags = scrape_address_tag(addr, 
                              proxy_url=proxy_url,
                              proxy_key=proxy_key,
                              additional_headers= {'dynamic': 'false'})[0]
    source = scrape_is_source_code_available(addr, 
                                             proxy_url=proxy_url,
                                             proxy_key=proxy_key,
                                             additional_headers= {'dynamic': 'false'})
    ret = (tags, source)
    return ret

def get_all_traces(conn_auth, addr):
    file_path = Path(cache_path) / "sql_result_cache" / "traces" / str(addr) / f"{addr}.pkl"
    if file_path.exists():
        ret = None
        with open(str(file_path), "rb") as f:
            ret = pickle.load(f)
        return ret
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(database=conn_auth[0],
                            user=conn_auth[1],
                            password=conn_auth[2],
                            host=conn_auth[3],
                            port=conn_auth[4])
    cursor = conn.cursor()
    sql = f"SELECT hash FROM transactions WHERE from_address = '{addr}'"
    cursor.execute(sql)
    result = cursor.fetchall()
    all_traces = []
    for res in result:
        all_traces += get_traces_from_txn_hash(conn, txn_hash=res[0])
    conn.close()
    with open(str(file_path), "wb") as f:
        pickle.dump(all_traces, f)
    
    return all_traces

#use breadth first search to create graph
def graph_traces_multi(conn_auth, q, q_lock, visited, v_lock, cond, num_fail, num_success, edge_list, proxy_url = None, 
                proxy_key = None, end_level = None, start_time = None, end_time = None):
    
    running = True
    try:
        while cond.value != 0:
            q_lock.acquire()
            if not q:
                if running:
                    running = False
                    cond.value -= 1
                q_lock.release()
                time.sleep(0.5)
                continue
            addr, layer = q.pop(0)
            q_lock.release()
            if not running:
                running = True
                cond.value += 1
            try:
                tags, source = get_source_tag(addr,
                                          proxy_url=proxy_url,
                                          proxy_key=proxy_key,
                                          additional_headers= {'dynamic': 'false'})
                num_success.value += 1
            except Exception as e:
                num_fail.value += 1
                continue
            stop = source if len(tags) == 0 else True
            for tag in tags:
                if 'hack' in tag.lower() or 'phish' in tag.lower():
                    stop = False
                    break
            try:
                v_lock.acquire()
                visited[addr] = (tags, source)
                v_lock.release()
                if stop:
                    #print(f"Skip Quering: {addr}", flush=True)
                    continue
                queries = get_all_traces(conn_auth, addr)
            except Exception as e:
                continue
            if not queries:
                continue
            for edge in queries:
                from_address = edge[2]
                to_address = edge[3]

                v_lock.acquire()
                from_addr_in_visited = from_address in visited
                to_addr_in_visited = (not to_address) or (to_address in visited)
                if not from_addr_in_visited:
                    visited[from_address] = 0
                if not to_addr_in_visited:
                    visited[to_address] = 0
                v_lock.release()
                
                if not end_level or (layer + 1 < end_level):
                    if not from_addr_in_visited:
                        q_lock.acquire()
                        q.append((from_address, layer + 1))
                        q_lock.release()

                    if not to_addr_in_visited:
                        q_lock.acquire()
                        q.append((to_address, layer + 1))
                        q_lock.release()

                edge_list.append(edge)
    except:
        if running:
            cond.value -= 1

#use breadth first search to create graph
def graph_create_traces_multithread(conn_auth, address, proxy_url = None, proxy_key = None, num_thread = 10, 
                             end_level=None, end_time=None):
    
    def combine_graph(multi_thread_ret, visited):
        G = nx.MultiDiGraph()
        for edge in multi_thread_ret:
            G.add_edge(edge[2] if edge[2] else "None", edge[3] if edge[3] else "None", txn=edge)
        for key in visited:
            if key in G.nodes:
                G.nodes[key]['tags'] = visited[key]
        return G
    address = address.lower()
    file_path = Path(cache_path) / "sql_result_cache" / "trace_graph" / str(address) / (str(end_level) if end_level else "")
    visited_path = file_path / f"{address}_visited.pkl"
    edgelist_path = file_path / f"{address}_edgeList.pkl"
    graph_path = file_path / f"{address}_graph.pkl"
    
    if graph_path.exists():
        Graph = None
        with open(str(graph_path), "rb") as f:
            Graph = pickle.load(f)
        return Graph
    
    visited_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        manager = mp.Manager()
    except:
        manager= None
    q = manager.list()
    q_lock = manager.Lock()
    visited = manager.dict()
    v_lock = manager.Lock()
    cond = mp.Value('i', num_thread)
    q.append((address, 0))
    visited[address] = 0
    edge_list = manager.list()
    num_fail = mp.Value('i', 0)
    num_success = mp.Value('i', 0)

    process = [mp.Process(target=graph_traces_multi, 
                          args=(conn_auth, q, q_lock, visited, v_lock, cond, num_fail, num_success, edge_list, 
                                proxy_url, proxy_key, end_level)) for i in range(num_thread)]
    for p in process:
        p.start()
    start_time = time.time()
    while cond.value > 0 and (not end_time or (time.time() - start_time < end_time * 60)):
        print("length of queue: ", len(q), " length of visited: ", len(visited)," cond: ", 
              cond.value, " num_fail: ", num_fail.value , "num_succ: ", num_success.value, flush=True)
        with open(str(visited_path), 'wb') as f:
            pickle.dump(visited.copy(), f)
        with open(str(edgelist_path), 'wb') as f:
            pickle.dump(list(edge_list), f)
        time.sleep(60)
    if end_time:
        for p in process:
            try:
                p.terminate()
            except Exception as e:
                pass
    for p in process:
        p.join()
    with open(str(visited_path), 'wb') as f:
        pickle.dump(visited.copy(), f)
    with open(str(edgelist_path), 'wb') as f:
        pickle.dump(list(edge_list), f)
    G = combine_graph(list(edge_list), visited.copy())
    with open(str(graph_path), "wb") as f:
        pickle.dump(G, f)
    return G

#use breadth first search to create graph
def graph_transactions_multi(conn_auth, q, q_lock, visited, v_lock, cond, edge_list, proxy_url = None, 
                proxy_key = None, end_level = None, start_time = None, end_time = None):
    
    def sample_elements(lst, num_elements):
        if len(lst) <= num_elements:
            return lst
        else:
            step = len(lst) / num_elements
            indices = np.round(np.arange(0, len(lst), step))
            return [lst[int(idx)] for idx in indices]
    
    running = True
    try:
        while cond.value != 0:
            q_lock.acquire()
            if not q:
                if running:
                    running = False
                    cond.value -= 1
                q_lock.release()
                time.sleep(0.5)
                continue;
            addr, layer = q.pop(0)
            q_lock.release()
            if not running:
                running = True
                cond.value += 1
            try:
                tags, source = get_source_tag(addr,
                                          proxy_url=proxy_url,
                                          proxy_key=proxy_key,
                                          additional_headers= {'dynamic': 'false'})
            except Exception as e:
                continue
            stop = source if len(tags) == 0 else True
            for tag in tags:
                if 'hack' in tag.lower() or 'phish' in tag.lower():
                    stop = False
                    break
            try:
                v_lock.acquire()
                visited[addr] = (tags, source)
                v_lock.release()
                if stop:
                    #print(f"Skip Quering: {addr}", flush=True)
                    continue;
                conn = psycopg2.connect(database=conn_auth[0],
                            user=conn_auth[1],
                            password=conn_auth[2],
                            host=conn_auth[3],
                            port=conn_auth[4])
                queries = query_transactions(conn, from_address = addr)
                conn.close()
            except Exception as e:
                continue
            if not queries:
                continue
            for edge in queries:
                from_address = edge[3]
                to_address = edge[4]

                v_lock.acquire()
                from_addr_in_visited = from_address in visited
                to_addr_in_visited = (not to_address) or (to_address in visited)
                if not from_addr_in_visited:
                    visited[from_address] = 0
                if not to_addr_in_visited:
                    visited[to_address] = 0
                v_lock.release()
                
                if not end_level or (layer + 1 < end_level):
                    if not from_addr_in_visited:
                        q_lock.acquire()
                        q.append((from_address, layer + 1))
                        q_lock.release()

                    if not to_addr_in_visited:
                        q_lock.acquire()
                        q.append((to_address, layer + 1))
                        q_lock.release()

                edge_list.append(edge)
    except:
        if running:
            cond.value -= 1

#use breadth first search to create graph
def graph_create_transactions_multithread(conn_auth, address, proxy_url = None, proxy_key = None, num_thread = 10, 
                             end_level=None, end_time=None):
    
    def combine_graph(multi_thread_ret, visited):
        G = nx.MultiDiGraph()
        for edge in multi_thread_ret:
            G.add_edge(edge[3], edge[4] if edge[4] else "None", txn=edge)
        for key in visited:
            G.nodes[key]['tags'] = visited[key]
        return G
    address = address.lower()
    file_path = Path(cache_path) / "sql_result_cache" / "transaction_graph" / str(address) / (str(end_level) if end_level else "")
    visited_path = file_path / f"{address}_visited.pkl"
    edgelist_path = file_path / f"{address}_edgeList.pkl"
    graph_path = file_path / f"{address}_graph.pkl"
    
    if graph_path.exists():
        Graph = None
        with open(str(graph_path), "rb") as f:
            Graph = pickle.load(f)
        return Graph
    
    visited_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        manager = mp.Manager()
    except:
        manager= None
    q = manager.list()
    q_lock = manager.Lock()
    visited = manager.dict()
    v_lock = manager.Lock()
    cond = mp.Value('i', num_thread)
    q.append((address, 0))
    visited[address] = 0
    edge_list = manager.list()

    process = [mp.Process(target=graph_transactions_multi, 
                          args=(conn_auth, q, q_lock, visited, v_lock, cond, edge_list, proxy_url, 
                                proxy_key, end_level)) for i in range(num_thread)]
    for p in process:
        p.start()
    start_time = time.time()
    while cond.value > 0 and (not end_time or (time.time() - start_time < end_time * 60)):
        print("length of queue: ", len(q), " length of visited: ", len(visited)," cond: ", cond.value, flush=True)
        with open(str(visited_path), 'wb') as f:
            pickle.dump(visited.copy(), f)
        with open(str(edgelist_path), 'wb') as f:
            pickle.dump(list(edge_list), f)
        time.sleep(60)
    if end_time:
        for p in process:
            try:
                p.terminate()
            except Exception as e:
                pass
    for p in process:
        p.join()
    with open(str(visited_path), 'wb') as f:
        pickle.dump(visited.copy(), f)
    with open(str(edgelist_path), 'wb') as f:
        pickle.dump(list(edge_list), f)
    G = combine_graph(list(edge_list), visited.copy())
    with open(str(graph_path), "wb") as f:
        pickle.dump(G, f)
    return G

#find all contract creation paths given an account
def account_contract_create_paths(conn_auth, address):
    def prev(trace_addr):
        if trace_addr is None:
            return ''
        last_ind = trace_addr.rfind(',')
        if last_ind == -1:
            return ''
        return trace_addr[:last_ind]
    def get_paths(txn_traces):
        trace_dict = dict()
        create = []
        for i in txn_traces:
            index = i[13] if i[13] is not None else ''
            trace_dict[index] = i
            if i[7] == 'create':
                create.append(i)
        ret = []
        for trace in create:
            seq = []
            seq.append(trace)
            ind = trace[13] if trace[13] is not None else ''
            while ind:
                ind = prev(ind)
                seq.append(trace_dict[ind])
            seq = seq[::-1]
            ret.append(seq)
        return ret
    def get_contracts(conn_auth, addrs):
        conn = psycopg2.connect(database=conn_auth[0],
                            user=conn_auth[1],
                            password=conn_auth[2],
                            host=conn_auth[3],
                            port=conn_auth[4])
        ret = dict()
        for addr in addrs:
            res = get_contract_from_addr(conn, address=addr)
            if res:
                ret[addr] = res[0]
        conn.close()
        return ret
            
    addr_traces = get_all_traces(conn_auth, address)
    txn_dict = dict()
    for traces in addr_traces:
        if traces[0] not in txn_dict:
            txn_dict[traces[0]] = []
        txn_dict[traces[0]].append(traces)
    paths = []
    addr_set = set()
    for txn in txn_dict:
        curr = get_paths(txn_dict[txn])
        for path in curr:
            temp = []
            for trace in path:
                temp.append(trace[2])
            temp.append(path[-1][3])
            for i in temp:
                addr_set.add(i)
            paths.append(temp)       
    paths.sort(key=lambda x: len(x))
    contract_dict = get_contracts(conn_auth, addr_set)
    addr_set = set()
    addr_set.add(address)
    self_created_paths = []
    external_paths = paths
    while True:
        size = len(external_paths)
        i = 0
        while i < len(external_paths):
            path = external_paths[i]
            isvalid = True
            for addr in path[:-1]:
                if addr not in addr_set:
                    isvalid = False
                    break
            if isvalid:
                addr_set.add(path[-1])
                self_created_paths.append(path)
                external_paths.pop(i)
            else:
                i += 1
        if size == len(external_paths):
            break
    return self_created_paths, external_paths, contract_dict

#find the account that created the contract
def get_contract_creator(conn_auth, contract):
    conn = psycopg2.connect(database=conn_auth[0],
                            user=conn_auth[1],
                            password=conn_auth[2],
                            host=conn_auth[3],
                            port=conn_auth[4])
    contract = contract.lower()
    traces = query_traces(conn, to_address=contract)
    txn_hash = None
    for trace in traces:
        if trace[7] == 'create':
            txn_hash = trace[0]
            break
    txn = get_transactions_from_txn_hash(conn, txn_hash=txn_hash)
    conn.close()
    return txn[0][3]

#scrape decompiled contract address bytecode from dedaub's web decompiler
def scrape_decompile(contract_addr, proxy_url=None, proxy_key=None, get_source=False,
                     use_cache=True, random_cache=True, additional_headers={}):
    
    loaded_from_cache = False
    
    dtype = "address" + "_source" if get_source else "_decompiled"

    if use_cache:
        # Try to load the cache first
        response_text = load_cache(contract_addr, dtype, random_cache, "html", "dedaub")
        if response_text:
            loaded_from_cache = True

    if not response_text:
        
        if get_source:
            end = "source"
        else:
            end = "decompiled"

        url = f"https://library.dedaub.com/ethereum/address/{contract_addr}/{end}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
        response = requests.get(url, headers=headers)

        # If the response is not 200, gives proxy a try
        if response.status_code != 200 and proxy_url and proxy_key:
            payload = {"api_key": proxy_key, "url": url}

            # We go through the additional headers and put it in payload
            for key, value in additional_headers.items():
                payload[key] = value

            response = requests.get(proxy_url, params=payload)

        if response.status_code == 200:
            response_text = response.text
        else:
            raise ValueError("Error fetching txn data from Dedaub Library:", response.status_code)

    soup = BeautifulSoup(response_text, 'html.parser')
    
    content = json.loads(soup.find(id="__NEXT_DATA__").contents[0])
    
    has_source = content['props']['pageProps']['contractPayload']['accountMetadata']['hasSource']
    
    if not has_source and get_source and not loaded_from_cache:
        return scrape_decompile(contract_addr, proxy_url, proxy_key, has_source,
                                use_cache, random_cache, additional_headers)
    
    if get_source:
        ret = ""
        data = content['props']['pageProps']['contractPayload']['source']['value']
        for i in data:
            ret += i['content'].replace("\\n", "\n")
    else:
        ret = content['props']['pageProps']['contractPayload']['decompiled'].replace("\\n", "\n")
    
    if not loaded_from_cache:
        cache_response(response_text, contract_addr, dtype, "html", "dedaub")
    return ret