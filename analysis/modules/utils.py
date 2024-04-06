import sc as sc
import Account
import Txn
import networkx as nx
import os
from tqdm import tqdm
from multiprocessing import Pool
import call_targets_analysis as cta
from pathlib import Path
import json

def analyze_call_targets_from_contract(contract_info):
    contract = contract_info[0]
    init_code = contract_info[1][0]
    runtime_code = contract_info[1][1]
    call_targets = {}

    panoramix_call_targets = cta.panoramix_get_all_call_targets_from_hex_code(runtime_code)
    mythril_call_targets = cta.mythril_get_direct_call_targets_from_hex_code(runtime_code)

    call_targets['panoramix'] = panoramix_call_targets
    call_targets['mythril'] = mythril_call_targets

    return [contract, call_targets]

def trace_creation(smart_contract, cache_dir, error_path):
    """
    param smart_contract: The address of the smart contract to trace the creation chain
    param cache_dir: The directory to store the cache, used by sc
    param error_path: The path to store the error log (append)
    """

    # Set the cache path
    sc.set_cache_path(cache_dir)

    # dict to save the results
    results = {}

    # Use sc to get the creation chain
    try:
        creator_info, _ = sc.scrape_contract_creator(smart_contract)

        # get the creation time stamp
        creation_time = creator_info['timestamp']
        results['creation_time'] = creation_time

        # Get the length of the deployment chain
        deployment_chain_length = len(creator_info['creator'])
        results['deployment_chain_length'] = deployment_chain_length

        # Get the original eoa creator
        original_eoa_creator = creator_info['creator'][0]
        # We want to make sure that the original creator is an EOA
        if sc.scrape_address_type(original_eoa_creator)[0] != 'Address':
            raise Exception(f'The original creator is not an EOA {smart_contract}')

        # Now we query the original creator information
        first_txn_date, last_txn_date = sc.scrape_address_txn_time(original_eoa_creator)
        results['time_eoa_creation'] = first_txn_date
        results['time_eoa_last_txn'] = last_txn_date

        # Given EOA, we track all txns from the start to the end and find the ones used to create contract
        all_transactions = sc.alchemy_query_transaction_from_address(os.environ['ALCHEMY'], original_eoa_creator, end_time="2023-05-01", still_query=True)

        # Now we process all transactions to find the ones used to create the contract
        # Save the information from txn to the created contract with code
        txn_to_created_sc = {}

        for query in tqdm(all_transactions, desc='Processing transactions'):
            # Make sure that query has ['result']['transfers'] list
            if not 'result' in query or not 'transfers' in query['result']:
                continue

            # Now go through each txn
            for txn in tqdm(query['result']['transfers'], desc='Processing txns'):
                # Pull the created contract information from the alchemy
                created_contract_info = sc.pull_contract_from_txn(os.environ['ALCHEMY'], txn['hash'])

                # Iterate through this dict and update direct_created_sc
                for contract, code in created_contract_info.items():
                    if not txn['hash'] in txn_to_created_sc:
                        txn_to_created_sc[txn['hash']] = {}

                    txn_to_created_sc[txn['hash']][contract] = code

        return txn_to_created_sc

    except Exception as e:
        # save the smart contract address and exception to the error log
        with open(error_path, 'a') as f:
            f.write(smart_contract + '\t' + str(e) + '\n')

def perform_temporal_call_target_analysis_given_traced_transaction(trace_transaction):

    # read the cache path from os.environ
    cta.set_cache_path(os.environ['CACHE_PATH'])

    
    # result path
    result_path = Path(__file__).parent.parent / 'preliminary' / 'result' / 'call_targets.json'

    call_targets_result = {}

    # only anylze when tnhe result file does not exist
    if result_path.is_file():
        # Load the result back into the call_targets_result
        with open(result_path, 'r') as f:
            call_targets_result = json.load(f)

    to_analyze_contract = []

    for txn, created_contract_info in trace_transaction.items():
        for contract, code in created_contract_info.items():
            if not contract in call_targets_result:
                # code here is a tuple of (init_code, runtime_code)
                to_analyze_contract.append((contract, code))

    with Pool(processes=40) as pool:
        for contract, call_targets in tqdm(pool.imap_unordered(analyze_call_targets_from_contract,to_analyze_contract), total=len(to_analyze_contract)):
            # Every time we get a result, we save it to the result file
            call_targets_result[contract] = call_targets
            with open(result_path, 'w') as f:
                json.dump(call_targets_result, f)

def perform_temporal_call_target_analysis_given_contract(smart_contract, cache_dir, error_path):
    """
    Track back to the creator of the smart contract and get all deployed contracts
    Then perform call target analysis on each contract
    """

    # Get temporal txns with code in it
    temporal_txns = trace_creation(smart_contract, cache_dir, error_path)

    perform_temporal_call_target_analysis_given_traced_transaction(temporal_txns)
