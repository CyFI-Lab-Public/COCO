from pathlib import Path
# import analysis module in the modules folder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modules import call_targets_analysis as cta
import json
from tqdm import tqdm
import mingpy.ming_sc as sc
from multiprocessing import Pool

# read cache path from system environment
import os
cache_path = os.environ.get('CACHE_PATH')

def analyze_call_targets_from_contract(contract):
    return [contract, cta.get_all_call_targets_from_contract(contract)]


def main():
    # Open the contract_creator.json in preliminary/result folder

    # Open the json file with a list of contract addresses to analyze
    to_analyze_contracts = []
    with open(Path(__file__).parent.parent / 'preliminary' / 'result' / 'contract_creator.json', 'r') as f:
        to_analyze_contracts = list(json.load(f).keys())

    cta.set_cache_path("cache_path")

    # result path
    result_path = Path(__file__).parent.parent / 'preliminary' / 'result' / 'call_targets_initial_dataset.json'

    call_targets_result = {}

    # only anylze when tnhe result file does not exist
    if result_path.is_file():
        # Load the result back into the call_targets_result
        with open(result_path, 'r') as f:
            call_targets_result = json.load(f)

    with Pool(processes=40) as pool:

        if not '--rewrite' in sys.argv:
            # Only check the contracts which are not in the result file
            to_analyze_contracts = [contract for contract in to_analyze_contracts if contract not in call_targets_result]

        for contract, call_targets in tqdm(pool.imap_unordered(analyze_call_targets_from_contract, to_analyze_contracts), total=len(to_analyze_contracts)):
            # Every time we get a result, we save it to the result file
            call_targets_result[contract] = call_targets
            with open(result_path, 'w') as f:
                json.dump(call_targets_result, f)


    # write the result to the result path
    # with open(result_path, 'w') as f:
    #     json.dump(call_targets_result, f, indent=4)

    # Contract with hardeded coded result. The following two could have overlapping
    hard_coded_targets = {}
    non_hard_coded_targets = {}

    # Contract with more than one targets no matter it is hardcoded or not
    more_than_one_all_targets = set()
    more_than_one_hardcoded_targets = set()

    # Contract with no targets
    no_targets = set()

    # Time to process the call targets results
    for contract, targets_info in call_targets_result.items():
        # if the contract has no targets, add it to the no_targets set
        if len(targets_info['panoramix']['panoramix_decompiler']) and len(targets_info['mythril']) == 0:
            no_targets.add(contract)
            continue

        # Now we go through each target in the decompiled result to see if it is eth address or something else
        count_hard_code = 0
        count_non_hard_code = 0
        for target in targets_info['panoramix']['panoramix_decompiler']:
            # if the target is eth address, we add it to the hard_coded_targets
            # count for hardcoded targets
            if sc.match_eth_account(target):
                count_hard_code += 1
            else:
                count_non_hard_code += 1

        for target in targets_info['mythril']:
            print(f"mythril: {target}")
            # if the target is eth address, we add it to the hard_coded_targets
            # count for hardcoded targets
            if sc.match_eth_account(target):
                count_hard_code += 1
            else:
                count_non_hard_code += 1

        if count_hard_code:
            hard_coded_targets[contract] = count_hard_code

        if count_non_hard_code:
            non_hard_coded_targets[contract] = count_non_hard_code

        if count_hard_code > 1:
            more_than_one_hardcoded_targets.add(contract)

        if count_hard_code + count_non_hard_code > 1:
            more_than_one_all_targets.add(contract)

    # Print out those data
    print(f'Number of contracts with no targets: {len(no_targets)}')
    print(f'Number of contracts with more than one targets: {len(more_than_one_all_targets)}')
    print(f'Number of contracts with more than one hardcoded targets: {len(more_than_one_hardcoded_targets)}')
    print(f'Number of contracts with hardcoded targets: {len(hard_coded_targets)}')

if __name__ == '__main__':
    main()
