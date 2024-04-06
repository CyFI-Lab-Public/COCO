# This script is used to evaluate the popularity of:
# 1. Smart contract deploying smart contract
# 2. Samrt contract contacting orale contract
# TODO: Need to find a way to distinguish the normal smart contract with the oracle smart contract (event only?)

import json
import mingpy.ming_sc as sc
from pathlib import Path
from tqdm import tqdm
import time
import os

def get_all_seeds():
    # Get the path of this script
    path = Path(__file__).parent.absolute()

    # Get the path of the seeds
    seeds_path = path / ".." / "seeds" / "seeds.json"

    # Read the seeds
    with open(seeds_path) as f:
        seeds = json.load(f)

    return seeds

def main():
    
    # read the cache path from os.environ
    sc.set_cache_path(os.environ['CACHE_PATH'])

    # cache_result path
    cache_result_path = Path(__file__).parent.absolute() / "result" / "contract_creator.json"

    # Create the parent folder if not exist
    cache_result_path.parent.mkdir(parents=True, exist_ok=True)

    cache_result = {}

    # Read the cache result
    if cache_result_path.exists():
        with open(cache_result_path) as f:
            cache_result = json.load(f)

    # Get all the seeds
    seeds = get_all_seeds()

    all_contracts = list(seeds['contract'].keys())

    for contract in tqdm(all_contracts, desc="Scraping contract information"):
        # Skip the contract that has been scraped
        if contract in cache_result:
            continue

        # Get the contract creator information
        try:
            print("Scraping contract creator for contract: {}".format(contract))
            contract_creator = sc.scrape_contract_creator(contract)

            cache_result[contract] = contract_creator

        except:
            pass

        # Sleep for 30 seconds to avoid being banned
        time.sleep(30)

    with open(cache_result_path, "w") as f:
        json.dump(cache_result, f, indent=4)

    # Try to see how many cases that the contract creator is a contract
    contract_created_by_contract = set()
    # Try to keep the parents of those contracts
    contract_creating_contarct = set()
    # See how many creators behind those contracts
    back_creators = set()
    # Same back_creators but directly creating contracts
    back_creators_directly_create = set()
    for contract, contract_creator_info in cache_result.items():
        contract_creator = contract_creator_info[0]
        if contract_creator['parent_contract'] and contract_creator['parent_contract'] != contract:
            contract_created_by_contract.add(contract)
            back_creators.add(contract_creator['creator'])
            contract_creating_contarct.add(contract_creator['parent_contract'])

    # See how many contracs those back creators directly created
    for contract, contract_creator_info in cache_result.items():
        contract_creator = contract_creator_info[0]
        if contract_creator['creator'] in back_creators and contract_creator['parent_contract'] is None:
            back_creators_directly_create.add(contract)

    # See how many transactions those back creators have
    back_creators_tx_count = 8868
    # for back_creator in back_creators:
    #     try:
    #         back_creators_tx_count += sc.infura_query_transaction_count(os.environ['INFURA_PROJECT_ID'], back_creator)
    #     except Exception as e:
    #         print(e)
    #         continue

    # For all contracts, pull the byte/hex code back
    # We need set the cache path to infura
    kids_code = set()
    code_of_contracts_created_by_contracts = set()
    parents_code = set()

    sc.set_cache_path(os.environ['CACHE_PATH'])

    for contract, contract_creator_info in tqdm(cache_result.items(), total=len(cache_result), desc="Pulling contract code"):
        code = sc.infura_get_contract_hex_code(os.environ['INFURA_PROJECT_ID'], contract)
        if contract in contract_created_by_contract:
            code_of_contracts_created_by_contracts.add(code)
        kids_code.add(code)

    # Now try to get the code of the contracts creating other contracts
    for contract in tqdm(contract_creating_contarct, total=len(contract_creating_contarct), desc="Pulling parents contract code"):
        code = sc.infura_get_contract_hex_code(os.environ['INFURA_PROJECT_ID'], contract)
        parents_code.add(code)

    print(f" {len(contract_created_by_contract)} contracts is created by {len(contract_creating_contarct)} contracts \n {len(kids_code)} flagged contracts have unique code\n {len(code_of_contracts_created_by_contracts)} contracts created by other contracts have unique code\n {len(parents_code)} parent contracts have unique code\n {len(code_of_contracts_created_by_contracts.intersection(parents_code))} deployed contracts share the same code as the parent contrats\n {len(back_creators)} creators are involved\n They also created {len(back_creators_directly_create)} contracts directly\n They have {back_creators_tx_count} total transactions on blockchain\n")

if __name__ == "__main__":
    main()
