import analysis

def contracts_pipeline(contract_list):
    # The pipeline that starts from a list of contracts

    # Turn the contract_list into a lower case set
    contract_set = set([contract.lower() for contract in contract_list])

    # Get all the creators (EOAs) of these contracts
    contract_creator = analysis.get_contract_creator(contract_set)

    # We want to generate the graph for each contract creator
    # We get all creators first
    creators = set(contract_creator.values())

    # Now we generate the contract creation graph for each creator
    creators_contract_creation_list = analysis.get_creators_contract_creation_list(creators)
