"""
High-level pipeline scripts for investigating the fradulent smart contract
The scripts take one smart contract address as input and output results
"""
import sc as sc
import mg
import argparse

def analyze_smart_contract(smart_contract, cache_dir):
    # Set the cache directory of sc
    sc.set_cache_path(cache_dir)

    # In order to generate the data for the tables, we need to generate the full graph

    # Tracing back through the creation transactions
    # The return is a graph, each node is a smart contract, each edge is a transaction, root should be an EOA
    creation_chain = sc.trace_creation(smart_contract)

    # We trace forward throughe transaction given this creation chain
    full_graph = sc.trace_forward(creation_chain)

    # We also performs code analysis to compensate the graph with potential txn edges
    full_graph_with_code = sc.code_analysis_flavor(full_graph)

    # Now we include the victims in the graph
    full_graph_with_victims = sc.add_victims(full_graph_with_code)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Analyse a smart contract")

    # Add the arguments
    parser.add_argument('-s', '--smart-contract', type=str, help='smart contract address')
    parser.add_argument('-c', '--cache-dir', type=str, help='cache directory')

    # Parse the arguments
    args = parser.parse_args()

    # Make sure that users specify both smart contract address and cache directory
    if args.smart_contract is None or args.cache_dir is None:
        parser.print_help()
        return

    analyze_smart_contract(args.smart_contract, args.cache_dir)

if __name__ == "__main__":
    main()
