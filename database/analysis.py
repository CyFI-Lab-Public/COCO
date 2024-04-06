# This file is not about db, but will use db json files to do some analysis
import db
import networkx as nx
from tqdm import tqdm
from pathlib import Path
# Get the folder of the current file
current_folder = Path(__file__).parent
# Import the ming_sc module
# import sys
# sys.path.append((current_folder.parent/"analysis"/"modules").as_posix())
# import ming_sc
import multiprocessing as mp

def get_root_nodes(graph):
    return [x for x in graph.nodes if graph.in_degree(x) == 0 and graph.out_degree(x) > 0]

def get_trace_graph_root_nodes(graph):
    # Traces graph could have more loop on the root node, which prohibits us from finding it using in_degree and out_degree.
    for u, v, data in graph.edges(data=True):
        if data['trace_address'] is None:
            return [u]

    return [x for x in graph.nodes if graph.in_degree(x) == 0 and graph.out_degree(x) > 0]

def get_contract_creator(contract_list):
    # Given a list of contract addresses, scrape the contract creators (EOAs) through scrapingdog and multiproces. Note that creators in this case are not necessarily to be the dircet creator of the contract.
    # Return format:
    # {
    #   "address1": "creator1",
    #   "address2": "creator2",
    #   ...
    # }

    # Convert all addresses to lower case
    contract_list = [address.lower() for address in contract_list]

    # Get contract creation txns first
    contract_txn = db.get_creation_txn_given_contract(contract_list)

    # Get all the txn hashes
    txn_hashes = list(contract_txn.values())

    # Get all txn records through db
    txn_records = db.get_all_txns_given_hash_list(txn_hashes)

    # Now compose the return
    ret = {}

    for contract, txn_hash in contract_txn.items():
        if txn_hash not in txn_records:
            continue

        txn_record = txn_records[txn_hash]

        if 'from_address' not in txn_record:
            continue

        ret[contract] = txn_record['from_address']

    return ret

def get_all_traces_from_address(address):
    # Get all the traces from an address
    # We first get all transactions
    address = address.lower()
    transactions = db.get_all_txn_from_to_address(address)

    # Get all 'from' transactions
    from_transactions = transactions['from']

    # Get all hashes
    from_txn_hashes = [txn['hash'] for txn in from_transactions]

    # Get all traces
    traces = db.get_all_traces_given_txn_list(from_txn_hashes)

    return traces

def get_deployed_contracts(address):
    # Get contracts created by an address
    # Return format:
    # {
    #   'txn_hash': {
    #       'from_address': [to_address, to_address, ...],
    #       'from_address': [to_address, to_address, ...],
    #       ...
    #   },
    #   'txn_hash': {
    #       'from_address': [to_address, to_address, ...],
    #       'from_address': [to_address, to_address, ...],
    #       ...
    #   },
    #   ...
    # }

    address = address.lower()

    # Get all traces from the address
    txn_traces = get_all_traces_from_address(address)

    # Get graph from traces
    txn_traces_graph = generate_txn_traces_graph(txn_traces)

    # Get all to node from the graph when the edge trace_type is create
    deployed_contracts = {}

    for txn, graph in txn_traces_graph.items():
        for from_addr, to_addr, edge_data in graph.edges(data=True):
            if edge_data['trace_type'] == 'create':
                if txn not in deployed_contracts:
                    deployed_contracts[txn] = {}

                if from_addr not in deployed_contracts[txn]:
                    deployed_contracts[txn][from_addr] = []

                deployed_contracts[txn][from_addr].append(to_addr)

    return deployed_contracts


def generate_txn_traces_graph(txn_traces):
    # Given a txn to traces dict, generate a txn to traces graph
    result = {}

    for txn, traces in tqdm(txn_traces.items(), total=len(txn_traces)):
        graph = nx.MultiDiGraph()

        for trace in traces:
            # Try to get from and to address, write traces down if we cant find both
            from_addr = trace.get('from_address')
            to_addr = trace.get('to_address')

            if not (from_addr and to_addr):
                # Write this trace down
                print(f"Missing from or to address in trace: {trace}")
                continue

            from_addr = from_addr.lower()
            to_addr = to_addr.lower()

            # Add the edge
            # trace_address could be null, in which case, we use []
            if not trace.get('trace_address'):
                trace_address = []
            else:
                trace_address = trace['trace_address']

            graph.add_edge(from_addr, to_addr, trace=trace, trace_type=trace['trace_type'], trace_address=trace_address)

        result[txn] = graph

    return result

def mp_generate_contract_creation_list_given_txn_traces(txn_traces):
    # Return format:
    # {
    #       'create': [
    #           [addr1, addr2, ..., created_contract_addr],
    #           [addr1, addr2, ..., created_contract_addr],
    #           ...
    #       ],
    #       'suiside': [
    #           [addr1, addr2, ..., killed_contract_addr],
    #           [addr1, addr2, ..., killed_contract_addr],
    #           ...
    #       ]
    # }

    ret = {}

    txn, traces = txn_traces

    trace_graph = nx.MultiDiGraph()

    print(f"Processing txn: {txn}")

    # Generate the graph first
    for trace in traces:
        from_addr = trace.get('from_address')
        to_addr = trace.get('to_address')
        if not (from_addr and to_addr):
            print(f"txn with no from_addr and to_addr {txn}")
            return [txn, ret]
        trace_address = trace.get('trace_address')
        if trace_address:
            trace_address = trace_address.split(',')
        trace_graph.add_edge(from_addr.lower(), to_addr.lower(), trace=trace, trace_type=trace['trace_type'], trace_address=trace_address)

    # We first get all contracts this txn created and killed
    contract_creating_traces = []
    contract_killing_traces = []

    for trace in traces:
        if 'trace_type' not in trace:
            continue
        if trace['trace_type'] == 'create':
            contract_creating_traces.append(trace)

        elif trace['trace_type'] == 'suiside':
            contract_killing_traces.append(trace)

    # Get the root
    root = get_trace_graph_root_nodes(trace_graph)

    if len(root) != 1:
        print(f"Transaction {txn} has more than one root node")

    assert len(root) == 1

    for each_contract_creation_trace in contract_creating_traces:
        path = []
        trace_address = each_contract_creation_trace.get('trace_address')

        if not trace_address:
            path.append(root[0])

        else:
            trace_address = trace_address.split(',')
            address = each_contract_creation_trace['to_address'].lower()

            # We need a visited nodes set cuz it is possible to have loop transaction
            visited = set()

            node = root[0]

            while node != address:

                if node in visited:
                    continue
                else:
                    visited.add(node)

                # Get out edges from the node
                node_out_edges = trace_graph.out_edges(node, data=True)

                # Go through each edge and get all trace
                all_trace_addrs = {}
                for edge in node_out_edges:
                    # if trace_address is empty, we are iterating the out edge of root nodes
                    if not edge[2]['trace_address']:
                        assert edge[0] == root[0]
                        path.append(edge[0])
                        node = edge[1]
                        break

                    # Otherwise, we need to make sure that the trace_address on this edge is the start of the target trace_address
                    txn_trace_addr = edge[2]['trace_address']
                    all_trace_addrs[tuple(txn_trace_addr)] = edge

                # Now we sort the all_trace_addrs with length of each key from long to short and iterate it
                sorted_trace_addrs = sorted(all_trace_addrs.items(), key=lambda x: len(x[0]), reverse=True)
                for each_sort_trace_addr in sorted_trace_addrs:
                    # Check if each_sort_trace_addr is the start of trace_address list
                    if trace_address[:len(each_sort_trace_addr[0])] == list(each_sort_trace_addr[0]):
                        # If it is, we add the edge[0] to the path, update node to edge[1] and update trace_address
                        path.append(each_sort_trace_addr[1][0])
                        node = each_sort_trace_addr[1][1]
                        break

            path.append(address)

        # Add this to ret
        if 'create' not in ret:
            ret['create'] = []
        ret['create'].append(path)

    for each_contract_killing_trace in contract_killing_traces:
        path = []
        trace_address = each_contract_killing_trace['trace_address']

        if not trace_address:
            path.append(root[0])

        else:
            address = each_contract_killing_trace['from_address'].lower()

            # We need a visited nodes set cuz it is possible to have loop transaction
            visited = set()

            node = root[0]

            while node != address:

                if node in visited:
                    continue
                else:
                    visited.add(node)

                # Get out edges from the node
                node_out_edges = trace_graph.out_edges(node, data=True)

                # Go through each edge and get all trace
                all_trace_addrs = {}
                for edge in node_out_edges:
                    # if trace_address is empty, we are iterating the out edge of root nodes
                    if not edge[2]['trace_address']:
                        assert edge[0] == root[0]
                        path.append(edge[0])
                        node = edge[1]
                        break

                    # Otherwise, we need to make sure that the trace_address on this edge is the start of the target trace_address
                    txn_trace_addr = edge[2]['trace_address']
                    all_trace_addrs[tuple(txn_trace_addr)] = edge

                # Now we sort the all_trace_addrs with length of each key from long to short and iterate it
                sorted_trace_addrs = sorted(all_trace_addrs.items(), key=lambda x: len(x[0]), reverse=True)
                for each_sort_trace_addr in sorted_trace_addrs:
                    # Check if each_sort_trace_addr is the start of trace_address list
                    if trace_address[:len(each_sort_trace_addr[0])] == list(each_sort_trace_addr[0]):
                        # If it is, we add the edge[0] to the path, update node to edge[1] and update trace_address
                        path.append(each_sort_trace_addr[1][0])
                        node = each_sort_trace_addr[1][1]
                        break

            path.append(address)

        # Add this to ret
        if 'suiside' not in ret:
            ret['suiside'] = []
        ret['suiside'].append(path)

    return [txn, ret]

def get_creators_contract_creation_list(creators):
    # Given a list of creators, build contract creation graph for each creator
    # Return format:
    # {
    #   "creator1": { 'txn1': {create': [path1, path2, ...], 'suiside': [path1, path2, ...]'}}
    #   "creator2": { 'txn2': {create': [path1, path2, ...], 'suiside': [path1, path2, ...]'}}
    #   ...
    # }
    # For each graph, the nodes are contracts, and the edgies are the contract creation relationship (create, suiside)

    # Since each creator is an EOA, we first get all transactions from that EOA

    # Update the creators list to the lower case one
    creators = [creator.lower() for creator in creators]

    ret = {}

    creator_txns = db.get_all_txn_from_to_address_list(creators)

    # The return should have both from and to, but we only care about from atm. Therefore, we get all txns with 'from' first
    txn_list = set()

    for creator, txns in creator_txns.items():
        for txn in txns['from']:
            txn_list.add(txn['hash'])

    txn_traces = db.get_all_traces_given_txn_list(txn_list)

    # Now the task is to generate the contract creation graph based on the traces information we have
    txn_contract_creation_suiside = {}

    # We use mp to process each txn to speed up
    # First, we get everything into tuple as arguments
    args = []
    for txn, traces in txn_traces.items():
        args.append((txn, traces))

    # Then we use mp to process
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(mp_generate_contract_creation_list_given_txn_traces, args)

        for result in tqdm(results, total=len(args)):
            # It's possible that txn is not creating/suisiding any contract
            if result[1]:
                txn_contract_creation_suiside[result[0]] = result[1]

    for creator, txns in creator_txns.items():
        for txn in txns['from']:
            if txn['hash'] in txn_contract_creation_suiside:
                if creator not in ret:
                    ret[creator] = {}
                ret[creator][txn['hash']] = txn_contract_creation_suiside[txn['hash']]

    return ret
