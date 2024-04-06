"""This modules contarns the detection code of call targets and the analysis of the call targets."""
from copy import copy
import logging
from typing import List, cast

from mythril.analysis.solver import get_transaction_sequence, UnsatError
from mythril.analysis.module.base import DetectionModule, EntryPoint
from mythril.laser.ethereum.state.global_state import GlobalState
from mythril.laser.ethereum.state.annotation import StateAnnotation
from mythril.analysis.report import Issue
from mythril.analysis.swc_data import CALL_TARGETS
from mythril.laser.ethereum.transaction.transaction_models import (
    ContractCreationTransaction,
)
from mythril.support.support_args import args
from mythril.laser.smt import Solver
import json
from pathlib import Path
import hashlib

log = logging.getLogger(__name__)

def save_cache(content, folder_name=None, file_name=None):
    # Read the base path from args
    base_path = args.cache_folder

    # If user does not define the cache path, use the current folder
    if not base_path:
        base_path = Path.cwd()

    # Append the folder name
    if folder_name:
        base_path = Path(base_path, folder_name)

    # If the folder does not exist, create it
    if not base_path.exists():
        base_path.mkdir(parents=True)

    # Now try the file name
    # If user defines the file name, use it
    if file_name:
        file_path = Path(base_path, file_name)

    else:
        # If user does not define the file name, use the hash of the content
        file_name = hashlib.sha256(content.encode()).hexdigest()
        file_path = Path(base_path, file_name)

    # Now save the content to the file
    with open(file_path, 'w') as f:
        f.write(content)


class CyFIAccount():
    def __init__(self, mythril_account, world_state=None) -> None:
        self.address_sexpr = mythril_account.address.raw.sexpr()
        self.address_value = mythril_account.address.value
        self.address_symbolic = mythril_account.address.symbolic
        self.nonce = mythril_account.nonce
        self.balance_sexpr = mythril_account.balance().raw.sexpr()
        self.balance_value = mythril_account.balance().value
        self.balance_symbolic = mythril_account.balance().symbolic

        if mythril_account.code.bytecode:
            # The byte code be the string and code be a tuple of smybolic value or a tuple of int
            if isinstance(mythril_account.code.bytecode, str) or (isinstance(mythril_account.code.bytecode, tuple) and isinstance(mythril_account.code.bytecode[0], int)):
                self.bytecode = mythril_account.code.bytecode
            elif isinstance(mythril_account.code.bytecode, tuple):
                self.bytecode = [x.raw.sexpr() for x in mythril_account.code.bytecode]
        else:
            self.bytecode = ""

        self.contract_name = mythril_account.contract_name
        self.storage_keys_set = list(mythril_account.storage.keys_set)
        self.storage = {}

        self.constraints = []

        if world_state:
            for constraint in world_state.constraints:
                self.constraints.append(constraint.raw.sexpr())

        for key in self.storage_keys_set:
            storage = mythril_account.storage[key]
            self.storage[key.value] = {}
            self.storage[key.value]['symbolic'] = storage.symbolic
            self.storage[key.value]['expr'] = storage.raw.sexpr()

            if storage.symbolic and world_state:

                # We could use constraints in world_state to salve possible value or storage
                sol = Solver()

                # Add the constraints
                for constraint in world_state.constraints:
                    sol.add(constraint)
                
                value = ""

                if sol.check():
                    model = sol.model()

                    # Solve the expr with auto complete
                    value = model.eval(storage.raw, model_completion=True)

                    value = value.as_string()

                self.storage[key.value]['value'] = value

            else:
                self.storage[key.value]['value'] = storage.value

    def to_json(self):
        return {
            'address_sexpr': self.address_sexpr,
            'address_value': self.address_value,
            'address_symbolic': self.address_symbolic,
            'nonce': self.nonce,
            'balance_sexpr': self.balance_sexpr,
            'balance_value': self.balance_value,
            'balance_symbolic': self.balance_symbolic,
            'bytecode': self.bytecode,
            'contract_name': self.contract_name,
            # 'storage_keys_set': self.storage_keys_set,
            'storage': self.storage,
            'constraints': self.constraints,
        }


class CyFITxn():
    def __init__(self, txn, world_state_before = None, world_state_after=None, global_state=None):
        self.base_fee_symbolic = txn.base_fee.symbolic
        self.base_fee = txn.base_fee.raw.sexpr()
        self.call_data = txn.call_data._calldata.raw.sexpr()
        self.call_value = txn.call_value.raw.sexpr()
        self.call_value_symbolic = txn.call_value.symbolic
        self.callee_before = CyFIAccount(txn.callee_account, world_state_before)
        self.caller_sexpr = txn.caller.raw.sexpr()
        self.caller_value = txn.caller.value
        self.caller_symbolic = txn.caller.symbolic
        self.bytecode = txn.code.bytecode
        self.origin_sexpr = txn.origin.raw.sexpr()
        self.origin_value = txn.origin.value
        self.origin_symbolic = txn.origin.symbolic

        self.callee_after = None
        if world_state_after:
            # The callee account in txn is the oled callee (when the txn created)
            # We also need to include the callee account from the current world_state
            if self.callee_before.address_value and self.callee_before.address_value in world_state_after.accounts:
                self.callee_after = CyFIAccount(world_state_after.accounts[self.callee_before.address_value], world_state_after)

    def to_json(self):
        return {
            'base_fee': self.base_fee,
            'base_fee_symbolic': self.base_fee_symbolic,
            'call_data': self.call_data,
            'call_value': self.call_value,
            'call_value_symbolic': self.call_value_symbolic,
            'callee_before': self.callee_before.to_json(),
            'caller_sexpr': self.caller_sexpr,
            'caller_value': self.caller_value,
            'caller_symbolic': self.caller_symbolic,
            'bytecode': self.bytecode,
            'origin_sexpr': self.origin_sexpr,
            'origin_value': self.origin_value,
            'origin_symbolic': self.origin_symbolic,
            'callee_after': self.callee_after.to_json() if self.callee_after else None,
        }

class CyFICall():
    def __init__(self, state, world_state=None):
        self.gas_sexpr = state.mstate.stack[-1].raw.sexpr()
        self.gas_value = state.mstate.stack[-1].value
        self.gas_symbolic = state.mstate.stack[-1].symbolic
        self.to_sexpr = state.mstate.stack[-2].raw.sexpr()
        self.to_value = state.mstate.stack[-2].value
        self.to_symbolic = state.mstate.stack[-2].symbolic
        self.caller = CyFIAccount(state.environment.active_account, world_state)

    def to_json(self):
        return {
            "gas_sexpr": self.gas_sexpr,
            "gas_value": self.gas_value,
            "gas_symbolic": self.gas_symbolic,
            "to_sexpr": self.to_sexpr,
            "to_value": self.to_value,
            "to_symbolic": self.to_symbolic,
            "caller": self.caller.to_json()
        }

class CallTargetAnalysisAnnotation(StateAnnotation):
    def __init__(self) -> None:
        self.call_targets = []

    def __copy__(self):
        result = CallTargetAnalysisAnnotation()
        result.call_targets = copy(self.call_targets)
        return result

class CallTargetAnalysis(DetectionModule):
    name = "Call Target Analysis"
    swc_id = CALL_TARGETS
    description = "This module detects the call targets and the analysis of the call targets."
    entry_point = EntryPoint.CALLBACK
    pre_hooks = ["CALL", "DELEGATECALL", "STATICCALL", "CALLCODE", "RETURN"]

    def _execute(self, state:GlobalState) -> None:
        return self._analyze_call_targets(state)

    def _analyze_call_targets(self, state: GlobalState):
        """
        :param state: the current state
        :return: returns the issues for that corresponding state
        """

        # Prepared the data to dump

        # Get current instruction
        instruction = state.get_current_instruction()

        if instruction["opcode"] in ['RETURN']:
            # Get the transaction stack so that we could find the ContractCroationTransaction
            txn_stack = state.transaction_stack

            # No go through each txn
            for txn in txn_stack:
                if not isinstance(txn[0], ContractCreationTransaction):
                    continue

                # Now we start processing the contract creation transaction
                # Cache the transaction first
                txn = txn[0]
                cache_txn = CyFITxn(txn, world_state_before=txn.world_state, world_state_after=state.world_state, global_state=state)
                save_cache(json.dumps(cache_txn.to_json(), indent=4), "ContractCreationTransaction")

        if instruction["opcode"] in ['CALL', 'DELEGATECALL', 'STATICCALL', 'CALLCODE']:
            call_transfer = CyFICall(state, state.world_state)
            save_cache(json.dumps(call_transfer.to_json(), indent=4), "CallTransfer")

detector = CallTargetAnalysis()
