"""
Python class representing a transaction on ethereum blockchain with static features
"""

class Txn():
    def __init__(self):
        self.block_number = None
        self.timestamp = None
        self.sender = None
        self.receiver = None
        self.value = None
        self.gas = None
