"""
Python class representing an account on ethereum blockchain with static features
Note: The transactions are not stored in the class
"""

class Account():
    def __init__(self):
        self.address = None
        self.creation_date = None
        self.malicious = False

    def is_eoa(self):
        # Raise not implemented error
        raise NotImplementedError

    def is_smart_contract(self):
        # Raise not implemented error
        raise NotImplementedError

    def is_malicious(self):
        return self.malicious
        
class SC(Account):
    def __init__(self):
        self.smart_contract = True

    def is_smart_contract(self):
        return self.smart_contract

class EOA(Account):
    def __init__(self):
        self.eoa = True

    def is_eoa(self):
        return self.eoa
