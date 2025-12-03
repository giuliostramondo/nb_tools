import subprocess
import yaml
import re
import bech32
from ecdsa import SigningKey, SECP256k1
import hashlib
import pickle
import os

class Validator:
    def __init__(self, moniker, valoper, account, delegator_shares, vp, status):
        self.moniker = moniker
        self.delegator_shares = delegator_shares
        self.vp = vp
        self.valoper = valoper
        self.account = account
        self.status = status
        self.delegations = []

    def add_delegations(self, delegations):
        for d in delegations:
            self.add_delegation(d['delegation']['delegator_address'],int(d['balance']['amount']))

    def add_delegation(self, delegator, amount):
        self.delegations += [(delegator,amount, amount * 100/ self.delegator_shares)]

    def sort_delegations(self):
        self.delegations = sorted(self.delegations, key= lambda x:x[1], reverse=True)


class StakingState:
    _loading_from_pickle = False   # class-level flag
    def __new__(cls, *args, force_pull=False, **kwargs):
        # If we are currently inside pickle.load → skip custom logic
        if cls._loading_from_pickle:
            return super().__new__(cls)
        # Normal loading path
        if not force_pull and os.path.exists("staking_state.pkl"):
            cls._loading_from_pickle = True
            try:
                with open("staking_state.pkl", "rb") as f:
                    obj = pickle.load(f)
                    obj._loaded_from_pickle = True
                    return obj
            finally:
                cls._loading_from_pickle = False

        # Fresh instance
        obj = super().__new__(cls)
        obj._loaded_from_pickle = False
        return obj

    def __init__(self, _endpoint, force_pull=False):
        if getattr(self, "_loaded_from_pickle", False):
            print("Loading staking data from staking_state.pkl")
            return
        if force_pull:
            print("Overwriting staking data in staking_state.pkl")
        print("Pulling staking data from endpoint... (this will take a while only the first time)")
        self.endpoint = _endpoint
        pool = self.fetch_atomone_staking_pool()
        self.total_bonded = int(pool['bonded_tokens'])
        self.total_not_bonded = int(pool['not_bonded_tokens'])
        self.validators = []
        self.add_validators()
        self.sort_validators()
        print("Storing staking data in staking_state.pkl")
        with open('staking_state.pkl', "wb") as f:
            pickle.dump(self, f)

    def fetch_atomone_validators(self):
        validators_returned_len = 100
        page = 1
        validators = []
        while validators_returned_len == 100:
            cmd = [
                "atomoned", "query", "staking", "validators",
                "--page", str(page),
                "--node", self.endpoint,
                "--output", "yaml",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"'atomoned' command failed (exit {proc.returncode}).\nSTDERR:\n{proc.stderr}"
                )
            ansi_stripped = re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", proc.stdout).strip()
            try:
                parsed = yaml.safe_load(ansi_stripped)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Failed to parse YAML output: {e}")
            validators += parsed['validators']
            validators_returned_len = len(parsed['validators'])
            page += 1
        return validators

    def fetch_atomone_staking_pool(self):
        cmd = [
            "atomoned", "query", "staking", "pool",
            "--node", self.endpoint,
            "--output", "yaml",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"'atomoned' command failed (exit {proc.returncode}).\nSTDERR:\n{proc.stderr}"
            )
        ansi_stripped = re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", proc.stdout).strip()
        try:
            parsed = yaml.safe_load(ansi_stripped)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML output: {e}")
        return parsed


    def val_to_acc(self, val_bech32):
        _, data = bech32.bech32_decode(val_bech32)
        return bech32.bech32_encode('atone', data)

    def add_validators(self):
        res = self.fetch_atomone_validators()
        for v in res:
            self.add_validator(v)

    def fetch_atomone_delegations_to(self, validator_address):
        delegations_returned_len = 100
        page = 1
        delegations = []
        while delegations_returned_len == 100:
            cmd = [
                "atomoned", "query", "staking", "delegations-to", validator_address,
                "--page", str(page),
                "--node", self.endpoint,
                "--output", "yaml",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"'atomoned' command failed (exit {proc.returncode}).\nSTDERR:\n{proc.stderr}"
                )
            ansi_stripped = re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", proc.stdout).strip()
            try:
                parsed = yaml.safe_load(ansi_stripped)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Failed to parse YAML output: {e}")
            delegations += parsed["delegation_responses"]
            delegations_returned_len = len(parsed["delegation_responses"])
            page += 1
        return delegations

    def add_validator(self, v):
        moniker = v['description']['moniker']
        delegator_shares = int(float(v['delegator_shares']))
        tokens = int(float(v['tokens']))
        vp = delegator_shares / self.total_bonded * 100
        valoper = v['operator_address']
        acc = self.val_to_acc(valoper)
        status = v['status']
        v = Validator(moniker, valoper, acc, delegator_shares, vp, status)
        delegations = self.fetch_atomone_delegations_to(valoper)
        v.add_delegations(delegations)
        v.sort_delegations()
        self.validators += [v]

    def sort_validators(self):
        self.validators = sorted(self.validators, key= lambda x:x.vp, reverse=True)

    def dump_state(self, include_delegations=False):
        print("Total Bonded: " +str(self.total_bonded))
        print("Total Not Bonded: " +str(self.total_not_bonded))
        position = 1
        for v in self.validators:
            if v.status == "BOND_STATUS_BONDED":
                print("Moniker: "+ v.moniker + " position: "+str(position))
            else:
                print("Moniker: "+ v.moniker)
            print("\tValoper: "+ v.valoper)
            print("\taccount: "+ v.account)
            print("\tDelegator Shares: "+ str(v.delegator_shares))
            print("\tVP: "+str(v.vp))
            print("\tStatus: "+v.status)
            if include_delegations:
                for d in v.delegations:
                    print(f"\t\t{d[0]} -> {d[1]} {d[2]}%")
            if v.status == "BOND_STATUS_BONDED":
                position += 1



#stakingState = StakingState("https://atomone.rpc.nodeshub.online:443")

