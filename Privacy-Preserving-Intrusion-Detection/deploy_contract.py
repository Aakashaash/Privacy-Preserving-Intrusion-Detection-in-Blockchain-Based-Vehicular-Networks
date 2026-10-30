
import json
import os
from web3 import Web3
from solcx import compile_source, install_solc

# ---------- Configuration & Environment Bindings ----------
RPC_URL = os.getenv("WEB3_RPC_URL", "http://127.0.0.1:8545")
CONTRACT_SRC_PATH = "SecurityLedger.sol"
ABI_OUT_DIR = "contracts"
ABI_OUT_PATH = os.path.join(ABI_OUT_DIR, "SecurityLedger.abi.json")

SOLIDITY_VERSION = "0.8.20"

def get_solidity_source():
    """Reads contract payload securely from designated source paths."""
    if not os.path.exists(CONTRACT_SRC_PATH):
        raise FileNotFoundError(
            f"Error: Contract missing at '{CONTRACT_SRC_PATH}'. "
            f"Please save SecurityLedger.sol under the contracts directory first."
        )
    with open(CONTRACT_SRC_PATH, "r", encoding="utf-8") as f:
        return f.read()

def run():
    print(f"\n[DEPLOYER] Setting up native Solidity compiler environment v{SOLIDITY_VERSION}...")
    install_solc(SOLIDITY_VERSION)
    
    source_code = get_solidity_source()
    
    print("[DEPLOYER] Compiling smart contract source to binary bytecode...")
    compiled_sol = compile_source(
        source_code,
        solc_version=SOLIDITY_VERSION,
        output_values=["abi", "bin"]
    )
    
    # Resolve exact compile map keys dynamically
    contract_key = f"<stdin>:SecurityLedger"
    if contract_key not in compiled_sol:
        contract_key = next(iter(compiled_sol.keys()))
        
    contract_interface = compiled_sol[contract_key]
    abi = contract_interface['abi']
    bytecode = contract_interface['bin']
    
    # Export system ABI file required by step7_blockchain.py
    os.makedirs(ABI_OUT_DIR, exist_ok=True)
    with open(ABI_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(abi, f, indent=4)
    print(f"[DEPLOYER] JSON Interface Signature mapped explicitly to: {ABI_OUT_PATH}")

    # Establish network node target
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Deployment aborted. Target JSON-RPC provider unavailable at: {RPC_URL}")
        
    deployer_account = w3.eth.accounts[0]
    w3.eth.default_account = deployer_account
    print(f"[DEPLOYER] Connected securely. Broadcast account address: {deployer_account}")
    print(f"[DEPLOYER] Sandbox ledger height: Block #{w3.eth.block_number}")

    # Build transaction object mapping
    LedgerContract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    print("[DEPLOYER] Deploying contract instance to network consensus layer...")
    tx_hash = LedgerContract.constructor().transact({
        'from': deployer_account,
        'gas': 3_000_000
    })
    
    print(f"[DEPLOYER] Pending Transaction Signature: {tx_hash.hex()}")
    print("[DEPLOYER] Awaiting network transaction execution receipt...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    deployed_address = tx_receipt.contractAddress
    print("\n" + "="*65)
    print(" 🎉 CONTRACT DEPLOYED SUCCESSFULLY TO THE CONSENSUS LAYER")
    print("="*65)
    print(f"  Target Contract Address:  {deployed_address}")
    print(f"  Deployment Block Height:  #{tx_receipt.blockNumber}")
    print(f"  Gas Units Allocated:      {tx_receipt.gasUsed}")
    print("="*65)
    
    print(f"\n[ACTION REQUIRED] Run this command in your console terminal environment to link your pipeline runtime:")
    print(f'  $env:CONTRACT_ADDR="{deployed_address}"')
    print("="*65 + "\n")
    
    return deployed_address

if __name__ == "__main__":
    run()
