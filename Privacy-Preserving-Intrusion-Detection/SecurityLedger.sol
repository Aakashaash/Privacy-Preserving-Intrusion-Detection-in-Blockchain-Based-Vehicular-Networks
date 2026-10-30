// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SecurityLedger {
    struct Event {
        string vehicleId;
        string eventId;
        string eventType;
        uint256 timestamp;
        uint256 previousTrust;
        uint256 updatedTrust;
        string verificationStatus;
        string modelVersion;
        bytes32 evidenceHash;
        uint256 approvals;
        bool committed;
    }
    struct Validator {
        uint256 stake;
        uint256 reputation;   // scaled 1e4
        bool active;
    }

    address public owner;
    mapping(address => Validator) public validators;
    uint256 public totalStake;
    uint256 public constant QUORUM_PCT = 67;
    uint256 public constant MIN_STAKE  = 1 ether;
    uint256 public constant SLASH_PCT  = 5;

    Event[] public events;
    mapping(bytes32 => mapping(address => bool)) public attestations;
    mapping(bytes32 => uint256) public approvalWeight;
    mapping(bytes32 => uint256) public totalWeightAtProposal;

    event ValidatorJoined(address indexed v, uint256 stake);
    event ValidatorSlashed(address indexed v, uint256 amount);
    event EventProposed(uint256 indexed id, bytes32 hash, string vehicleId);
    event EventCommitted(uint256 indexed id, bytes32 hash);
    event ReputationUpdated(address indexed v, uint256 newReputation);

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }
    modifier onlyValidator() { require(validators[msg.sender].active, "not validator"); _; }

    constructor() {
        owner = msg.sender;
        validators[msg.sender] = Validator(1 ether, 10000, true);
        totalStake = 1 ether;
    }

    function stake() external payable {
        require(msg.value >= MIN_STAKE, "below min stake");
        if (!validators[msg.sender].active) {
            validators[msg.sender] = Validator(msg.value, 5000, true);
        } else {
            validators[msg.sender].stake += msg.value;
        }
        totalStake += msg.value;
        emit ValidatorJoined(msg.sender, msg.value);
    }

    function votingWeight(address v) public view returns (uint256) {
        Validator memory val = validators[v];
        if (!val.active) return 0;
        return (val.stake / 1e12) * (10000 + val.reputation) / 10000;
    }

    function proposeEvent(
        string calldata vehicleId, string calldata eventId,
        string calldata eventType, uint256 previousTrust,
        uint256 updatedTrust, string calldata verificationStatus,
        string calldata modelVersion, bytes32 evidenceHash
    ) external onlyValidator returns (uint256) {
        bytes32 h = keccak256(abi.encodePacked(vehicleId, eventId, evidenceHash));
        uint256 id = events.length;
        events.push(Event(vehicleId, eventId, eventType, block.timestamp,
                          previousTrust, updatedTrust, verificationStatus,
                          modelVersion, evidenceHash, 0, false));
        totalWeightAtProposal[h] = (totalStake / 1e12) * 15000 / 10000;
        emit EventProposed(id, h, vehicleId);
        return id;
    }

    function attest(bytes32 eventHash) external onlyValidator {
        require(!attestations[eventHash][msg.sender], "already attested");
        attestations[eventHash][msg.sender] = true;
        approvalWeight[eventHash] += votingWeight(msg.sender);
        uint256 total = totalWeightAtProposal[eventHash];
        if (total > 0 && (approvalWeight[eventHash] * 100) / total >= QUORUM_PCT) {
            _commit(eventHash);
        }
    }

    function _commit(bytes32 h) internal {
        for (uint i = 0; i < events.length; i++) {
            bytes32 eh = keccak256(abi.encodePacked(
                events[i].vehicleId, events[i].eventId, events[i].evidenceHash));
            if (eh == h && !events[i].committed) {
                events[i].committed = true;
                events[i].approvals = approvalWeight[h];
                emit EventCommitted(i, h);
                return;
            }
        }
    }

    function slash(address validator) external onlyOwner {
        Validator storage v = validators[validator];
        require(v.active, "not active");
        uint256 amt = (v.stake * SLASH_PCT) / 100;
        v.stake -= amt; totalStake -= amt;
        v.reputation = v.reputation >= 500 ? v.reputation - 500 : 0;
        emit ValidatorSlashed(validator, amt);
        emit ReputationUpdated(validator, v.reputation);
    }

    function totalEvents() external view returns (uint256) { return events.length; }
    function isCommitted(uint256 id) external view returns (bool) { return events[id].committed; }
}