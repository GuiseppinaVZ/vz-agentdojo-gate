# Variable Zero: authority gate for AI agents (v0.3), evaluated on AgentDojo

Code for the technical report "Variable Zero: An Authority Gate for AI Agents Under Adaptive Attack" (Ricci, 2026). Report: [DOI to be added]. Data and logs: [DOI to be added].

## What the gate does

The gate sits between a language-model agent and its tools. Every action that changes something (sending or scheduling money, updating a payment, changing a password or user details) is checked in code against a rulebook before it runs. The model is never asked whether an action is safe. Data may fill in the details of an action the user asked for; it can never add a new action.

Banking rulebook, frozen before testing:

- **R1, authority for the action:** the user's request must ask for this kind of action.
- **R2, recipient:** the recipient must be named by the user or be a known counterparty from the account's own history.
- **R3, values from the user:** sensitive values such as a new password must come from the user.
- **R4, amount band:** for a known payee, the amount stays within 20% of its usual amount unless the user named it.

Modes: `strict` (nobody answers a halt), `simulated2` (a simulated user answers truthfully, corrected version), `simulated` (first version, kept for transparency), and two exploratory halt wordings, `strict_permanent` and `strict_permanent_exit`.

## Files

| File | What it is |
|---|---|
| `vz_gate_agentdojo.py` | The gate and the banking rulebook |
| `run_vz_agentdojo.py` | Usefulness and template-attack runs (no gate, strict, simulated user) |
| `adaptive_attack.py` | Adaptive attacker, replays of winning attacks, cross-transfer, step D |
| `clarify.py` | Simulated user that answers the agent's questions from the reference solution |
| `openai_responses_llm.py` | AgentDojo adapter for the OpenAI Responses API |
| `test_gate_offline.py`, `test_fooled_agent.py`, `test_end_to_end_fake_model.py` | Offline checks, no API key needed |
| `loop20.json`, `loop33.json` | Conversation sets for the exploratory loop tests |

## Running it

Requires Python 3.10 or newer and AgentDojo 0.1.35 (`pip install -r requirements.txt`). API keys are read from the environment (`OPENAI_API_KEY`, or `OPENROUTER_API_KEY` for models routed through OpenRouter) and are never stored in files.

Offline check, no key needed:

```
python test_fooled_agent.py
```

Model outputs vary between runs, so exact counts will differ on repetition. The logs of our runs are in the data package.

## License

Source-available under the PolyForm Noncommercial License 1.0.0: free to read, check and rerun for research and other noncommercial purposes. Commercial use requires a license from Variable Zero. See `LICENSE`.

Built on AgentDojo (ETH Zurich SPY Lab), which has its own license.
