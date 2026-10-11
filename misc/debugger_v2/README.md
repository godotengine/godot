# Debugger protocol v2 tests

These black-box tests exercise the dual-channel debugger transport, execution
control, reconnect behavior, lazy variable loading, and bounded data queues.
They run against a locally built Godot 3 editor binary.

Run the protocol model tests from the repository root:

```sh
python misc/debugger_v2/test_protocol_model.py
python misc/debugger_v2/test_debug_session_model.py
```

Run an engine scenario with a custom editor binary:

```sh
python misc/debugger_v2/fake_debug_server.py --godot <path-to-godot> --scenario handshake
```

Available scenarios are `handshake`, `control`, `timeout`, `reconnect`,
`data_disconnect`, `lazy`, `backpressure`, `executable_lines`, and
`call_stepping`. Run `test_editor_frontend.py --godot <path-to-godot>` to
exercise the editor frontend against a child game process.
