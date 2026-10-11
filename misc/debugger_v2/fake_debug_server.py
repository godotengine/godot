#!/usr/bin/env python3
"""Black-box protocol-v2 handshake and execution-control test server."""

from __future__ import annotations

import argparse
import select
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


MAGIC = "GDDBG2"
MAJOR = 2
MINOR = 0

TYPE_NIL = 0
TYPE_BOOL = 1
TYPE_INT = 2
TYPE_STRING = 4
TYPE_ARRAY = 19
FLAG_64 = 1 << 16

EXECUTABLE_LINE_STOPS = [9, 14, 15, 19, 20, 21]


class ProtocolError(RuntimeError):
    pass


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProtocolError("connection closed in the middle of a packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_variant(data: bytes, offset: int = 0) -> tuple[Any, int]:
    if len(data) - offset < 4:
        raise ProtocolError("truncated Variant header")
    raw_type = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    variant_type = raw_type & 0xFF

    if variant_type == TYPE_NIL:
        return None, offset
    if variant_type == TYPE_BOOL:
        if len(data) - offset < 4:
            raise ProtocolError("truncated bool")
        return bool(struct.unpack_from("<I", data, offset)[0]), offset + 4
    if variant_type == TYPE_INT:
        if raw_type & FLAG_64:
            if len(data) - offset < 8:
                raise ProtocolError("truncated int64")
            return struct.unpack_from("<q", data, offset)[0], offset + 8
        if len(data) - offset < 4:
            raise ProtocolError("truncated int32")
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if variant_type == TYPE_STRING:
        if len(data) - offset < 4:
            raise ProtocolError("truncated string length")
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        padded = (length + 3) & ~3
        if len(data) - offset < padded:
            raise ProtocolError("truncated string")
        return data[offset : offset + length].decode("utf-8"), offset + padded
    if variant_type == TYPE_ARRAY:
        if len(data) - offset < 4:
            raise ProtocolError("truncated array length")
        count = struct.unpack_from("<I", data, offset)[0] & 0x7FFFFFFF
        offset += 4
        result: list[Any] = []
        for _ in range(count):
            value, offset = _decode_variant(data, offset)
            result.append(value)
        return result, offset

    raise ProtocolError(f"unsupported Variant type {variant_type}")


def _encode_variant(value: Any) -> bytes:
    if value is None:
        return struct.pack("<I", TYPE_NIL)
    if isinstance(value, bool):
        return struct.pack("<II", TYPE_BOOL, int(value))
    if isinstance(value, int):
        if -(1 << 31) <= value <= (1 << 31) - 1:
            return struct.pack("<Ii", TYPE_INT, value)
        return struct.pack("<Iq", TYPE_INT | FLAG_64, value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        padding = b"\0" * ((-len(encoded)) % 4)
        return struct.pack("<II", TYPE_STRING, len(encoded)) + encoded + padding
    if isinstance(value, (list, tuple)):
        return (
            struct.pack("<II", TYPE_ARRAY, len(value))
            + b"".join(_encode_variant(item) for item in value)
        )
    raise ProtocolError(f"cannot encode Python type {type(value).__name__}")


def _receive_packet(connection: socket.socket) -> Any:
    size = struct.unpack("<I", _read_exact(connection, 4))[0]
    body = _read_exact(connection, size)
    value, used = _decode_variant(body)
    if used != len(body):
        raise ProtocolError("trailing bytes after Variant packet")
    return value


def _send_packet(connection: socket.socket, value: Any) -> None:
    body = _encode_variant(value)
    connection.sendall(struct.pack("<I", len(body)) + body)


def _ack_for(hello: list[Any]) -> list[Any]:
    return [
        MAGIC,
        MAJOR,
        MINOR,
        hello[3],
        hello[4],
        hello[5],
        1,
        hello[7],
        0,
        "hello_ack",
        [250, 1000, 100],
    ]


def _command(
    hello: list[Any],
    sequence: int,
    request_id: int,
    stop_id: int,
    message_type: str,
    payload: list[Any] | None = None,
) -> list[Any]:
    return [
        MAGIC,
        MAJOR,
        MINOR,
        hello[3],
        hello[4],
        hello[5],
        sequence,
        request_id,
        stop_id,
        message_type,
        payload or [],
    ]


def _receive_envelope(connection: socket.socket, message_type: str) -> list[Any]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            packet = _receive_packet(connection)
        except ProtocolError as error:
            if str(error).startswith("unsupported Variant type"):
                continue
            raise
        if isinstance(packet, list) and len(packet) == 11 and packet[9] == message_type:
            return packet
    raise ProtocolError(f"timed out waiting for {message_type}")


def _assert_ack(packet: list[Any], request_id: int, accepted: bool, reason: str, duplicate: bool = False) -> None:
    if packet[7] != request_id:
        raise ProtocolError(f"ack request id {packet[7]} != {request_id}")
    payload = packet[10]
    if payload[:3] != [accepted, duplicate, reason]:
        raise ProtocolError(f"unexpected ack payload for request {request_id}: {payload}")


def _exercise_control(control: socket.socket, hello: list[Any], result: dict[str, Any]) -> None:
    _send_packet(control, _command(hello, 2, 2, 0, "pause"))
    _assert_ack(_receive_envelope(control, "ack"), 2, True, "accepted")
    stopped = _receive_envelope(control, "stopped")
    if len(stopped[10]) != 3:
        raise ProtocolError(f"stopped event is not minimal: {stopped[10]}")
    stop_id = stopped[8]
    sequence = 10

    pause = _command(hello, sequence, 10, stop_id, "pause")
    _send_packet(control, pause)
    _assert_ack(_receive_envelope(control, "ack"), 10, True, "already_paused")

    _send_packet(control, pause)
    _assert_ack(_receive_envelope(control, "ack"), 10, True, "already_paused", duplicate=True)

    sequence += 1
    _send_packet(control, _command(hello, sequence, 11, stop_id - 1, "next"))
    _assert_ack(_receive_envelope(control, "ack"), 11, False, "stale_stop_id")

    sequence += 1
    _send_packet(control, _command(hello, sequence, 9, stop_id, "next"))
    _assert_ack(_receive_envelope(control, "ack"), 9, False, "out_of_order_request_id")

    acknowledgement_msec: list[float] = []
    for cycle in range(20):
        sequence += 1
        request_id = 20 + cycle
        started = time.perf_counter()
        _send_packet(control, _command(hello, sequence, request_id, stop_id, "next"))
        _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
        acknowledgement_msec.append((time.perf_counter() - started) * 1000.0)
        _receive_envelope(control, "running")
        stopped = _receive_envelope(control, "stopped")
        if stopped[8] <= stop_id:
            raise ProtocolError("stop id did not increase after next")
        stop_id = stopped[8]

    sorted_latency = sorted(acknowledgement_msec)
    p95_index = max(0, int(len(sorted_latency) * 0.95) - 1)
    result["ack_p95_msec"] = sorted_latency[p95_index]
    if result["ack_p95_msec"] >= 50.0:
        raise ProtocolError(f"local control ack P95 is {result['ack_p95_msec']:.2f} ms")

    result["control_closed_at"] = time.perf_counter()
    control.shutdown(socket.SHUT_RDWR)
    control.close()


def _exercise_timeout(control: socket.socket, hello: list[Any], result: dict[str, Any]) -> None:
    _send_packet(control, _command(hello, 2, 2, 0, "pause"))
    _assert_ack(_receive_envelope(control, "ack"), 2, True, "accepted")
    _receive_envelope(control, "stopped")
    started = time.perf_counter()
    diagnostic = _receive_envelope(control, "diagnostic")
    result["timeout_recovery_msec"] = (time.perf_counter() - started) * 1000.0
    if diagnostic[10] != ["control_timeout_auto_continue"]:
        raise ProtocolError(f"unexpected timeout diagnostic: {diagnostic[10]}")
    if not 900.0 <= result["timeout_recovery_msec"] < 1250.0:
        raise ProtocolError(
            f"heartbeat timeout recovery took {result['timeout_recovery_msec']:.2f} ms"
        )


def _exercise_lazy_data(
    control: socket.socket,
    data: socket.socket,
    control_hello: list[Any],
    data_hello: list[Any],
    result: dict[str, Any],
) -> None:
    stopped = _receive_envelope(control, "stopped")
    if len(stopped[10]) != 3:
        raise ProtocolError("stopped eagerly contains debugger data")
    stop_id = stopped[8]

    started = time.perf_counter()
    _send_packet(data, _command(data_hello, 2, 100, stop_id, "stack_trace", [0, 20]))
    stack = _receive_envelope(data, "stack_trace_result")
    result["stack_msec"] = (time.perf_counter() - started) * 1000.0
    if result["stack_msec"] >= 200.0 or not stack[10][1]:
        raise ProtocolError(f"lazy stack response invalid or slow: {stack[10]}")
    frame_id = stack[10][1][0][0]

    _send_packet(data, _command(data_hello, 3, 101, stop_id, "scopes", [frame_id]))
    scopes = _receive_envelope(data, "scopes_result")[10][0]
    scope_by_name = {scope[0]: scope[1] for scope in scopes}
    if set(scope_by_name) != {"Locals", "Members", "Globals"}:
        raise ProtocolError(f"unexpected scopes: {scopes}")

    locals_handle = scope_by_name["Locals"]
    _send_packet(data, _command(data_hello, 4, 102, stop_id, "variables", [locals_handle, 0, 100]))
    local_page = _receive_envelope(data, "variables_result")[10]
    descriptors = {item[0]: item for item in local_page[2]}
    required = {"huge_array", "huge_dictionary", "long_text", "object_value"}
    if not required.issubset(descriptors):
        raise ProtocolError(f"missing lazy locals: {sorted(required - set(descriptors))}")
    if len(descriptors["long_text"][2]) > 260 or descriptors["long_text"][3] == 0:
        raise ProtocolError("long string was not previewed with a child handle")

    array_handle = descriptors["huge_array"][3]
    started = time.perf_counter()
    _send_packet(data, _command(data_hello, 5, 103, stop_id, "variables", [array_handle, 0, 1000]))
    array_page = _receive_envelope(data, "variables_result")[10]
    result["array_page_msec"] = (time.perf_counter() - started) * 1000.0
    if array_page[0] != 10000 or len(array_page[2]) != 100 or not array_page[3]:
        raise ProtocolError(f"array was not paged to 100 entries: {array_page[:2]}")
    if result["array_page_msec"] >= 200.0:
        raise ProtocolError(f"array page took {result['array_page_msec']:.2f} ms")

    dictionary_handle = descriptors["huge_dictionary"][3]
    _send_packet(data, _command(data_hello, 6, 105, stop_id, "variables", [dictionary_handle, 0, 1000]))
    dictionary_page = _receive_envelope(data, "variables_result")[10]
    if dictionary_page[0] != 300 or len(dictionary_page[2]) != 100 or not dictionary_page[3]:
        raise ProtocolError("dictionary was not paged to the negotiated limit")

    object_handle = descriptors["object_value"][3]
    _send_packet(data, _command(data_hello, 7, 106, stop_id, "inspect_object", [object_handle, 0, 100]))
    inspected = _receive_envelope(data, "inspect_object_result")[10]
    if inspected[0] <= 0 or not inspected[2]:
        raise ProtocolError("object inspection did not return lazy properties")

    _send_packet(control, _command(control_hello, 8, 50, stop_id, "continue"))
    _assert_ack(_receive_envelope(control, "ack"), 50, True, "accepted")
    _receive_envelope(control, "running")
    next_stopped = _receive_envelope(control, "stopped")
    next_stop_id = next_stopped[8]
    _send_packet(data, _command(data_hello, 8, 104, next_stop_id, "variables", [array_handle, 0, 100]))
    stale = _receive_envelope(data, "request_error")
    if stale[10][0] != "stale_handle":
        raise ProtocolError(f"old handle was not invalidated: {stale[10]}")

    _send_packet(control, _command(control_hello, 9, 51, next_stop_id, "continue"))
    _assert_ack(_receive_envelope(control, "ack"), 51, True, "accepted")
    _receive_envelope(control, "running")


def _exercise_backpressure(
    control: socket.socket,
    data: socket.socket,
    control_hello: list[Any],
    data_hello: list[Any],
    result: dict[str, Any],
) -> None:
    time.sleep(0.2)
    _send_packet(control, _command(control_hello, 2, 2, 0, "pause"))
    _assert_ack(_receive_envelope(control, "ack"), 2, True, "accepted")
    stopped = _receive_envelope(control, "stopped")
    stop_id = stopped[8]

    acknowledgement_msec: list[float] = []
    next_stop_msec: list[float] = []
    for cycle in range(100):
        request_id = 20 + cycle
        started = time.perf_counter()
        _send_packet(control, _command(control_hello, 10 + cycle, request_id, stop_id, "next"))
        _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
        acknowledgement_msec.append((time.perf_counter() - started) * 1000.0)
        _receive_envelope(control, "running")
        stopped = _receive_envelope(control, "stopped")
        next_stop_msec.append((time.perf_counter() - started) * 1000.0)
        stop_id = stopped[8]

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[max(0, int(len(ordered) * fraction) - 1)]

    result["pressure_ack_p95_msec"] = percentile(acknowledgement_msec, 0.95)
    result["pressure_stop_p95_msec"] = percentile(next_stop_msec, 0.95)
    if result["pressure_ack_p95_msec"] >= 50.0:
        raise ProtocolError("control acknowledgements were delayed by data pressure")
    if result["pressure_stop_p95_msec"] >= 300.0:
        raise ProtocolError("next-stop latency exceeded 300 ms under data pressure")

    _send_packet(data, _command(data_hello, 199, 499, stop_id, "stress_data_queue", [1024]))
    _receive_envelope(data, "stress_data_queue_result")
    _send_packet(data, _command(data_hello, 200, 500, stop_id, "queue_stats"))
    queue_stats = _receive_envelope(data, "queue_stats_result")[10]
    if queue_stats[1] > queue_stats[2] or queue_stats[3] <= 0:
        raise ProtocolError(f"data queue was not bounded or did not drop output: {queue_stats}")
    result["pressure_dropped"] = queue_stats[3]

    _send_packet(data, _command(data_hello, 201, 501, stop_id, "scene_snapshot"))
    scene_one = _receive_envelope(data, "scene_snapshot_result")[10]
    _send_packet(data, _command(data_hello, 202, 502, stop_id, "scene_snapshot"))
    scene_two = _receive_envelope(data, "scene_snapshot_result")[10]
    if scene_one[0] != 1 or scene_two[0] != 2 or not scene_one[1]:
        raise ProtocolError("explicit scene snapshots did not carry revisions")

    _send_packet(data, _command(data_hello, 203, 503, stop_id, "export_trace"))
    trace = _receive_envelope(data, "export_trace_result")[10][0]
    if not trace or any(len(entry) != 9 for entry in trace):
        raise ProtocolError("trace export has an invalid metadata schema")
    if any("pressure payload" in str(field) for entry in trace for field in entry):
        raise ProtocolError("trace leaked debugger values")

    result["control_closed_at"] = time.perf_counter()
    control.shutdown(socket.SHUT_RDWR)
    control.close()


def _request_top_frame_line(
    data: socket.socket,
    data_hello: list[Any],
    sequence: int,
    request_id: int,
    stop_id: int,
) -> int:
    _send_packet(
        data,
        _command(data_hello, sequence, request_id, stop_id, "stack_trace", [0, 1]),
    )
    stack = _receive_envelope(data, "stack_trace_result")[10]
    frames = stack[1]
    if not frames:
        raise ProtocolError("executable-line stop did not provide a stack frame")
    return int(frames[0][2])


def _request_stack(
    data: socket.socket,
    data_hello: list[Any],
    sequence: int,
    request_id: int,
    stop_id: int,
) -> list[list[Any]]:
    _send_packet(
        data,
        _command(data_hello, sequence, request_id, stop_id, "stack_trace", [0, 20]),
    )
    stack = _receive_envelope(data, "stack_trace_result")[10]
    frames = stack[1]
    if not frames:
        raise ProtocolError("call-stepping stop did not provide a stack frame")
    return frames


def _exercise_executable_lines(
    control: socket.socket,
    data: socket.socket,
    control_hello: list[Any],
    data_hello: list[Any],
    result: dict[str, Any],
) -> None:
    stopped = None
    diagnostics: list[Any] = []
    deadline = time.monotonic() + 3.0
    while stopped is None and time.monotonic() < deadline:
        readable, _, _ = select.select([control, data], [], [], 0.25)
        for connection in readable:
            packet = _receive_packet(connection)
            if not isinstance(packet, list) or len(packet) != 11:
                continue
            if packet[9] == "stopped":
                stopped = packet
            elif packet[9] in ("error", "noncontinuable_error", "diagnostic", "output"):
                diagnostics.append([packet[9], packet[10]])
    if stopped is None:
        raise ProtocolError(
            f"executable-line fixture did not stop; diagnostics={diagnostics}"
        )
    stop_id = stopped[8]
    control_sequence = 2
    data_sequence = 2
    request_id = 10
    data_request_id = 100
    actual = [
        _request_top_frame_line(
            data, data_hello, data_sequence, data_request_id, stop_id
        )
    ]

    for _ in range(len(EXECUTABLE_LINE_STOPS) - 1):
        _send_packet(
            control,
            _command(
                control_hello,
                control_sequence,
                request_id,
                stop_id,
                "next",
            ),
        )
        _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
        _receive_envelope(control, "running")
        try:
            stopped = _receive_envelope(control, "stopped")
        except ProtocolError as error:
            raise ProtocolError(
                f"fixture exited before the expected stop sequence; stops={actual}"
            ) from error
        stop_id = stopped[8]
        control_sequence += 1
        request_id += 1
        data_sequence += 1
        data_request_id += 1
        actual.append(
            _request_top_frame_line(
                data, data_hello, data_sequence, data_request_id, stop_id
            )
        )

    result["executable_line_stops"] = actual
    if actual != EXECUTABLE_LINE_STOPS:
        raise ProtocolError(
            "step sequence included a non-executable line: "
            f"expected {EXECUTABLE_LINE_STOPS}, got {actual}"
        )

    _send_packet(
        control,
        _command(
            control_hello,
            control_sequence,
            request_id,
            stop_id,
            "continue",
        ),
    )
    _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
    _receive_envelope(control, "running")


def _exercise_call_stepping(
    control: socket.socket,
    data: socket.socket,
    control_hello: list[Any],
    data_hello: list[Any],
    result: dict[str, Any],
) -> None:
    stopped = _receive_envelope(control, "stopped")
    stop_id = stopped[8]
    control_sequence = 2
    data_sequence = 2
    request_id = 10
    data_request_id = 100
    actual: list[tuple[str, int]] = []

    def capture(expected_function: str, expected_line: int, expected_caller: bool) -> None:
        nonlocal data_sequence, data_request_id
        frames = _request_stack(
            data, data_hello, data_sequence, data_request_id, stop_id
        )
        top = (str(frames[0][3]), int(frames[0][2]))
        actual.append(top)
        if top != (expected_function, expected_line):
            raise ProtocolError(
                f"unexpected call-stepping stop: expected {(expected_function, expected_line)}, got {top}"
            )
        if expected_caller:
            if len(frames) < 2 or str(frames[1][3]) != "_exercise" or int(frames[1][2]) != 21:
                raise ProtocolError(f"callee stop lost its caller frame: {frames}")
        data_sequence += 1
        data_request_id += 1

    def execute(command_name: str) -> None:
        nonlocal stop_id, control_sequence, request_id
        _send_packet(
            control,
            _command(
                control_hello,
                control_sequence,
                request_id,
                stop_id,
                command_name,
            ),
        )
        _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
        _receive_envelope(control, "running")
        next_stop = _receive_envelope(control, "stopped")
        if next_stop[8] <= stop_id:
            raise ProtocolError("stop id did not increase during call stepping")
        stop_id = next_stop[8]
        control_sequence += 1
        request_id += 1

    capture("_exercise", 21, False)
    for function_name, function_line in (("first", 9), ("second", 13), ("combine", 17)):
        execute("step")
        capture(function_name, function_line, True)
        execute("step_out")
        capture("_exercise", 21, False)

    execute("next")
    capture("_exercise", 22, False)
    result["call_stepping_stops"] = actual

    _send_packet(
        control,
        _command(control_hello, control_sequence, request_id, stop_id, "continue"),
    )
    _assert_ack(_receive_envelope(control, "ack"), request_id, True, "accepted")
    _receive_envelope(control, "running")


def _serve(listener: socket.socket, result: dict[str, Any], scenario: str) -> None:
    connections: list[socket.socket] = []
    try:
        listener.settimeout(5.0)
        hellos: list[list[Any]] = []
        for _ in range(2):
            connection, _ = listener.accept()
            connections.append(connection)
            connection.settimeout(3.0)
            packet = _receive_packet(connection)
            if not isinstance(packet, list) or len(packet) != 11:
                raise ProtocolError("hello is not a protocol-v2 envelope")
            if packet[:3] != [MAGIC, MAJOR, MINOR]:
                raise ProtocolError("hello has the wrong protocol signature")
            hellos.append(packet)
            _send_packet(connection, _ack_for(packet))

        roles = {hello[9] for hello in hellos}
        channels = {hello[5] for hello in hellos}
        if roles != {"control_hello", "data_hello"}:
            raise ProtocolError(f"unexpected channel roles: {sorted(roles)}")
        if channels != {0, 1}:
            raise ProtocolError(f"unexpected channel ids: {sorted(channels)}")
        if len({(hello[3], hello[4]) for hello in hellos}) != 1:
            raise ProtocolError("control and data channels do not share a session")
        result["hellos"] = hellos
        if scenario == "control":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            _exercise_control(connections[control_index], hellos[control_index], result)
            connections.pop(control_index)
        elif scenario == "timeout":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            _exercise_timeout(connections[control_index], hellos[control_index], result)
        elif scenario in ("reconnect", "data_disconnect"):
            original_session = hellos[0][3]
            original_generation = hellos[0][4]
            if scenario == "data_disconnect":
                data_index = next(index for index, hello in enumerate(hellos) if hello[9] == "data_hello")
                data_connection = connections.pop(data_index)
                data_connection.shutdown(socket.SHUT_RDWR)
                data_connection.close()
            else:
                for connection in connections:
                    connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                connections.clear()
            reconnected: list[list[Any]] = []
            for _ in range(2):
                connection, _ = listener.accept()
                connections.append(connection)
                connection.settimeout(3.0)
                packet = _receive_packet(connection)
                if not isinstance(packet, list) or len(packet) != 11:
                    raise ProtocolError("reconnect hello is not a protocol-v2 envelope")
                reconnected.append(packet)
                _send_packet(connection, _ack_for(packet))
            if {packet[9] for packet in reconnected} != {"control_hello", "data_hello"}:
                raise ProtocolError("reconnect did not restore both channel roles")
            if {packet[3] for packet in reconnected} != {original_session}:
                raise ProtocolError("reconnect changed the session id")
            if {packet[4] for packet in reconnected} != {original_generation + 1}:
                raise ProtocolError("reconnect did not advance the generation")
            result["reconnected_generation"] = original_generation + 1
        elif scenario == "lazy":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            data_index = next(index for index, hello in enumerate(hellos) if hello[9] == "data_hello")
            _exercise_lazy_data(
                connections[control_index],
                connections[data_index],
                hellos[control_index],
                hellos[data_index],
                result,
            )
        elif scenario == "backpressure":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            data_index = next(index for index, hello in enumerate(hellos) if hello[9] == "data_hello")
            _exercise_backpressure(
                connections[control_index],
                connections[data_index],
                hellos[control_index],
                hellos[data_index],
                result,
            )
            connections.pop(control_index)
        elif scenario == "executable_lines":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            data_index = next(index for index, hello in enumerate(hellos) if hello[9] == "data_hello")
            _exercise_executable_lines(
                connections[control_index],
                connections[data_index],
                hellos[control_index],
                hellos[data_index],
                result,
            )
        elif scenario == "call_stepping":
            control_index = next(index for index, hello in enumerate(hellos) if hello[9] == "control_hello")
            data_index = next(index for index, hello in enumerate(hellos) if hello[9] == "data_hello")
            _exercise_call_stepping(
                connections[control_index],
                connections[data_index],
                hellos[control_index],
                hellos[data_index],
                result,
            )
    except BaseException as error:  # Relay thread failures to the test runner.
        result["error"] = error
    finally:
        for connection in connections:
            connection.close()
        listener.close()


def run_test(godot: Path, scenario: str) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    port = listener.getsockname()[1]

    result: dict[str, Any] = {}
    server = threading.Thread(target=_serve, args=(listener, result, scenario), daemon=True)
    server.start()

    with tempfile.TemporaryDirectory(prefix="godot-debug-v2-") as temp_directory:
        project = Path(temp_directory)
        (project / "project.godot").write_text(
            '[application]\nconfig/name="debug-v2-handshake"\n', encoding="utf-8"
        )
        if scenario == "backpressure":
            script = (
                "extends SceneTree\n"
                "var total = 0\n"
                "func _init():\n"
                "    call_deferred(\"_exercise\")\n"
                "func _exercise():\n"
                "    for _index in range(4000):\n"
                "        print(\"x\")\n"
                "    yield(self, \"idle_frame\")\n"
                "    for index in range(1000000):\n"
                "        total += index\n"
                "    quit()\n"
            )
        elif scenario == "lazy":
            script_path = Path(__file__).resolve().parent / "fixtures" / "lazy_variables" / "main.gd"
            script = script_path.read_text(encoding="utf-8")
        elif scenario in ("control", "timeout"):
            script = (
                "extends SceneTree\n"
                "var total = 0\n"
                "func _init():\n"
                "    call_deferred(\"_exercise\")\n"
                "func _exercise():\n"
                "    for index in range(1000000):\n"
                "        total += index\n"
                "    quit()\n"
            )
        elif scenario in ("reconnect", "data_disconnect"):
            script = (
                "extends SceneTree\n"
                "func _init():\n"
                "    call_deferred(\"_wait\")\n"
                "func _wait():\n"
                "    yield(create_timer(0.75), \"timeout\")\n"
                "    quit()\n"
            )
        elif scenario == "executable_lines":
            script_path = Path(__file__).resolve().parent / "fixtures" / "executable_lines" / "main.gd"
            script = script_path.read_text(encoding="utf-8")
        elif scenario == "call_stepping":
            script_path = Path(__file__).resolve().parent / "fixtures" / "call_stepping" / "main.gd"
            script = script_path.read_text(encoding="utf-8")
        else:
            script = "extends SceneTree\nfunc _init():\n    quit()\n"
        (project / "main.gd").write_text(script, encoding="utf-8")
        command = [
            str(godot),
            "--path",
            str(project),
            "--remote-debug",
            f"127.0.0.1:{port}",
        ]
        if scenario == "executable_lines":
            command.extend(["--breakpoints", "res://main.gd:9"])
        elif scenario == "call_stepping":
            command.extend(["--breakpoints", "res://main.gd:21"])
        command.extend(["--script", "main.gd"])
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=15.0, check=False
        )

    server.join(timeout=6.0)
    if server.is_alive():
        raise ProtocolError("fake server did not finish")
    if "error" in result:
        raise ProtocolError(
            f"{result['error']}\nGodot stdout:\n{completed.stdout}\n"
            f"Godot stderr:\n{completed.stderr}"
        ) from result["error"]
    if completed.returncode != 0:
        raise ProtocolError(
            f"Godot exited with {completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )

    session_id = result["hellos"][0][3]
    if scenario == "control":
        recovery_msec = (time.perf_counter() - result["control_closed_at"]) * 1000.0
        if recovery_msec >= 1250.0:
            raise ProtocolError(f"control disconnect recovery took {recovery_msec:.2f} ms")
        print(
            "debug protocol v2 execution control: PASS "
            f"(ack_p95={result['ack_p95_msec']:.2f} ms, recovery={recovery_msec:.2f} ms)"
        )
    elif scenario == "timeout":
        print(
            "debug protocol v2 heartbeat recovery: PASS "
            f"(recovery={result['timeout_recovery_msec']:.2f} ms)"
        )
    elif scenario == "reconnect":
        print(
            "debug protocol v2 nonblocking reconnect: PASS "
            f"(generation={result['reconnected_generation']})"
        )
    elif scenario == "data_disconnect":
        print(
            "debug protocol v2 data-channel recovery: PASS "
            f"(generation={result['reconnected_generation']})"
        )
    elif scenario == "lazy":
        print(
            "debug protocol v2 lazy variables: PASS "
            f"(stack={result['stack_msec']:.2f} ms, page={result['array_page_msec']:.2f} ms)"
        )
    elif scenario == "backpressure":
        recovery_msec = (time.perf_counter() - result["control_closed_at"]) * 1000.0
        if recovery_msec >= 1250.0:
            raise ProtocolError(f"pressure disconnect recovery took {recovery_msec:.2f} ms")
        print(
            "debug protocol v2 backpressure: PASS "
            f"(ack_p95={result['pressure_ack_p95_msec']:.2f} ms, "
            f"stop_p95={result['pressure_stop_p95_msec']:.2f} ms, "
            f"dropped={result['pressure_dropped']})"
        )
    elif scenario == "executable_lines":
        print(
            "debug protocol v2 executable-line stepping: PASS "
            f"(stops={result['executable_line_stops']})"
        )
    elif scenario == "call_stepping":
        print(
            "debug protocol v2 same-line call stepping: PASS "
            f"(stops={result['call_stepping_stops']})"
        )
    else:
        print(f"debug protocol v2 dual-channel handshake: PASS (session={session_id})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True, type=Path)
    parser.add_argument(
        "--scenario",
        choices=("handshake", "control", "timeout", "reconnect", "data_disconnect", "lazy", "backpressure", "executable_lines", "call_stepping"),
        default="handshake",
    )
    arguments = parser.parse_args()
    run_test(arguments.godot.resolve(), arguments.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
