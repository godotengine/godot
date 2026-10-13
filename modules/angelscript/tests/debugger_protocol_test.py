"""Real TCP ScriptDebugger protocol smoke test; requires a built editor."""
import argparse
import json
import pathlib
import socket
import struct
import subprocess
import time


def encode(value):
    if isinstance(value, bool):
        return struct.pack('<II', 1, int(value))
    if isinstance(value, int):
        return struct.pack('<Iq', 2 | (1 << 16), value)
    if isinstance(value, str):
        raw = value.encode()
        return struct.pack('<II', 4, len(raw)) + raw + bytes((-len(raw)) % 4)
    if isinstance(value, list):
        return struct.pack('<II', 28, len(value)) + b''.join(encode(v) for v in value)
    raise TypeError(type(value))


def decode(raw, pos=0):
    header, = struct.unpack_from('<I', raw, pos)
    pos += 4
    kind = header & 0xffff
    wide = header & (1 << 16)
    if kind == 0:
        return None, pos
    if kind in (1, 2):
        fmt = '<q' if wide else '<i'
        value, = struct.unpack_from(fmt, raw, pos)
        return (bool(value) if kind == 1 else value), pos + (8 if wide else 4)
    if kind == 3:
        value, = struct.unpack_from('<d' if wide else '<f', raw, pos)
        return value, pos + (8 if wide else 4)
    if kind in (4, 21, 22):
        size, = struct.unpack_from('<I', raw, pos)
        pos += 4
        return raw[pos:pos + size].decode(), pos + size + (-size) % 4
    if kind == 34:  # PackedStringArray, used by debugger output.
        count, = struct.unpack_from('<I', raw, pos)
        pos += 4
        result = []
        for _ in range(count):
            size, = struct.unpack_from('<I', raw, pos)
            pos += 4
            result.append(raw[pos:pos + size].rstrip(b'\0').decode())
            pos += size + (-size) % 4
        return result, pos
    if kind in (29, 30, 31, 32, 33):
        count, = struct.unpack_from('<I', raw, pos)
        pos += 4
        fmt, width = {29: ('B', 1), 30: ('i', 4), 31: ('q', 8), 32: ('f', 4), 33: ('d', 8)}[kind]
        return list(struct.unpack_from('<' + fmt * count, raw, pos)), pos + count * width + (-(count * width)) % 4
    if kind == 28:
        count, = struct.unpack_from('<I', raw, pos)
        pos += 4
        result = []
        for _ in range(count & 0x7fffffff):
            value, pos = decode(raw, pos)
            result.append(value)
        return result, pos
    if kind == 27:
        count, = struct.unpack_from('<I', raw, pos)
        pos += 4
        result = {}
        for _ in range(count & 0x7fffffff):
            key, pos = decode(raw, pos)
            value, pos = decode(raw, pos)
            result[str(key)] = value
        return result, pos
    raise ValueError(f'unsupported variant kind {kind}')


def receive_exact(peer, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = peer.recv(size - len(chunks))
        if not part:
            raise EOFError('debug peer disconnected')
        chunks.extend(part)
    return chunks


def receive(peer):
    size, = struct.unpack('<I', receive_exact(peer, 4))
    raw = receive_exact(peer, size)
    result, used = decode(raw)
    assert used == size
    return result


def send(peer, name, thread, values):
    raw = encode([name, thread, values])
    peer.sendall(struct.pack('<I', len(raw)) + raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project = pathlib.Path(__file__).with_name('debugger_project').resolve()
    records = []
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        server.settimeout(30)
        port = server.getsockname()[1]
        # No pipes: output inherited and the DSH Windows confinement does not
        # require subprocess named-pipe support.
        process = subprocess.Popen([args.editor, '--headless', '--path', str(project),
                                    '--remote-debug', f'tcp://127.0.0.1:{port}',
                                    '--breakpoints', 'res://main.as:9', '--quit-after', '4'])
        try:
            peer, _ = server.accept()
            with peer:
                peer.settimeout(25)
                breaks = 0
                commands = ['step', 'out', 'next', 'continue']
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        name, thread, data = receive(peer)
                    except EOFError:
                        break
                    if name in ('debug_enter', 'debug_exit', 'stack_dump', 'stack_frame_vars', 'stack_frame_var', 'output', 'error'):
                        records.append({'name': name, 'thread': thread, 'data': data})
                    if name == 'debug_enter':
                        assert data[0] is True and data[2] is True, data
                        breaks += 1
                        send(peer, 'get_stack_dump', thread, [])
                        send(peer, 'get_stack_frame_vars', thread, [0])
                    elif name == 'stack_dump':
                        send(peer, commands[min(breaks - 1, len(commands) - 1)], thread, [])
                assert breaks == 4, f'expected 4 resumable stops, got {breaks}'
            process.wait(timeout=20)
            assert process.returncode == 0, process.returncode
            stacks = [r['data'] for r in records if r['name'] == 'stack_dump']
            assert [s[2] for s in stacks] == [9, 5, 10, 11], stacks
            assert [s[3] for s in stacks] == ['_ready', 'child', '_ready', '_ready'], stacks
            variables = [r['data'] for r in records if r['name'] == 'stack_frame_var']
            assert any(v[0] == 'x' and v[1] == 0 and v[3] == 4 for v in variables), variables
            assert any(v[0] == 'total' and v[1] == 2 and v[3] == 2 for v in variables), variables
            assert sum(r['name'] == 'debug_exit' for r in records) == 4
            assert any(r['name'] == 'output' and any(text.strip() == '6' for text in r['data'][0]) for r in records), records
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            pathlib.Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: real TCP breakpoints/step/out/next/continue; trace:', args.output)


if __name__ == '__main__':
    main()
