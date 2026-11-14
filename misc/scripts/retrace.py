#!/usr/bin/env python3

import argparse
import base64
import os
import struct
import subprocess


def run_addr2line(platform, cmd):
    if platform == "windows":
        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, os.path.join(os.getenv("HOME"), ".cargo/bin/addr2line"))
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass

        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, f"{arch}-w64-mingw32-addr2line")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass

        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, "addr2line")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass
    elif platform == "windows-msvc":
        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, "pdb-addr2line")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass
    elif platform == "macos":
        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, "atos")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass
    elif platform == "linuxbsd":
        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, os.path.join(os.getenv("HOME"), ".cargo/bin/addr2line"))
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass

        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, "llvm-addr2line")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass

        try:
            try_cmd = cmd.copy()
            try_cmd.insert(0, "addr2line")
            return subprocess.check_output(try_cmd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            pass

    return "???"


parser = argparse.ArgumentParser()
parser.add_argument("executable_path", help="Path to an executable or debug symbol file.")
parser.add_argument("trace_data", help="Base64-encoded stack trace data to symbolicate.")
args = parser.parse_args()

sym_data = base64.b64decode(args.trace_data)

offset = 0

platform_len = struct.unpack_from("=b", sym_data, offset)[0]
offset += 1
platform = struct.unpack_from(f"={platform_len}s", sym_data, offset)[0].decode("utf-8")
offset += platform_len

arch_len = struct.unpack_from("=b", sym_data, offset)[0]
offset += 1
arch = struct.unpack_from(f"={arch_len}s", sym_data, offset)[0].decode("utf-8")
offset += arch_len

mod_count = struct.unpack_from("=b", sym_data, offset)[0]
offset += 1

modules = []
load_addr = 0
for i in range(mod_count):
    mod_offset = struct.unpack_from("=Q", sym_data, offset)[0]
    offset += 8
    str_len = struct.unpack_from("=H", sym_data, offset)[0]
    offset += 2
    str = struct.unpack_from(f"={str_len}s", sym_data, offset)[0].decode("utf-8")
    offset += str_len
    modules.append({"name": str, "offset": mod_offset})
    if str == "main":
        load_addr = mod_offset

sym_count = struct.unpack_from("=b", sym_data, offset)[0]
offset += 1

base_cmd = []
if platform == "macos":
    base_cmd = ["-o", args.executable_path, "--arch", arch, "--fullPath", "-l", hex(load_addr)]
elif platform == "windows-msvc":
    base_cmd = ["-e", args.executable_path.replace(".exe", ".pdb"), "-f", "-p", "-C"]
else:
    base_cmd = ["-e", args.executable_path, "-f", "-p", "-C"]

prev_offest = offset
symbol = ""
cmd = base_cmd.copy()
for i in range(sym_count):
    sym_offset = struct.unpack_from("=Q", sym_data, offset)[0]
    offset += 8
    mod_idx = struct.unpack_from("=b", sym_data, offset)[0]
    offset += 1

    if platform == "macos":
        cmd.append(hex(sym_offset))
    elif platform == "windows":
        cmd.append(hex(sym_offset - load_addr + 0x140000000))
    elif platform == "linuxbsd" or platform == "windows-msvc":
        cmd.append(hex(sym_offset - load_addr))

symbols = run_addr2line(platform, cmd)

demangled_symbols = symbols.splitlines()

if len(demangled_symbols) != sym_count:
    # Symbolication failed, try processing each address individually.
    demangled_symbols = []
    offset = prev_offest
    for i in range(sym_count):
        sym_offset = struct.unpack_from("=Q", sym_data, offset)[0]
        offset += 8
        mod_idx = struct.unpack_from("=b", sym_data, offset)[0]
        offset += 1

        cmd = base_cmd.copy()
        if platform == "macos":
            cmd.append(hex(sym_offset))
        elif platform == "windows":
            cmd.append(hex(sym_offset - load_addr + 0x140000000))
        elif platform == "linuxbsd" or platform == "windows-msvc":
            cmd.append(hex(sym_offset - load_addr))

        symbols = run_addr2line(platform, cmd)
        demangled_symbols.append(symbols.splitlines()[0])

print("================================================================")
print("Load address:", hex(load_addr))
print("================================================================")

offset = prev_offest
for i in range(sym_count):
    sym_offset = struct.unpack_from("=Q", sym_data, offset)[0]
    offset += 8
    mod_idx = struct.unpack_from("=b", sym_data, offset)[0]
    offset += 1

    print(
        i + 1,
        hex(sym_offset),
        "(",
        modules[mod_idx]["name"],
        "+",
        hex(sym_offset - modules[mod_idx]["offset"]),
        ") -",
        demangled_symbols[i],
    )

print("================================================================")
print("-- END OF C++ BACKTRACE --")
print("================================================================")
