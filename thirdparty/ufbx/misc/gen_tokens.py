import os
import re
import itertools

script_path = os.path.dirname(__file__)
path = os.path.join(script_path, "..", "ufbx_write.c")

tokens = { "": "TOKEN_EMPTY" }

def to_token(s):
    return re.sub(r"[\s|]+", "_", s)

def collect_line(line):
    for m in re.finditer(r"(?<![\"A-Z])T\"([^\"]*)\"", line):
        s = m.group(1)
        if s not in tokens:
            tokens[s] = to_token(s)

def collect_lines(lines):
    it = iter(lines)

    for line in it:
        if line == "static const ufbxw_string ufbxwi_tokens[] = {":
            break
        else:
            collect_line(line)

    for line in it:
        if line == "};":
            break
        else:
            m = re.search(r"\"([^\"]+)\"", line)
            if m:
                s = m.group(1)
                if s not in tokens:
                    tokens[s] = to_token(s)

    for line in it:
        collect_line(line)

with open(path, "rt", encoding="utf-8") as f:
    raw_lines = f.readlines()

num_clrf = sum(1 for l in raw_lines if l.endswith("\r\n"))
line_end = "\r\n" if num_clrf >= len(raw_lines) // 2 else "\n"

lines = [l.rstrip() for l in raw_lines]
collect_lines(lines)

token_order = sorted(tokens.items())

def filter_line(line):
    def replacer(m):
        s = m.group(1)
        return f"UFBXWI_{tokens[s]}"

    return re.sub(r"(?<![\"A-Z])T\"([^\"]*)\"", replacer, line)

def filter_lines(lines):
    it = iter(lines)

    for line in it:
        if line == "typedef enum ufbxwi_token {":
            yield line
            break
        else:
            yield filter_line(line)

    yield "\tUFBXWI_TOKEN_NONE,"
    for _, token in token_order:
        yield f"\tUFBXWI_{token},"
    yield "\tUFBXWI_TOKEN_COUNT,"
    yield "\tUFBXWI_TOKEN_FORCE_32BIT = 0x7fffffff,"

    for line in it:
        if line == "} ufbxwi_token;":
            yield line
            break

    for line in it:
        if line == "static const ufbxw_string ufbxwi_tokens[] = {":
            yield line
            break
        else:
            yield filter_line(line)

    for s, _ in token_order:
        yield f"\t{{ \"{s}\", {len(s)} }},"

    for line in it:
        if line == "};":
            yield line
            break

    for line in it:
        yield filter_line(line)

new_lines = filter_lines(lines)

with open(path, "wt", encoding="utf-8") as f:
    f.writelines(l + line_end for l in new_lines)

print(f"Found {len(tokens)} tokens")
