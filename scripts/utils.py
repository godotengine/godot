def escape_string(s: str) -> str:
    def charcode_to_c_escapes(c):
        rev_result = []
        while c >= 256:
            c, low = (c // 256, c % 256)
            rev_result.append("\\%03o" % low)
        rev_result.append("\\%03o" % c)
        return "".join(reversed(rev_result))

    result = ""
    if isinstance(s, str):
        s = s.encode("utf-8")
    for c in s:
        if not (32 <= c < 127) or c in (ord("\\"), ord('"')):
            result += charcode_to_c_escapes(c)
        else:
            result += chr(c)
    return result


def forward_slashes(s: str) -> str:
    return s.replace('\\', '/')
