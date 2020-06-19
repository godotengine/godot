import sys

if sys.version_info < (3,):

    def isbasestring(s):
        return isinstance(s, basestring)

    def open_utf8(filename, mode):
        return open(filename, mode)

    def byte_to_str(x):
        return str(ord(x))

    import cStringIO

    def StringIO():
        return cStringIO.StringIO()

    def encode_utf8(x):
        return x

    def decode_utf8(x):
        return x

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    def escape_string(s):
        if isinstance(s, unicode):
            s = s.encode("ascii")
        result = ""
        for c in s:
            if not (32 <= ord(c) < 127) or c in ("\\", '"'):
                result += "\\%03o" % ord(c)
            else:
                result += c
        return result

    def qualname(obj):
        # Not properly equivalent to __qualname__ in py3, but it doesn't matter.
        return obj.__name__


else:

    def isbasestring(s):
        return isinstance(s, (str, bytes))

    def open_utf8(filename, mode):
        return open(filename, mode, encoding="utf-8")

    def byte_to_str(x):
        return str(x)

    import io

    def StringIO():
        return io.StringIO()

    import codecs

    def encode_utf8(x):
        return codecs.utf_8_encode(x)[0]

    def decode_utf8(x):
        return codecs.utf_8_decode(x)[0]

    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    def charcode_to_c_escapes(c):
        rev_result = []
        while c >= 256:
            c, low = (c // 256, c % 256)
            rev_result.append("\\%03o" % low)
        rev_result.append("\\%03o" % c)
        return "".join(reversed(rev_result))

    def escape_string(s):
        result = ""
        if isinstance(s, str):
            s = s.encode("utf-8")
        for c in s:
            if not (32 <= c < 127) or c in (ord("\\"), ord('"')):
                result += charcode_to_c_escapes(c)
            else:
                result += chr(c)
        return result

    def qualname(obj):
        return obj.__qualname__
