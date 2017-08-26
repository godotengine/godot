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
    def iteritems(d):
        return d.iteritems()
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
    def iteritems(d):
        return iter(d.items())
