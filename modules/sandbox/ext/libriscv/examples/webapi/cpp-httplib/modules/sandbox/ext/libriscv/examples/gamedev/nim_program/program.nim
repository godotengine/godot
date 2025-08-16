import api
import std/json

api.print("Hello Nim World!\n")

var i = api.dyncall1(0x12345678)
api.print("i = " & $i & "\n")

var hisName = "John"
let herAge = 31
var j = %*
  [
    { "name": hisName, "age": 30 },
    { "name": "Susan", "age": herAge }
  ]

var j2 = %* {"name": "Isaac", "books": ["Robot Dreams"]}
j2["details"] = %* {"age":35, "pi":3.1415}
echo j2

api.exit(0)

proc test1(a: int, b: int, c: int, d: int): int {.exportc.} =
    api.print("test1 called with: " & $a & " " & $b & " " & $c & " " & $d & "\n")
    return a + b + c + d

proc test2() {.exportc.} =
    var data = alloc(1024)
    dealloc(data)
    return

proc test3(str: cstring) {.exportc.} =
    try:
        raise newException(IOError, $str)
    except IOError as e:
        api.print("Test3 called with " & e.msg & "\n")

type
    Data* = object
        a: int32
        b: int32
        c: int32
        d: int32
        e: float32
        f: float32
        g: float32
        h: float32
        i: float64
        j: float64
        k: float64
        l: float64
        buffer: array[32, char]

proc test4(d: Data) {.exportc.} =
    var str = cast[cstring](addr d.buffer[0])
    api.print("test4 called with: " & $d.a & " " & $d.b & " " & $d.c & " " & $d.d & " " & $d.e & " " & $d.f & " " & $d.g & " " & $d.h & " " & $d.i & " " & $d.j & " " & $d.k & " " & $d.l & " " & $str & "\n")
    return

proc test5() {.exportc.} =
    var data : MyData
    data.buffer[0..20] = "Hello Buffered World!".toOpenArray(0, 20)
    api.dyncall4(addr data, 1, addr data)

proc remote_debug_test() {.exportc.} =
    api.print("Remote Debug Test\n")
    api.print("1\n")
    api.print("2\n")
    api.print("3\n")
    api.print("4\n")
    api.print("5\n")

proc bench_dyncall_overhead() {.exportc.} =
    api.dyncall3()
