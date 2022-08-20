class A:
    var x = 3

class B:
    var x = 4

class C:
    var x = 5

class Test:
    var a = A.new()
    var b: B = B.new()
    var c := C.new()

func test():
    var test_instance := Test.new()
    prints(test_instance.a.x)
    prints(test_instance.b.x)
    prints(test_instance.c.x)
