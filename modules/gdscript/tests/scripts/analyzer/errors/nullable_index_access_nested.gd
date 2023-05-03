class B:
    var bar

class A:
    var foo: B?

func test():
    var foobar: Array[A?] = [A.new()]
    foobar[0].bar