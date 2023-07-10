class B:
    func hey():
        pass

class A:
    var b: B?

func test():
    var foo: Array[A?] = [A.new()]
    foo[0].hey()