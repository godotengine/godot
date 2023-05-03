class B:
    func hey():
        pass

class A:
    var b: B?

func test():
    var foo: A = A.new()
    foo.b.hey()