class Foo:
    private func foo():
        pass

class Bar extends Foo:
    func bar():
        var t = foo
        t.call()

func test():
    pass
