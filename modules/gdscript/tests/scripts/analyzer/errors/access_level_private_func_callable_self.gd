class Foo:
    private func foo():
        pass

class Bar extends Foo:
    func bar():
        var t = self.foo
        t.call()

func test():
    pass
