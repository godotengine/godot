@private

class Foo:
    func foo():
        pass

class Bar extends Foo:
    func bar():
        self.foo()

func test():
    pass
