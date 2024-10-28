@private

class Foo:
    func foo():
        pass

class Bar extends Foo:
    func bar():
        foo()

func test():
    pass
