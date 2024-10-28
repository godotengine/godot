@private

class Foo:
    internal func foo():
        pass

func test():
    Foo.new().foo()
