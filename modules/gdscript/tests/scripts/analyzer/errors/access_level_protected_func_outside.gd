@private

class Foo:
    protected func foo():
        pass

func test():
    Foo.new().foo()
