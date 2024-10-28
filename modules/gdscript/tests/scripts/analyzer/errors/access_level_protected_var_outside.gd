@private

class Foo:
    protected var a = 1
    func foo():
        pass

func test():
    print(Foo.new().a)
