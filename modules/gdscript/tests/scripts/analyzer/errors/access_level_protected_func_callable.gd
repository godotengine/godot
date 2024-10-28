class Foo:
    protected func foo():
        pass

func test():
    var f := Foo.new()
    var t = f.foo
    t.call()
