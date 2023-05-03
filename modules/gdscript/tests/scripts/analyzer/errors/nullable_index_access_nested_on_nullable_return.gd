class A:
    var foo := A.new()
    func nullable_return() -> String?:
        return null

func test():
    var foobar := A.new()
    foobar.foo.nullable_return()[0]