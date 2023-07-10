func test():
    var foo: int? = null
    if foo: # narrow
        var bar: int = foo # explicit
        bar = foo
        print(bar)
