func test():
    var foo: int? = null
    (foo if true else 0) + 1 # Should error