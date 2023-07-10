func test():
    var foo: int? = 1
    var result := (foo if foo else 123) + 1 # Should be okay
    print(result)
