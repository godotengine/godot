func test():
    var foo: Array[int]? = null
    foo = [1]
    if foo is Array:
        print(foo[0])