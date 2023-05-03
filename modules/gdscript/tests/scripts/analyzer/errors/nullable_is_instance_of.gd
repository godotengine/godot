func test():
    var foo: Array[int]? = null
    foo = [1]
    if is_instance_of(foo, TYPE_NIL):
        print(foo[0])