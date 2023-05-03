func test():
    var ok := 0
    var foo: int? = null
    if foo == null:
        print("it's null!")
        ok += 1

    foo = 123
    if foo != null:
        print(-foo)
        ok += 1

    if ok == 2:
        print("Ok")
