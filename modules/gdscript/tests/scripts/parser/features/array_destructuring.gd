func test():
    print("Simple usage:")
    if true:
        var [var foo, var ..bar] = [123, "1234"]
        print(foo) # Should print "123"
        print(bar) # Should print "["1234"]"

    print("Complex usage:")
    if true:
        var [var a, var b, ..[var c, var d, ..[var e, var f]]] = [1, 2, 3, 4, 5, 6]
        prints(a, b, c, d, e, f) # Should print "1 2 3 4 5 6"

    print("Typed usage:")
    if true:
        var [var foo: int, var bar: String] = [123, "1234"]
        print(foo) # Should print "123"
        print(bar) # Should print "1234"

    print("Empty slot usage:")
    if true:
        var [_, var bar: String] = [123, "1234"]
        print(bar) # Should print "1234"

    print("Multiple lines usage:")
    if true:
        var [
            var foo: int,
            var bar: String
        ] = [123, "1234"]
        print(foo) # Should print "123"
        print(bar) # Should print "1234"

