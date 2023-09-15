func test():
    print("Simple usage:")
    if true:
        var { foo = var foo, var ..bar } = { foo = 123, bar = 1234 }
        print(foo) # Should print "123"
        print(bar) # Should print "{ "bar": 1234 }"

    print("Complex usage:")
    if true:
        var { foo = [_, { bar = var bar }] } = { foo = [{ bar = 123 }, { bar = "hello!" }] }
        print(bar) # Should print "hello!"

    print("Typed usage:")
    if true:
        var { foo = var foo: int, bar = var bar: Vector2 } = { foo = 123, bar = Vector2.ZERO }
        print(foo) # Should print "123"
        print(bar) # Should print "(0, 0)"

    print("Empty slot usage:")
    if true:
        var { foo = _, var ..bar } = { foo = 123, bar = 1234 }
        print(bar) # Should print "{ "bar": 1234 }"

    print("Multiple lines usage:")
    if true:
        var {
            userId = var user_id,
            id = var id,
            title = var title,
            completed = var completed,
        } = { userId = 1, id = 1, title = "Hi!", completed = "yep" }
        print(user_id) # Should print "1"
        print(id) # Should print "1"
        print(title) # Should print "Hi!"
        print(completed) # Should print "yep"

