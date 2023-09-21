# https://github.com/godotengine/godot/issues/73273

func other(callable : Callable):
    callable.call()

func four_parameters(_a, callable : Callable, b=func(): print(10)):
    callable.call()
    b.call()

func test():
    var v
    v=func():v=1
    if true: v=1
    print(v)
    print()

    v=func(): print(2) if false else print(3)
    @warning_ignore("unsafe_cast")
    (v as Callable).call()
    print()

    v=func():
        print(4)
        print(5)
    @warning_ignore("unsafe_cast")
    if true: (v as Callable).call()
    print()

    @warning_ignore("unsafe_call_argument")
    other(v)
    print()

    other(func(): print(6))
    print()

    other(func():
        print(7)
        print(8)
    )
    print()

    four_parameters(1,func():print(9))
    four_parameters(1,func():print(9), func(): print(11))
    four_parameters(1,func():
        print(12)
        print(13)
    , func(): print(11))
    print()

    from_ticket()

func from_ticket():
    var _v
    if true: _v = (func(): test())
    if true: _v = (func(): test())
    if true: _v = (func(): test())

    if true: _v = func(): test()
    if true: _v = func(): test()
    print(14)
