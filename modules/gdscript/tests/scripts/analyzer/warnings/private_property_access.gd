func test():
    var instance := TestClass.new()
    instance.public += 1
    instance._private += 1


class TestClass:

    var public := 1
    var _private := 1