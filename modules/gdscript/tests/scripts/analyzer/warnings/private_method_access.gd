func test():
    var instance := TestClass.new()
    instance.public()
    instance._private()


class TestClass:

    func public():
        pass
    
    func _private():
        pass
