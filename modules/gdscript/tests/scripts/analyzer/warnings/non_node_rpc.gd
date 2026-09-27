extends Object

@rpc
func test():
    pass

class InnerClass extends RefCounted:
    @rpc
    func test():
        pass

class ResourceClass extends Resource:
    @rpc
    func test():
        pass
