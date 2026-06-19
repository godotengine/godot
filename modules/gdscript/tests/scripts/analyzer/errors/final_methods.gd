class A:
    @final func foo():
        pass

    @final func bar():
        pass

class B extends A:
    func foo():
        pass

class C extends B:
    func bar():
        pass

func test():
    pass
