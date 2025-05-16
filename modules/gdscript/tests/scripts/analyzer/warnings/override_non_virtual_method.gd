class Parent:
    func test():
        pass

class ParentSafe:
    @virtual func test():
        pass

class Child extends Parent:
    func test():
        pass

class ChildSafe extends ParentSafe:
    func test():
        pass

class ChildCompletelySafe extends ParentSafe:
    @override func test():
        pass


class ChildOverrideInexistentMethod extends ParentSafe:
    @override func test_():
        pass


func test():
    pass
