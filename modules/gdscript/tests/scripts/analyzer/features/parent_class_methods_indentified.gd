extends Node

class A extends Node:
    func f1():
        pass

class B extends A:
    signal test_signal()

    func f2():
        pass

    func test_connect():
        test_signal.connect(f1)
        test_signal.connect(f2)

func test():
    var inst = B.new()
    add_child(inst)
    inst.test_connect()
