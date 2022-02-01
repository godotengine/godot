extends Node

class A extends Node:
    signal test_signal_a()

class B extends A:
    signal test_signal_b()

    func f1():
        pass

    func test_connect():
        test_signal_a.connect(f1)
        test_signal_b.connect(f1)

func test():
    var inst = B.new()
    add_child(inst)
    inst.test_connect()
