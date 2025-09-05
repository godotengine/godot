class_name OverBaseClass
func base_method():
    pass

class InnerClass:
    func inner_base_method():
        pass

class ChildClass extends OverBaseClass:
    func base_method():
        pass

class ChildClass2 extends InnerClass:
    func inner_base_method():
        pass

class NativeMethodOverride extends Node:

    func _physics_process(_delta: float) -> void:
        pass

func test():
    print("@override_warns")
