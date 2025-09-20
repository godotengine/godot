class BaseClass:
    func base_method():
        pass

class ChildClass extends BaseClass:
    @override
    func base_method():
        pass

class NativeMethodOverride extends Node:

    @override
    func _physics_process(delta: float) -> void:
        pass

    @override
    func doesnt_override_anything():
        pass
