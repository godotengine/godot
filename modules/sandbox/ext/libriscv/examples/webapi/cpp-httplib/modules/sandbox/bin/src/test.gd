extends Node

@export var sandbox : Sandbox_SrcExample

func _ready() -> void:
	print(sandbox.my_function(Vector4(1, 1, 1, 1)))
	print(sandbox.my_function2("Hello Sandboxed World!", ["An array"]))
