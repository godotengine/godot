class CustomRefCounted extends RefCounted:
	@oninstantiated var wrong_base_class

class CustomNode extends Node:
	@oninstantiated static var oninstantiated_static
	@oninstantiated @oninstantiated var duplicate_oninstantiated
	@onready @oninstantiated var onready_oninstantiated
	@oninstantiated @onready var oninstantiated_onready

func test():
	pass
