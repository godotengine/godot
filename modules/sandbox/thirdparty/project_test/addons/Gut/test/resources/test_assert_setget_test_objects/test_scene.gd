extends Node2D


@onready var node_with_setter_getter = $NodeChildMock :
	get:
		return node_with_setter_getter
	set(node):
		if node_with_setter_getter != null:
			node_with_setter_getter.queue_free()
		node_with_setter_getter = node
