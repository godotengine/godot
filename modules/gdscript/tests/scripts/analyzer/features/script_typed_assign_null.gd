extends Node

class LocalClass extends Node:
	pass

func test():
	var typed: LocalClass = get_node_or_null("does_not_exist")
	var untyped = null
	var node_1: LocalClass = typed
	var node_2: LocalClass = untyped
	var node_3 = typed
	var node_4 = untyped
	print(typed)
	print(untyped)
	print(node_1)
	print(node_2)
	print(node_3)
	print(node_4)
