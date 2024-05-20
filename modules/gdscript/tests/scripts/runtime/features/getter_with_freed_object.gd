# https://github.com/godotengine/godot/issues/68184

var node: Node:
	get:
		return node
	set(n):
		node = n


func test():
	node = Node.new()
	node.free()

	if !is_instance_valid(node):
		print("It is freed")
