func typed(node: Node) -> void:
	print(node)

func test():
	var node := Node.new()
	node.free()
	typed(node)
