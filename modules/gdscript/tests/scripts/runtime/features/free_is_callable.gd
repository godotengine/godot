func test():
	var node := Node.new()
	var callable: Callable = node.free
	callable.call()
	print(node)

	node = Node.new()
	callable = node["free"]
	callable.call()
	print(node)
