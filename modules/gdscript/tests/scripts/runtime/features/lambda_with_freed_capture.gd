# https://github.com/godotengine/godot/issues/79707

func test():
	var node := Node.new()
	var lambda = func(): print(node)
	node.free()
	lambda.call()
