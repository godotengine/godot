extends RefCounted

func test():
	var nope := $Node
	print("Cannot use $ without a Node base")
