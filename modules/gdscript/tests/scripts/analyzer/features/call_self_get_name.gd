extends Node

func test():
	set_name("TestNodeName")
	if get_name() == &"TestNodeName":
		print("Name is equal")
	else:
		print("Name is not equal")
	print(get_name() is StringName)
