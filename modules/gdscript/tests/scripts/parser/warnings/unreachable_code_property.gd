var test_property: int:
	get:
		return 0
		print("unreachable")
	set(value):
		test_property = value
		return
		print("unreachable")


func test():
	pass
