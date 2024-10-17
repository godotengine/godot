struct TestStruct:
	var test_int : int = 1
	var test_float : float = 2.5
	var test_dict : Dictionary = {4 : 5}
	var test_array : Array = [3, "Godot"]

func test():
	print(TestStruct())
	print(TestStruct(3))
	print(TestStruct(3, 1.5))
	print(TestStruct(3, 1.5, {"key": "val"}))
	print(TestStruct(3, 1.5, {"key": "val"}, [0.5]))

	print(PropertyInfo())
	print(Object.PropertyInfo())
	# print(self.TestStruct())
