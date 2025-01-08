enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }

var class_var: int = MyEnum.ENUM_VALUE_1

func test():
	print(class_var)
	class_var = MyEnum.ENUM_VALUE_2
	print(class_var)

	var local_var: int = MyEnum.ENUM_VALUE_1
	print(local_var)
	local_var = MyEnum.ENUM_VALUE_2
	print(local_var)
