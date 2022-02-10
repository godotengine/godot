enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }

var class_var: MyEnum = 0 as MyEnum

func test():
	print(class_var)
	class_var = 1 as MyEnum
	print(class_var)

	var local_var: MyEnum = 0 as MyEnum
	print(local_var)
	local_var = 1 as MyEnum
	print(local_var)
