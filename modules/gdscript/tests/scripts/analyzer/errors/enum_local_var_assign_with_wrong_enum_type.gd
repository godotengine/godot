enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2 }

func test():
	var local_var: MyEnum = MyEnum.ENUM_VALUE_1
	print(local_var)
	local_var = MyOtherEnum.OTHER_ENUM_VALUE_2
	print(local_var)
