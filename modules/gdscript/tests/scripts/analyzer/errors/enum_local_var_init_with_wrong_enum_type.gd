enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2 }

func test():
	var local_var: MyEnum = MyOtherEnum.OTHER_ENUM_VALUE_1
	print(local_var)
