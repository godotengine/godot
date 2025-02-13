enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2 }

var class_var: MyEnum = MyOtherEnum.OTHER_ENUM_VALUE_1 as MyEnum

func test():
	print(class_var)
	class_var = MyOtherEnum.OTHER_ENUM_VALUE_2 as MyEnum
	print(class_var)

	var local_var: MyEnum = MyOtherEnum.OTHER_ENUM_VALUE_1 as MyEnum
	print(local_var)
	local_var = MyOtherEnum.OTHER_ENUM_VALUE_2 as MyEnum
	print(local_var)
