enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2, OTHER_ENUM_VALUE_3 }

func test():
	print(MyOtherEnum.OTHER_ENUM_VALUE_3 as MyEnum)
