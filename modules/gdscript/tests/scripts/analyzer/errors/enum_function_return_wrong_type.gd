enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2 }

func enum_func() -> MyEnum:
	return MyOtherEnum.OTHER_ENUM_VALUE_1

func test():
	print(enum_func())
