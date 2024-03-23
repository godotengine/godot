enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }
enum MyOtherEnum { OTHER_ENUM_VALUE_1, OTHER_ENUM_VALUE_2 }

func enum_func(e: MyEnum) -> void:
	print(e)

func test():
	enum_func(MyOtherEnum.OTHER_ENUM_VALUE_1)
