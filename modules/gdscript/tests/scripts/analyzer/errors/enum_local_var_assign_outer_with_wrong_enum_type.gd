enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }

class InnerClass:
	enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }

func test():
	var local_var: MyEnum = MyEnum.ENUM_VALUE_1
	print(local_var)
	local_var = InnerClass.MyEnum.ENUM_VALUE_2
	print(local_var)
