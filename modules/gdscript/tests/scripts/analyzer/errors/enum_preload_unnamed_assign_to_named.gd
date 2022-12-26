enum MyEnum { VALUE_A, VALUE_B, VALUE_C = 42 }

func test():
	const P = preload("../features/enum_value_from_parent.gd")
	var local_var: MyEnum
	local_var = P.VALUE_B
	print(local_var)
