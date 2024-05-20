enum MyEnum { ENUM_VALUE_1, ENUM_VALUE_2 }

# Assigning int value to enum-typed variable without explicit cast causes a warning.
# While it is valid it may be a mistake in the assignment.
var class_var: MyEnum = 0

func test():
	print(class_var)
	class_var = 1
	print(class_var)

	var local_var: MyEnum = 0
	print(local_var)
	local_var = 1
	print(local_var)
