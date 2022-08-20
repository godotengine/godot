# Untyped inline property
var prop1:
	get:
		return prop1
	set(value):
		prop1 = value

# Typed inline property
var prop2 : int:
	get:
		return prop2
	set(value):
		prop2 = value

# Typed inline property with default value
var prop3 : int = 1:
	get:
		return prop3
	set(value):
		prop3 = value

# Typed inline property with backing variable
var _prop4 : int = 2
var prop4: int:
	get:
		return _prop4
	set(value):
		_prop4 = value

func test():
	print(prop1)
	print(prop2)
	print(prop3)
	print(prop4)

	print()

	prop1 = 1
	prop2 = 2
	prop3 = 3
	prop4 = 4

	print(prop1)
	print(prop2)
	print(prop3)
	print(prop4)
