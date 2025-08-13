#https://github.com/godotengine/godot/issues/107478

class BaseClass:
	var base_value = "hello"

	func get_base_value():
		return base_value

	func get_number():
		return 42

class DerivedClass extends BaseClass:
	var inherited_prop:
		get = get_base_value

	var number_prop:
		get = get_number

func test():
	var derived = DerivedClass.new()
	print(derived.inherited_prop + "\n" + str(derived.number_prop))
