#https://github.com/godotengine/godot/issues/107478
class BaseClass:
	var base_value = "initial"

	func set_base_value(value):
		base_value = "set_" + str(value)

	func get_base_value():
		return base_value

class DerivedClass extends BaseClass:
	var inherited_prop:
		get = get_base_value,
		set = set_base_value

func test():
	var derived = DerivedClass.new()
	print(derived.inherited_prop)
	derived.inherited_prop = "hello"
	print("\n" + derived.inherited_prop)
