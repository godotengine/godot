# GH-118877

class Test1:
	func _get_property_list():
		return [
			{ "name": "test_property", "type": TYPE_INT },
		]

class Test2:
	func _get_property_list():
		var properties = []

		properties.append({ "name": "test_property", "type": TYPE_INT })

		return properties

class Test3:
	func _get_property_list() -> Array[Dictionary]:
		var properties = []

		properties.append({ "name": "test_property", "type": TYPE_INT })

		return properties

func check(instance: Object) -> void:
	var has_property: bool = false
	for property in instance.get_property_list():
		if str(property.name) == "test_property":
			has_property = true
			break
	print(has_property)

func test():
	check(Test1.new())
	check(Test2.new())
	check(Test3.new())
