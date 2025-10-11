class BaseClass:
	@override
	func _get_property_list():
		return {"property" : "definition"}

class SuperClassMethodsRecognized extends BaseClass:
	func _init():
		# Recognizes super class methods.
		var _x = _get_property_list()

class SuperMethodsRecognized extends BaseClass:
	@override
	func _get_property_list():
		# Recognizes super method.
		var result = super()
		result["new"] = "new"
		return result

func test():
	var test1 = SuperClassMethodsRecognized.new()
	print(test1._get_property_list()) # Calls base class's method.
	var test2 = SuperMethodsRecognized.new()
	print(test2._get_property_list())
