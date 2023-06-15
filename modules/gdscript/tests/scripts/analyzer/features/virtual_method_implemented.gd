class TestOne:
	func _get_property_list():
		return {}

class TestTwo extends TestOne:
	func _init():
		var _x = _get_property_list()

func test():
	var x = TestTwo.new()
	var _x = x._get_property_list()
