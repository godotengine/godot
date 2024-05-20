func test():
	var instance := Parent.new()
	instance.my_function({"a": 1})
	instance = Child.new()
	instance.my_function({"a": 1})
	print("No failure")

class Parent:
	func my_function(_par1: Dictionary = {}) -> void:
		pass

class Child extends Parent:
	func my_function(_par1: Dictionary = {}) -> void:
		pass
