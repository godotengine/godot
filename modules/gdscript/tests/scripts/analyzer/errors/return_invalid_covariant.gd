class BaseClass extends RefCounted:

	func greetings() -> void:
		print("Hello!")

class OtherClass extends RefCounted:

	func greetings() -> void:
		print("Hello Other!")

class SubClass extends BaseClass:
	
	func greetings() -> void:
		print("Hello Sub Class!")

class BaseGreeting extends RefCounted:
	
	func get_obj() -> OtherClass:
		return BaseClass.new()

class SubGreeting extends BaseGreeting:
	
	func get_obj() -> SubClass:
		return SubClass.new()

func test():
	var base:BaseGreeting = SubGreeting.new()
	var obj = base.get_obj()
	obj.greetings()
