class BaseClass extends RefCounted:

	func greetings() -> void:
		print("Hello!")

class SubClass extends BaseClass:

	func greetings() -> void:
		print("Hello Sub Class!")

class SubSubClass extends SubClass:
	
	func greetings() -> void:
		print("Hello Sub Sub Class!")

class BaseGreeting extends RefCounted:

	func get_obj() -> BaseClass:
		return BaseClass.new()

class SubGreeting extends BaseGreeting:

	func get_obj() -> SubClass:
		return SubClass.new()

class SubSubGreeting extends SubGreeting:

	func get_obj() -> BaseClass:
		return SubSubClass.new()

func test():
	var base:BaseGreeting = SubGreeting.new()
	var obj = base.get_obj()
	obj.greetings()
