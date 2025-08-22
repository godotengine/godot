extends "res://addons/gut/test.gd"
func test_something():
	pass

func test_nothing():
	pass

class TestClass1:
	extends "res://addons/gut/test.gd"
	func test_context1_one():
		pass
	func test_context1_two():
		pass
	func print_something():
		print('hello world')

class DifferentPrefixClass:
	extends "res://addons/gut/test.gd"
	func test_something():
		pass
	func not_a_test():
		pass

class DoesNotExtend:
	func test_something_not_extended():
		pass

class TestDoesNotExtendTest:
	func test_something():
		pass

class TestExtendsTestClass1:
	extends TestClass1
