# Test non-nested/slightly nested class architecture.
class Test:
	var number = 25
	var string = "hello"


class TestSub extends Test:
	var other_string = "bye"


class TestConstructor:
	func _init(argument = 10):
		print(str("constructor with argument ", argument))


func test():
	var test_instance = Test.new()
	test_instance.number = 42

	var test_sub = TestSub.new()
	Utils.check(test_sub.number == 25)  # From Test.
	Utils.check(test_sub.other_string == "bye")  # From TestSub.

	var _test_constructor = TestConstructor.new()
	_test_constructor = TestConstructor.new(500)
