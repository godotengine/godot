# Test non-nested/slightly nested class architecture.
class Test:
	var number = 25
	var string = "hello"


class TestSub extends Test:
	var string = "bye"


class TestConstructor:
	func _init(argument = 10):
		print(str("constructor with argument ", argument))


func test():
	var test_instance = Test.new()
	test_instance.number = 42

	# Creating a subclass currently crashes the engine.
	#var test_sub = TestSub.new()
	#assert(test_sub.number == 25)  # From Test.
	#assert(test_sub.string == "bye")  # From TestSub.

	TestConstructor.new()
	TestConstructor.new(500)
