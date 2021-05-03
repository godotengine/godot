# Test deeply nested class architectures.
class Test:
	var depth = 1

	class Nested:
		var depth = 10


class Test2 extends Test:
	var depth = 2


class Test3 extends Test2:
	var depth = 3


class Test4 extends Test3:
	var depth = 4

	class Nested:
		var depth = 100


func test():
	print(Test.new().depth)
	# These currently crash the engine.
	#print(Test2.new().depth)
	#print(Test3.new().depth)
	#print(Test4.new().depth)
	print(Test.Nested.new().depth)
	print(Test4.Nested.new().depth)
