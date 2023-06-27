# Test deeply nested class architectures.
class Test:
	var depth = 1

	class Nested:
		var depth_nested = 10


class Test2 extends Test:
	var depth2 = 2


class Test3 extends Test2:
	var depth3 = 3


class Test4 extends Test3:
	var depth4 = 4

	class Nested2:
		var depth4_nested = 100


func test():
	print(Test.new().depth)
	print(Test2.new().depth)
	print(Test2.new().depth2)
	print(Test3.new().depth)
	print(Test3.new().depth3)
	print(Test4.new().depth)
	print(Test4.new().depth4)
	print(Test.Nested.new().depth_nested)
	print(Test4.Nested2.new().depth4_nested)
