# Test access visibility of parent elements in nested class architectures.
class Parent:
	const parent_const := 1

	var parent_variable := 2

	signal parent_signal

	var parent_attribute: int:
		get:
			return 3

	func parent_func():
		return 4

	class Nested:
		const nested_const := 5


class Child extends Parent:
	func child_test():
		print(parent_const)
		print(self.parent_const)
		print(parent_variable)
		print(self.parent_variable)
		print(parent_signal.get_name())
		print(self.parent_signal.get_name())
		print(parent_attribute)
		print(self.parent_attribute)
		print(parent_func.get_method())
		print(self.parent_func.get_method())
		print(parent_func())
		print(self.parent_func())
		print(Nested.nested_const)
		print(self.Nested.nested_const)
		print(Parent.Nested.nested_const)


func test():
	Child.new().child_test()
