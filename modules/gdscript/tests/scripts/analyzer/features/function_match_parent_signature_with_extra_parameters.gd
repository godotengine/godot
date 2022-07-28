func test():
	var instance := Parent.new()
	var result := instance.my_function(1)
	print(result)
	assert(result == 1)
	instance = Child.new()
	result = instance.my_function(2)
	print(result)
	assert(result == 0)

class Parent:
	func my_function(par1: int) -> int:
		return par1

class Child extends Parent:
	func my_function(_par1: int, par2: int = 0) -> int:
		return par2
