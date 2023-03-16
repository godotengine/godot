class A extends Node:
	pass

func test():
	var x = A.new()

	x.free()

	var ok = x
	var bad: A = x
