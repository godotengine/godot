extends Node

enum { VALUE_A, VALUE_B, VALUE_C = 42 }

class Test:
	var a = VALUE_A
	var b = VALUE_B
	var c = VALUE_C

func test():
	var test_instance = Test.new()
	prints("a", test_instance.a, test_instance.a == VALUE_A)
	prints("b", test_instance.b, test_instance.b == VALUE_B)
	prints("c", test_instance.c, test_instance.c == VALUE_C)
