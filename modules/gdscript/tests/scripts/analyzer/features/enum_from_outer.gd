enum Named { VALUE_A, VALUE_B, VALUE_C = 42 }

class Test:
	var a = Named.VALUE_A
	var b = Named.VALUE_B
	var c = Named.VALUE_C

func test():
	var test_instance = Test.new()
	prints("a", test_instance.a, test_instance.a == Named.VALUE_A)
	prints("b", test_instance.b, test_instance.b == Named.VALUE_B)
	prints("c", test_instance.c, test_instance.c == Named.VALUE_C)
