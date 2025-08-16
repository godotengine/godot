extends Node

const NO_TYPE_CONST = 0
const TYPE_CONST: int = 1
const GUESS_TYPE_CONST := 2

class Test:
	var a = NO_TYPE_CONST
	var b = TYPE_CONST
	var c = GUESS_TYPE_CONST

func test():
	var test_instance = Test.new()
	prints("a", test_instance.a, test_instance.a == NO_TYPE_CONST)
	prints("b", test_instance.b, test_instance.b == TYPE_CONST)
	prints("c", test_instance.c, test_instance.c == GUESS_TYPE_CONST)
