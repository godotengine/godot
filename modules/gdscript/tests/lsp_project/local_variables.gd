extends Node

var member := 2

func test_member() -> void:
	var test := member + 42
	test += 3
	member += 5
	test = return_arg(test)
	print(test)

func return_arg(arg: int) -> int:
	arg += 2
	return arg