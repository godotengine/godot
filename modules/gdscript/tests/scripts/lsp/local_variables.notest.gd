extends Node

var member := 2
#   ^^^^^^ member -> member

func test_member() -> void:
	var test := member + 42
	#   |  |    ^^^^^^ -> member
	#   ^^^^ test -> test
	test += 3
	#<^^ -> test
	member += 5	
	#<^^^^ -> member
	test = return_arg(test)
	#  |              ^^^^ -> test
	#<^^ -> test
	print(test)
	#     ^^^^ -> test

func return_arg(arg: int) -> int:
#               ^^^ arg -> arg
	arg += 2
	#<^ -> arg
	return arg
	#      ^^^ -> arg