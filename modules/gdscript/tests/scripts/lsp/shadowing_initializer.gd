extends Node

var value := 42
#   ^^^^^ member:value -> member:value

func variable():
	var value = value + 42
	#!  |   |   ^^^^^ -> member:value
	#   ^^^^^ variable:value -> variable:value
	print(value)
	#     ^^^^^ -> variable:value

func array():
	var value = [1,value,3,value+4]
	#!  |   |      |   |   ^^^^^ -> member:value
	#!  |   |      ^^^^^ -> member:value
	#   ^^^^^ array:value -> array:value
	print(value)
	#     ^^^^^ -> array:value

func dictionary():
	var value = {
	#   ^^^^^ dictionary:value -> dictionary:value
		"key1": value,
		#!      ^^^^^ -> member:value
		"key2": 1 + value + 3,
		#!          ^^^^^ -> member:value
	}
	print(value)
	#     ^^^^^ -> dictionary:value

func for_loop():
	for value in value:
	#   |   |    ^^^^^ -> member:value
	#   ^^^^^ for:value -> for:value
		print(value)
		#     ^^^^^ -> for:value

func for_range():
	for value in range(5, value):
	#   |   |             ^^^^^ -> member:value
	#   ^^^^^ for:range:value -> for:range:value
		print(value)
		#     ^^^^^ -> for:range:value

func matching():
	match value:
	#     ^^^^^ -> member:value
		42: print(value)
		#         ^^^^^ -> member:value
		[var value, ..]: print(value)
		#    |   |             ^^^^^ -> match:array:value
		#    ^^^^^ match:array:value -> match:array:value
		var value: print(value)
		#   |   |        ^^^^^ -> match:var:value
		#   ^^^^^ match:var:value -> match:var:value
