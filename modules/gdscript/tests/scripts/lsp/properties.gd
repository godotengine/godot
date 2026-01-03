extends Node

var prop1 := 42
#   ^^^^^ prop1 -> prop1
var prop2 : int = 42
#   ^^^^^ prop2 -> prop2
var prop3 := 42:
#   ^^^^^ prop3 -> prop3
	get:
		return prop3 + 13
		#      ^^^^^ -> prop3
	set(value):
	#   ^^^^^ prop3:value -> prop3:value
		prop3 = value - 13
		#   |   ^^^^^ -> prop3:value
		#<^^^ -> prop3
var prop4: int:
#   ^^^^^ prop4 -> prop4
	get:
		return 42
var prop5 := 42:
#   ^^^^^ prop5 -> prop5
	set(value):
	#   ^^^^^ prop5:value -> prop5:value
		prop5 = value - 13
		#   |   ^^^^^ -> prop5:value
		#<^^^ -> prop5

var prop6:
#   ^^^^^ prop6 -> prop6
	get = get_prop6,
	#     ^^^^^^^^^ -> get_prop6
	set = set_prop6
	#     ^^^^^^^^^ -> set_prop6
func get_prop6():
#    ^^^^^^^^^ get_prop6 -> get_prop6
	return 42
func set_prop6(value):
#    |       | ^^^^^ set_prop6:value -> set_prop6:value
#    ^^^^^^^^^ set_prop6 -> set_prop6
	print(value)
	#     ^^^^^ -> set_prop6:value

var prop7:
#   ^^^^^ prop7 -> prop7
	get = get_prop7
	#     ^^^^^^^^^ -> get_prop7
func get_prop7():
#    ^^^^^^^^^ get_prop7 -> get_prop7
	return 42

var prop8:
#   ^^^^^ prop8 -> prop8
	set = set_prop8
	#     ^^^^^^^^^ -> set_prop8
func set_prop8(value):
#    |       | ^^^^^ set_prop8:value -> set_prop8:value
#    ^^^^^^^^^ set_prop8 -> set_prop8
	print(value)
	#     ^^^^^ -> set_prop8:value

const const_var := 42
#     ^^^^^^^^^ const_var -> const_var
static var static_var := 42
#          ^^^^^^^^^^ static_var -> static_var
