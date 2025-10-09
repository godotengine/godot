extends Node

const NoNameB := preload("res://scripts/noname_b.gd")
var member_variable


func _ready() -> void:
	print("Hello, " + NoNameB.NONAME_B_NAME)
	print(member_variable)


class NoNameAInner:
	var member_variable: = 2

	func _ready() -> void:
		print(member_variable)
		prints("hello", member_variable)
