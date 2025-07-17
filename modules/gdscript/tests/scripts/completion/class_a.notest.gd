extends Node

class InnerA:
	class InnerInnerA:
		enum EnumOfInnerInnerA {
		ENUM_VALUE_1,
		ENUM_VALUE_2,
	}

	enum EnumOfInnerA {
		ENUM_VALUE_1,
		ENUM_VALUE_2,
	}

	signal signal_of_inner_a
	var property_of_inner_a
	func func_of_inner_a():
		pass

enum EnumOfA {
	ENUM_VALUE_1,
	ENUM_VALUE_2,
}

signal signal_of_a

var property_of_a

func func_of_a():
	pass


func _func_of_a_underscore():
	pass


static func func_of_a_static():
	pass

func func_of_a_args(a: int):
	pass

func func_of_a_callable(call := func():
	var x_of_a = 10):
	pass
