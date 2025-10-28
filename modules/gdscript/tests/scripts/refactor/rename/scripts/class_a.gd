extends Node

class_name ClassA

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

	signal signal_of_a
	signal signal_of_inner_a
	var property_of_inner_a
	func func_of_inner_a():
		signal_of_a.emit()
		signal_of_inner_a.emit()

enum EnumOfA {
	ENUM_VALUE_1,
	ENUM_VALUE_2,
}

signal signal_of_a

var property_of_a
@export var export_property_of_a: Color


func func_of_a():
	prints("EnumOfInnerInnerB value:", B.InnerB.InnerInnerB.EnumOfInnerInnerB.ENUM_VALUE_1)

	for i in range(5):
		if i == 3:
			break
		if i % 2 == 0:
			continue


func _func_of_a_underscore():
	pass


static func func_of_a_static() -> void:
	pass

func func_of_a_args(a: int):
	signal_of_a.emit()

func func_of_a_callable(call := func():
	var x_of_a = 10
	print(x_of_a)):
	pass

static func secretely_a_all_along() -> ClassCisB:
	return ClassCisB.get_new_me()
