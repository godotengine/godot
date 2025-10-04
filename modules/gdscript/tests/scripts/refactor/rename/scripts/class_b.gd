extends Node

class_name B

class InnerB:
	class InnerInnerB:
		enum EnumOfInnerInnerB {
			ENUM_VALUE_1,
			ENUM_VALUE_2,
		}

	enum EnumOfInnerB {
		ENUM_VALUE_1,
		ENUM_VALUE_2,
	}

	signal signal_of_inner_b
	var property_of_inner_b
	func func_of_inner_b():
		print(InnerInnerB.EnumOfInnerInnerB.ENUM_VALUE_1)
		print(EnumOfInnerB.ENUM_VALUE_1)
		print(EnumOfB.ENUM_VALUE_1)
		print(EnumOfB.ENUM_VALUE_2)
		signal_of_inner_b.emit()
		pass

enum EnumOfB {
	ENUM_VALUE_1,
	ENUM_VALUE_2,
}

signal signal_of_b

var property_of_b

func func_of_b():
	print(property_of_b)
	signal_of_b.emit()
	pass

static func hello_from_b() -> void:
	print(ClassA.secretely_a_all_along())
