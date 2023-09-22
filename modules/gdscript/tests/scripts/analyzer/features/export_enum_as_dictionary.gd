class_name TestExportEnumAsDictionary

enum MyEnum {A, B, C}

const Utils = preload("../../utils.notest.gd")

@export var x1 = MyEnum
@export var x2 = MyEnum.A
@export var x3 := MyEnum
@export var x4 := MyEnum.A
@export var x5: MyEnum

func test():
	for property in get_property_list():
		if property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE:
			print(Utils.get_property_signature(property))
			print("  ", Utils.get_property_additional_info(property))
