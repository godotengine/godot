class_name TestExportEnumAsDictionary

enum MyEnum {A, B, C}

const Utils = preload("../../utils.notest.gd")

@export var test_1 = MyEnum
@export var test_2 = MyEnum.A
@export var test_3 := MyEnum
@export var test_4 := MyEnum.A
@export var test_5: MyEnum

func test():
	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			Utils.print_property_extended_info(property)
