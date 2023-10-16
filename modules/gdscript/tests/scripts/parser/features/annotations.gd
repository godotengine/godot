extends Node

@export_enum("A", "B", "C") var a0
@export_enum("A", "B", "C",) var a1

@export_enum(
	"A",
	"B",
	"C"
) var a2

@export_enum(
	"A",
	"B",
	"C",
) var a3

@export
var a4: int

@export()
var a5: int

@export() var a6: int
@warning_ignore("onready_with_export") @onready @export var a7: int
@warning_ignore("onready_with_export") @onready() @export() var a8: int

@warning_ignore("onready_with_export")
@onready
@export
var a9: int

@warning_ignore("onready_with_export")
@onready()
@export()
var a10: int

@warning_ignore("onready_with_export")
@onready()
@export()

var a11: int


func test():
	for property in get_property_list():
		if property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE:
			print(property)
