extends "external_parser_script1_base.notest.gd"

const External2 = preload("external_parser_script2.notest.gd")
const External1c = preload("external_parser_script1c.notest.gd")

@export var e1c: External1c

var array: Array[External2] = [ External2.new() ]
var baz: int

func get_external2() -> External2:
	return External2.new()
