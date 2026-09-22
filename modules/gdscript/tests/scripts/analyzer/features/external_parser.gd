extends "external_parser_base1.notest.gd"

const External1 = preload("external_parser_script1.notest.gd")

func baz(e1: External1) -> void:
	print(e1.e1c.bar)
	print(e1.baz)

func test_external_base_parser_type_resolve(_v: TypeFromBase):
	pass

func test():
	var ext := External1.new()
	print(ext.array[0].test2)
	print(ext.get_external2().get_external3().test3)
	# TODO: This actually produces a runtime error, but we're testing the analyzer here
	#baz(ext)
