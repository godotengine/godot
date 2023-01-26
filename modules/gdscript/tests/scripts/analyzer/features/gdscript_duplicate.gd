const TestClass = preload("gdscript_duplicate_class.notest.gd")

func test():
	# TestClass.duplicate() fails
	@warning_ignore("return_value_discarded")
	(TestClass as GDScript).duplicate()
