const TestClass = preload("gdscript_duplicate_class.notest.gd")

func test():
	# (TestClass as GDScript).duplicate() exists
	TestClass.duplicate()
