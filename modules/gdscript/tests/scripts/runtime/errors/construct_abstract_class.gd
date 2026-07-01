extends RefCounted

@abstract class AbstractClass:
	pass

var class_as_gdscript: GDScript = AbstractClass
var class_as_variant: Variant = AbstractClass

func subtest_gdscript():
	print(class_as_gdscript.can_instantiate())
	print(class_as_gdscript.new())

func subtest_variant():
	@warning_ignore_start("unsafe_method_access")
	print(class_as_variant.can_instantiate())
	print(class_as_variant.new())
	@warning_ignore_restore("unsafe_method_access")

func test():
	subtest_gdscript()
	subtest_variant()
