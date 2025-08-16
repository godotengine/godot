extends GutTest

# ImplicitInitializerIssue class defined in res://test/resources/implicit_initializer_class

# This test works fine
func test_double_foo():
	var DoubleImplicitInitializer = double(ImplicitInitializerIssue)
	var d = DoubleImplicitInitializer.new()
	assert_not_null(d)
	assert_is(d, DoubleImplicitInitializer)


# This works, but will cause the next double of Foo to error
func test_double_node_uses_foo():
	var DoubleUses = double(load('res://test/resources/uses_implicit_initializer.gd'))
	assert_not_null(DoubleUses, 'double class was created')

	var inst = DoubleUses.new()
	assert_not_null(inst, 'instance is not null')
	assert_not_null(inst.iii, 'instance foo var was initialized')


# This test fails because the p_script->implicit_initializer error cause the
# object returned to not be a Foo.
func test_double_foo_after_doubling_node_uses_foo():
	var DoubleImplicitInitializer = double(ImplicitInitializerIssue)
	var d = DoubleImplicitInitializer.new()
	assert_not_null(d)
	assert_is(d, DoubleImplicitInitializer,
		'this failes because of the p_script->implicit_initializer error')



# Future attempts to double Foo will result in DoubleFoo being null because GUT
# thinks that Foo is an Inner Class of something.  So double(Foo) results in a
# GUT error.  For reference, this is how GUT determines if a thing is an Inner
# Class:
# func is_inner_class(obj):
#	return is_gdscript(obj) and obj.resource_path == ''
func test_double_foo_after_doubling_node_uses_foo_again():
	var DoubleImplicitInitializer = double(ImplicitInitializerIssue)
	var d = DoubleImplicitInitializer.new()
	assert_not_null(d)
	assert_is(d, DoubleImplicitInitializer,
		'this failes because of the p_script->implicit_initializer error')

