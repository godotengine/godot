extends GutInternalTester

class TestType2Str:
	extends GutInternalTester

	var strutils = load('res://addons/gut/strutils.gd').new()
	# manually copied from help.
	var all_types = [
		TYPE_AABB ,
		TYPE_ARRAY ,
		TYPE_BASIS ,
		TYPE_BOOL,
		TYPE_CALLABLE ,
		TYPE_COLOR ,
		TYPE_DICTIONARY ,
		TYPE_FLOAT,
		TYPE_INT,
		TYPE_MAX ,
		TYPE_NODE_PATH ,
		TYPE_OBJECT ,
		TYPE_PACKED_BYTE_ARRAY ,
		TYPE_PACKED_COLOR_ARRAY ,
		TYPE_PACKED_FLOAT32_ARRAY ,
		TYPE_PACKED_FLOAT64_ARRAY ,
		TYPE_PACKED_INT32_ARRAY ,
		TYPE_PACKED_INT64_ARRAY ,
		TYPE_PACKED_STRING_ARRAY ,
		TYPE_PACKED_VECTOR2_ARRAY ,
		TYPE_PACKED_VECTOR3_ARRAY ,
		TYPE_PLANE ,
		TYPE_PROJECTION ,
		TYPE_QUATERNION ,
		TYPE_RECT2,
		TYPE_RECT2I,
		TYPE_RID ,
		TYPE_SIGNAL ,
		TYPE_STRING_NAME ,
		TYPE_STRING,
		TYPE_TRANSFORM2D ,
		TYPE_TRANSFORM3D ,
		TYPE_VECTOR2,
		TYPE_VECTOR2I,
		TYPE_VECTOR3,
		TYPE_VECTOR3I ,
		TYPE_VECTOR4 ,
		TYPE_VECTOR4I ,
	]

	class ExtendsControl:
		extends Control

		func do_something():
			pass # does nothing, heh

	class ExtendsNothing:
		var foo = 'bar'

	func test_types_contains_all_types(t = use_parameters(all_types)):
		assert_has(strutils.types, t)

	func test_string():
		assert_eq(strutils.type2str('this'), '"this"')

	func test_int():
		assert_eq(strutils.type2str(1234), '1234')

	var float_params = [[1.234, '1.234'], [1.0, '1.0'], [.1, '0.1'], [9.87000, '9.87']]
	func test_float(p = use_parameters(float_params)):
		assert_eq(strutils.type2str(p[0]), p[1])

	func test_script():
		assert_eq(strutils.type2str(self), str(self, '(test_strutils.gd/TestType2Str)'))

	func test_script_2():
		var dm = autofree(DoubleMe.new())
		assert_eq(strutils.type2str(dm), str(dm) + '(double_me.gd)')

	func test_scene():
		var scene = autofree(DoubleMeScene.instantiate())
		assert_eq(strutils.type2str(scene),  str(scene, '(double_me_scene.gd)'))

	func test_file_instance():
		var f = FileAccess.open('res://test/unit/test_strutils.gd', FileAccess.READ)
		assert_eq(strutils.type2str(f), str(f))

	func test_vector2():
		var v2 = Vector2(20, 30)
		assert_eq(strutils.type2str(v2), 'VECTOR2(20.0, 30.0)')

	func test_null():
		assert_eq(strutils.type2str(null), str(null))

	func test_boolean():
		assert_eq(strutils.type2str(true), str(true))
		assert_eq(strutils.type2str(false), str(false))

	func test_color():
		var c  = Color(.1, .2, .3)
		assert_eq(strutils.type2str(c), 'COLOR(0.1, 0.2, 0.3, 1.0)')

	func test_loaded_scene():
		assert_eq(strutils.type2str(DoubleMeScene), str(DoubleMeScene) + '(double_me_scene.tscn)')

	func test_doubles():
		var d = double(DoubleMe).new()
		assert_eq(strutils.type2str(d), str(d) + '(double of double_me.gd)')

	func test_another_double():
		var d = double(DoubleExtendsNode2D).new()
		assert_eq(strutils.type2str(d), str(d) + '(double of double_extends_node2d.gd)')

	func test_double_inner():
		gut.get_doubler().inner_class_registry.register(InnerClasses)
		var d = double(InnerClasses.InnerA).new()
		assert_eq(strutils.type2str(d), str(d) + '(double of inner_classes.gd/InnerA)')

	func test_partial_double():
		var d = partial_double(DoubleMe).new()
		assert_string_contains(strutils.type2str(d), "partial-double")

	# # func test_singleton_double_includes_singleton_name():
	# # 	var d = double_singleton("Input").new()
	# # 	assert_string_contains(strutils.type2str(d), "double of Input")

	# # func test_singleton_double_includes_word_singleton():
	# # 	var d = double_singleton("Input").new()
	# # 	assert_string_contains(strutils.type2str(d), "Singleton")

	func test_assert_null():
		assert_eq(strutils.type2str(null), str(null))

	func test_object_null():
		var scene = autofree(load(DOUBLE_ME_SCENE_PATH).instantiate())
		assert_eq(strutils.type2str(scene.get_parent()), "<null>")

	# # currently does not print the inner class, maybe later.
	func test_inner_class():
		var ec = autofree(ExtendsControl.new())
		assert_eq(strutils.type2str(ec), str(ec) + '(test_strutils.gd/TestType2Str/ExtendsControl)')

	func test_simple_class():
		var en = autofree(ExtendsNothing.new())
		assert_eq(strutils.type2str(en), str(en) + '(test_strutils.gd/TestType2Str/ExtendsNothing)')

	func test_returns_null_for_just_freed_objects():
		var n = autofree(Node.new())
		n.free()
		assert_eq(str(n), '<Freed Object>', 'sometimes fails based on timing.')

	func test_memory_leak():
		print(gut.get_orphan_counter().orphan_count(), ' t-1')
		var a = Node
		print(gut.get_orphan_counter().orphan_count(), ' t-2')
		print(gut.get_orphan_counter().orphan_count(), ' t-3')
		assert_no_new_orphans()

	func test_can_use_type2str_on_a_node_instance():
		var a = autofree(Node.new())
		var txt = strutils.type2str(a)
		assert_eq(txt, str(a))

	func test_can_use_type2str_on_Node():
		var txt = strutils.type2str(Node)
		assert_eq(txt, 'Node')

	func test_duplicated_resources_does_not_error():
		# checks issue #481 which I could not duplicate.
		var some_mat = Material.new()
		var dupe_mat = some_mat.duplicate()
		assert_not_null(str(dupe_mat))



class TestTruncateString:
	extends GutTest

	var strutils = load('res://addons/gut/strutils.gd').new()

	func test_when_less_than_max_entire_string_returned():
		var result  = strutils.truncate_string('this is small', 100)
		assert_eq(result, 'this is small')

	func test_when_more_than_max_then_string_is_smaller():
		var result = strutils.truncate_string('123456789012345678901234567890', 20)
		assert_lt(result.length(), 30)

	func test_when_neg_one_passed_whole_string_is_returned():
		var s = 'this is some text that I have  typed into here for you to read.'
		var result = strutils.truncate_string(s, -1)
		assert_eq(result, s)


class TestIndent:
	extends GutTest

	var strutils = load('res://addons/gut/strutils.gd').new()

	func test_one_indent_for_single_line():
		var indented = strutils.indent_text('hello world', 1, '  ')
		assert_string_starts_with(indented, '  ')

	func test_four_indents_for_single_line():
		var indented = strutils.indent_text('hello world', 4, '__')
		assert_string_starts_with(indented, '________')

	func test_with_newlines():
		var indented = strutils.indent_text("hello\nworld", 2, '__')
		assert_eq(indented, "____hello\n____world")

	func test_newlines_with_trailing_newline():
		var indented = strutils.indent_text("hello\nworld\n", 2, '__')
		assert_eq(indented, "____hello\n____world\n")
