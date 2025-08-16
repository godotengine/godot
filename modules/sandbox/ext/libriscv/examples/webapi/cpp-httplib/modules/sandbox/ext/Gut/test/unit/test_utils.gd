extends 'res://addons/gut/test.gd'

const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'

var InnerClasses = load(INNER_CLASSES_PATH)
var Utils = load('res://addons/gut/utils.gd')


func test_can_make_one():
	assert_not_null(autofree(Utils.new()))

func test_is_double_returns_false_for_non_doubles():
	assert_false(GutUtils.is_double(autofree(Node.new())))

func test_is_double_returns_true_for_doubles():
	var d = double(Node).new()
	assert_true(GutUtils.is_double(d))

func test_is_double_returns_false_for_primitives():
	assert_false(GutUtils.is_double('hello'), 'string')
	assert_false(GutUtils.is_double(1), 'int')
	assert_false(GutUtils.is_double(1.0), 'float')
	assert_false(GutUtils.is_double([]), 'array')
	assert_false(GutUtils.is_double({}), 'dictionary')
	# that's probably enough spot checking


class OverloadsGet:
	var a = []
	@warning_ignore("native_method_override")
	func get(index):
		return a[index]

func test_is_double_works_with_classes_that_overload_get():
	var og = autofree(OverloadsGet.new())
	assert_false(GutUtils.is_double(og))

func test_is_instance_false_for_classes():
	assert_false(GutUtils.is_instance(Node2D))

func test_is_instance_true_for_new():
	var n = autofree(Node.new())
	assert_true(GutUtils.is_instance(n))

func test_is_instance_false_for_instanced_things():
	var i = load('res://test/resources/SceneNoScript.tscn')
	assert_false(GutUtils.is_instance(i))


func test_get_native_class_name_does_not_generate_orphans():
	var _n = GutUtils.get_native_class_name(Node2D)
	assert_no_new_orphans()

func test_get_native_class_name_does_not_free_references():
	var _n = GutUtils.get_native_class_name(InputEventKey)
	pass_test("we got here")

func test_is_native_class_returns_true_for_native_classes():
	assert_true(GutUtils.is_native_class(Node))

func test_is_inner_class_true_for_inner_classes():
	assert_true(GutUtils.is_inner_class(InnerClasses.InnerA))

func test_is_inner_class_false_for_base_scripts():
	assert_false(GutUtils.is_inner_class(InnerClasses))

func test_is_inner_class_false_for_non_objs():
	assert_false(GutUtils.is_inner_class('foo'))

func test_is_install_valid_true_by_default():
	assert_true(GutUtils.is_install_valid())

func test_is_install_valid_false_if_function_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.FUNCTION = 'res://does_not_exist.txt'
	assert_false(GutUtils.is_install_valid(template_paths))

func test_is_install_valid_false_if_init_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.INIT = 'res://does_not_exist.txt'
	assert_false(GutUtils.is_install_valid(template_paths))

func test_is_install_valid_false_if_script_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.SCRIPT = 'res://does_not_exist.txt'
	assert_false(GutUtils.is_install_valid(template_paths))

func test_is_install_valid_false_when_godot_version_too_low():
	var ver_nums = GutUtils.VersionNumbers.new('50.50.50', '50.50.50')
	assert_false(GutUtils.is_install_valid(GutUtils.DOUBLE_TEMPLATES, ver_nums))

func test_make_install_check_text_contains_missing_tempalte_text_when_function_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.FUNCTION = 'res://does_not_exist.txt'
	var text = GutUtils.make_install_check_text(template_paths)
	assert_string_contains(text, 'template files are missing', false)

func test_make_install_check_text_contains_missing_tempalte_text_when_init_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.INIT = 'res://does_not_exist.txt'
	var text = GutUtils.make_install_check_text(template_paths)
	assert_string_contains(text, 'template files are missing', false)

func test_make_install_check_text_contains_missing_tempalte_text_when_script_template_missing():
	var template_paths = GutUtils.DOUBLE_TEMPLATES.duplicate()
	template_paths.SCRIPT = 'res://does_not_exist.txt'
	var text = GutUtils.make_install_check_text(template_paths)
	assert_string_contains(text, 'template files are missing', false)

func test_make_install_check_text_contains_info_about_invalid_version():
	var ver_nums = GutUtils.VersionNumbers.new('50.50.50', '50.50.50')
	var text = GutUtils.make_install_check_text(GutUtils.DOUBLE_TEMPLATES, ver_nums)
	assert_string_contains(text, 'requires Godot ', false)

func test_make_install_check_text_contains_text_about_no_configured_directories():
	pending()


class TestGetSceneScript:
	extends 'res://addons/gut/test.gd'

	class MockSceneState:
		# ------------------------------
		# Tools for faking out SceneState functionality
		# ------------------------------
		var nodes = []

		func add_node(path):
			var to_add = {
				node_path = NodePath(path),
				props = []
			}
			nodes.append(to_add)
			return nodes.size() -1

		func add_node_prop(index, name, value):
			nodes[index].props.append({name = name, value = value})

		# ------------------------------
		# Mocked SceneState methods
		# ------------------------------
		func get_node_count():
			return nodes.size()

		func get_node_path(index):
			return nodes[index].node_path

		func get_node_property_name(index, prop_index):
			return nodes[index].props[prop_index].name

		func get_node_property_value(index, prop_index):
			return nodes[index].props[prop_index].value

		func get_node_property_count(index):
			return nodes[index].props.size()


	class MockScene:
		var state = MockSceneState.new()
		func get_state():
			return state


	func test_gets_scene_script_when_script_is_first_property():
		var mock_scene = MockScene.new()
		mock_scene.state.add_node('.')
		mock_scene.state.add_node_prop(0, 'script', 'foo')
		var result = GutUtils.get_scene_script_object(mock_scene)
		assert_eq(result, 'foo')

	func test_gets_scene_script_when_script_is_second_property():
		var mock_scene = MockScene.new()
		mock_scene.state.add_node('.')
		mock_scene.state.add_node_prop(0, 'something', 'else')
		mock_scene.state.add_node_prop(0, 'script', 'foo')
		var result = GutUtils.get_scene_script_object(mock_scene)
		assert_eq(result, 'foo')

	func test_gets_scene_script_when_root_node_is_not_first_node():
		var mock_scene = MockScene.new()
		mock_scene.state.add_node('/some/path')

		mock_scene.state.add_node('.')
		mock_scene.state.add_node_prop(1, 'something', 'else')
		mock_scene.state.add_node_prop(1, 'script', 'foo')

		var result = GutUtils.get_scene_script_object(mock_scene)
		assert_eq(result, 'foo')




class TestGetEnumValue:
	extends GutTest

	enum TEST1{
		ZERO,
		ONE,
		TWO,
		THREE,
		TWENTY_ONE
	}


	func test_returns_index_when_given_index():
		var val = GutUtils.get_enum_value(0, TEST1)
		assert_eq(val, 0)

	func test_returns_null_when_invalid_index():
		var val = GutUtils.get_enum_value(99, TEST1)
		assert_eq(val, null)

	func test_returns_value_when_given_string():
		var val = GutUtils.get_enum_value('TWO', TEST1)
		assert_eq(val, 2)

	func test_returns_value_when_given_lowercase_string():
		var val = GutUtils.get_enum_value('three', TEST1)
		assert_eq(val, 3)

	func test_replaces_spaces_with_underscores():
		var val = GutUtils.get_enum_value('twenty ONE', TEST1)
		assert_eq(val, TEST1.TWENTY_ONE)

	func test_returns_null_if_string_not_a_key():
		var val = GutUtils.get_enum_value('not a key', TEST1)
		assert_null(val)

	func test_can_provide_default_value():
		var val = GutUtils.get_enum_value('not a key', TEST1, 'asdf')
		assert_eq(val, 'asdf')

	func test_when_int_passed_as_string_it_converts_it():
		var val = GutUtils.get_enum_value('1', TEST1, 999)
		assert_eq(val, 1)

	func test_with_double_strategy():
		var val = GutUtils.get_enum_value(
			0, GutUtils.DOUBLE_STRATEGY,
			999)
		assert_eq(val, 0)

	func test_with_double_strategy2():
		var val = GutUtils.get_enum_value(
			1, GutUtils.DOUBLE_STRATEGY,
			999)
		assert_eq(val, 1)

	func test_converts_floats_to_int():
		var val = GutUtils.get_enum_value(1.0, TEST1, 9999)
		assert_eq(val, 1)

	func test_does_not_round_floats():
		var val = GutUtils.get_enum_value(2.9, TEST1, 9999)
		assert_eq(val, 2)





