extends GutTest


class TestScriptParser:
	extends GutTest

	const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
	var DoubleMe = load(DOUBLE_ME_PATH)
	var ExtendsNode = load('res://test/resources/doubler_test_objects/double_extends_node2d.gd')
	const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
	var InnerClasses = load(INNER_CLASSES_PATH)
	var ScriptParser = load('res://addons/gut/script_parser.gd')

	func test_can_make_one():
		assert_not_null(ScriptParser.new())

	func test_can_parse_a_script():
		var collector = ScriptParser.new()
		collector.parse(DoubleMe)
		assert_eq(collector.scripts.size(), 1)

	func test_parsing_same_thing_does_not_add_to_scripts():
		var collector = ScriptParser.new()
		collector.parse(DoubleMe)
		collector.parse(DoubleMe)
		assert_eq(collector.scripts.size(), 1)

	func test_parse_returns_script_parser():
		var collector = ScriptParser.new()
		var result = collector.parse(DoubleMe)
		assert_is(result, ScriptParser.ParsedScript)

	func test_parse_returns_cached_version_on_2nd_parse():
		var collector = ScriptParser.new()
		collector.parse(DoubleMe)
		var result = collector.parse(DoubleMe)
		assert_is(result, ScriptParser.ParsedScript)

	func test_can_get_instance_parse_result_from_gdscript():
		var collector = ScriptParser.new()
		collector.parse(autofree(DoubleMe.new()))
		var result = collector.parse(DoubleMe)
		assert_is(result, ScriptParser.ParsedScript)
		assert_eq(collector.scripts.size(), 1)

	func test_parsing_more_adds_more_scripts():
		var collector = ScriptParser.new()
		collector.parse(DoubleMe)
		collector.parse(ExtendsNode)
		assert_eq(collector.scripts.size(), 2)

	func test_can_parse_path_string():
		var collector = ScriptParser.new()
		collector.parse(DOUBLE_ME_PATH)
		assert_eq(collector.scripts.size(), 1)

	func test_when_passed_an_invalid_path_null_is_returned():
		var collector = ScriptParser.new()
		var result = collector.parse('res://foo.bar')
		assert_null(result)

	func test_inner_class_sets_subpath():
		var collector = ScriptParser.new()
		var parsed = collector.parse(InnerClasses, InnerClasses.InnerCA)
		assert_eq(parsed.subpath, 'InnerCA')

	func test_inner_class_sets_script_path():
		var collector = ScriptParser.new()
		var parsed = collector.parse(InnerClasses, InnerClasses.InnerCA)
		assert_eq(parsed.script_path, INNER_CLASSES_PATH)




class HasAccessors:
	var my_property = 'default' :
		get: return my_property
		set(val): my_property = val


class TestParsedScript:
	extends GutTest

	const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
	const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
	var DoubleMe = load(DOUBLE_ME_PATH)
	var InnerClasses = load(INNER_CLASSES_PATH)

	var ParsedScript = load('res://addons/gut/script_parser.gd').ParsedScript

	class ClassWithInner:
		class InnerClass:
			var foo = 'bar'

	func test_can_make_one_from_gdscript():
		assert_not_null(ParsedScript.new(DoubleMe))

	func test_can_make_one_from_instance():
		var inst = autofree(DoubleMe.new())
		assert_not_null(ParsedScript.new(inst))

	func test_instance_and_gdscript_have_same_methods():
		var gd_parser = ParsedScript.new(DoubleMe)
		var inst = autofree(DoubleMe.new())
		var inst_parser = ParsedScript.new(inst)

		assert_eq(gd_parser.get_sorted_method_names(), inst_parser.get_sorted_method_names())

	func test_new_from_gdscript_sets_path():
		var parser = ParsedScript.new(DoubleMe)
		assert_eq(parser.script_path, DOUBLE_ME_PATH)

	func test_new_from_inst_sets_path():
		var inst = autofree(DoubleMe.new())
		var parser = ParsedScript.new(inst)
		assert_eq(parser.script_path, DOUBLE_ME_PATH)

	func test_can_get_method_by_name():
		var parser = ParsedScript.new(DoubleMe)
		assert_not_null(parser.get_method('_get'))

	func test_can_get_super_method_by_name():
		var parser = ParsedScript.new(DoubleMe)
		assert_not_null(parser.get_super_method('_get'))

	func test_non_super_methods_are_not_in_get_super_method_by_name():
		var parser = ParsedScript.new(DoubleMe)
		assert_null(parser.get_super_method('has_string_and_array_defaults'))

	func test_can_get_local_method_by_name():
		var parser = ParsedScript.new(DoubleMe)
		assert_not_null(parser.get_local_method('has_string_and_array_defaults'))

	func test_can_super_methods_not_included_in_local_method_by_name():
		var parser = ParsedScript.new(DoubleMe)
		assert_null(parser.get_local_method('_get'))

	func test_overloaded_local_methods_are_local():
		var parser = ParsedScript.new(DoubleMe)
		assert_not_null(parser.get_local_method('_init'))

	func test_get_local_method_names_excludes_supers():
		var parser = ParsedScript.new(DoubleMe)
		var names = parser.get_local_method_names()
		assert_does_not_have(names, '_get')

	func test_get_super_method_names_excludes_locals():
		var parser = ParsedScript.new(DoubleMe)
		var names = parser.get_super_method_names()
		assert_does_not_have(names, 'has_string_and_array_defaults')

	func test_subpath_is_null_by_default():
		var parser = ParsedScript.new(DoubleMe)
		assert_null(parser.subpath)

	func test_cannot_set_subpath():
		var parser = ParsedScript.new(DoubleMe)
		parser.subpath = 'asdf'
		assert_null(parser.subpath)

	func test_subpath_set_when_passing_inner_and_parent():
		var parser = ParsedScript.new(InnerClasses, InnerClasses.InnerA)
		assert_eq(parser.subpath, 'InnerA')

	func test_subpath_set_for_deeper_inner_classes():
		var parser = ParsedScript.new(InnerClasses, InnerClasses.InnerB.InnerB1)
		assert_eq(parser.subpath, 'InnerB.InnerB1')

	func test_resource_is_loaded_script():
		var parser = ParsedScript.new(DoubleMe)
		assert_eq(parser.resource, DoubleMe)

	func test_resource_is_loaded_inner():
		var InnerB1 = InnerClasses.InnerB.InnerB1
		var parser = ParsedScript.new(InnerClasses, InnerB1)
		assert_eq(parser.resource, InnerB1)

	func test_extends_text_has_path_for_scripts():
		var parsed = ParsedScript.new(DoubleMe)
		assert_eq(parsed.get_extends_text(), str("extends '", DOUBLE_ME_PATH, "'"))

	func test_extends_text_uses_class_name_for_natives():
		var parsed = ParsedScript.new(Node2D)
		assert_eq(parsed.get_extends_text(), 'extends Node2D')
		parsed.unreference()
		parsed = null

	func test_extends_text_adds_inner_classes_to_end():
		var InnerB1 = InnerClasses.InnerB.InnerB1
		var parsed = ParsedScript.new(InnerClasses, InnerB1)
		assert_eq(parsed.get_extends_text(),
			str("extends '", INNER_CLASSES_PATH, "'.InnerB.InnerB1"))

	func test_parsing_native_does_not_generate_orphans():
		var parsed = ParsedScript.new(Node2D)
		await get_tree().process_frame # avoids error godot:69411
		parsed.unreference()
		parsed = null
		assert_no_new_orphans()

	func test_parsing_native_ref_counted_does_not_generate_error():
		var parsed = ParsedScript.new(StreamPeerTCP)
		assert_not_null(parsed)

	func test_get_accessor_marked_as_accessor():
		var parsed = ParsedScript.new(HasAccessors)
		var method = parsed.get_method('@my_property_getter')
		assert_true(method.is_accessor())


	func test_set_accessor_marked_as_accessor():
		var parsed = ParsedScript.new(HasAccessors)
		var method = parsed.get_method('@my_property_setter')
		assert_true(method.is_accessor())




class TestParsedMethod:
	extends GutTest

	var ScriptParser = load('res://addons/gut/script_parser.gd')
	var _empty_meta = {
		"args":[],
		"default_args": [],
		"flags":0,
		"name":"empty"
	}

	func test_can_make_one():
		var pm = ScriptParser.ParsedMethod.new(_empty_meta)
		assert_not_null(pm)

	func test_is_eligible_for_doubling_by_default():
		var pm = ScriptParser.ParsedMethod.new(_empty_meta)
		assert_true(pm.is_eligible_for_doubling())

	var flag_arr = [METHOD_FLAG_STATIC, METHOD_FLAG_VIRTUAL, METHOD_FLAG_OBJECT_CORE]
	func test_when_has_bad_flag_it_is_not_eligible_for_doubling(p = use_parameters(flag_arr)):
		var meta = _empty_meta.duplicate()
		meta.flags = meta.flags | p
		var pm = ScriptParser.ParsedMethod.new(meta)
		assert_false(pm.is_eligible_for_doubling())

	var flag_arr2 = [METHOD_FLAG_EDITOR, METHOD_FLAG_NORMAL, METHOD_FLAGS_DEFAULT, METHOD_FLAG_VARARG,
		METHOD_FLAG_EDITOR | METHOD_FLAG_NORMAL | METHOD_FLAGS_DEFAULT | METHOD_FLAG_VARARG]
	func test_when_has_ok_flag_it_is_eligible_for_doubling(p = use_parameters(flag_arr2)):
		var meta = _empty_meta.duplicate()
		meta.flags = meta.flags | p
		var pm = ScriptParser.ParsedMethod.new(meta)
		assert_true(pm.is_eligible_for_doubling())

	func test_when_method_black_listed_it_is_not_eligible_for_doubling():
		var meta = _empty_meta.duplicate()
		meta.name = ScriptParser.BLACKLIST[0]
		var pm = ScriptParser.ParsedMethod.new(meta)
		assert_false(pm.is_eligible_for_doubling())


