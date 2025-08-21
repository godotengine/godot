extends SceneTree


# does not have test_parsing_native_does_not_generate_orphans
#     added <RefCounted#-9223371895372511569>::<RefCounted#-9223371895372511569>
# has test_parsing_native_does_not_generate_orphans::<Object#null>
# something went wrong
# * test_parsing_native_does_not_generate_orphans
#     [ERROR]:  test_summary was null, waiting and then trying to get it again.
#     [ERROR]:  test_summary for @@7:<Node#65296926470>(test_script_parser.gd/TestParsedScript).test_parsing_native_does_not_generate_orphans is null somehow



class Test:
	var pass_texts = []
	var fail_texts = []
	var pending_texts = []
	var orphans = 0
	var line_number = 0

	# must have passed an assert and not have any other status to be passing
	func is_passing():
		return pass_texts.size() > 0 and fail_texts.size() == 0 and pending_texts.size() == 0

	# failing takes precedence over everything else, so any failures makes the
	# test a failure.
	func is_failing():
		return fail_texts.size() > 0

	# test is only pending if pending was called and the test is not failing.
	func is_pending():
		return pending_texts.size() > 0 and fail_texts.size() == 0

	func did_something():
		return is_passing() or is_failing() or is_pending()


	# NOTE:  The "failed" and "pending" text must match what is outputted by
	# the logger in order for text highlighting to occur in summary.
	func to_s():
		var pad = '     '
		var to_return = ''
		for i in range(fail_texts.size()):
			to_return += str(pad, '[Failed]:  ', fail_texts[i], "\n")
		for i in range(pending_texts.size()):
			to_return += str(pad, '[Pending]:  ', pending_texts[i], "\n")
		return to_return

	func get_status():
		var to_return = 'no asserts'
		if(pending_texts.size() > 0):
			to_return = 'pending'
		elif(fail_texts.size() > 0):
			to_return = 'fail'
		elif(pass_texts.size() > 0):
			to_return = 'pass'

		return to_return


var _tests = {}
var _test_order = []
var ParsedScript = load('res://addons/gut/script_parser.gd')


func get_test_obj(obj_name):
	if(!_tests.has(obj_name)):
		# print('does not have ', obj_name)
		_tests[obj_name] = Test.new()
		_test_order.append(obj_name)

	return _tests[obj_name]


var func_names = [
	'test_can_make_one',
	'test_can_parse_a_script',
	'test_parsing_same_thing_does_not_add_to_scripts',
	'test_parse_returns_script_parser',
	'test_parse_returns_cached_version_on_2nd_parse',
	'test_can_get_instance_parse_result_from_gdscript',
	'test_parsing_more_adds_more_scripts',
	'test_can_parse_path_string',
	'test_when_passed_an_invalid_path_null_is_returned',
	'test_inner_class_sets_subpath',
	'test_inner_class_sets_script_path',
	'test_can_make_one_from_gdscript',
	'test_can_make_one_from_instance',
	'test_instance_and_gdscript_have_same_methods',
	'test_new_from_gdscript_sets_path',
	'test_new_from_inst_sets_path',
	'test_can_get_method_by_name',
	'test_can_get_super_method_by_name',
	'test_can_get_local_method_by_name',
	'test_can_super_methods_not_included_in_local_method_by_name',
	'test_overloaded_local_methods_are_local',
	'test_get_local_method_names_excludes_supers',
	'test_get_super_method_names_excludes_locals',
	'test_is_blacklisted_returns_true_for_blacklisted_methods',
	'test_is_black_listed_returns_false_for_non_blacklisted_methods',
	'test_is_black_listed_returns_null_for_methods_that_DNE',
	'test_subpath_is_null_by_default',
	'test_cannot_set_subpath',
	'test_subpath_set_when_passing_inner_and_parent',
	'test_subpath_set_for_deeper_inner_classes',
	'test_resource_is_loaded_script',
	'test_resource_is_loaded_inner',
	'test_extends_text_has_path_for_scripts',
	'test_extends_text_uses_class_name_for_natives',
	'test_extends_text_adds_inner_classes_to_end',
	'test_parsing_native_does_not_generate_orphans',
]

func make_and_free_a_script_parser():
	var parsed = ParsedScript.new(Node2D)
	parsed.unreference()
	parsed = null


func run_through_all_names_with_alteration(pre, suf):
	for n in func_names:
		var altered = pre + n + suf

		var result = get_test_obj(altered)
		if(result == null):
			print("!!!!!!!! ", altered, " was null")

		make_and_free_a_script_parser()

		result = get_test_obj(altered)
		if(result == null):
			print("!!!!!!!! ", altered, " was null")

func do_it_a_bunch():
	run_through_all_names_with_alteration('', '')
	run_through_all_names_with_alteration('a_', '')
	run_through_all_names_with_alteration('asdfasdfasdfasdfasd___asdasdfadfads_', '_basdfasdfasdfasdf asdf asdfa sdf')


func _init():
	print('start')
	do_it_a_bunch()
	print('end')
	quit()