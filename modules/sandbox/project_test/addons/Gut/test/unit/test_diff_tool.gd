extends 'res://addons/gut/test.gd'


class TestArrayCompareResultInterace:
	extends 'res://addons/gut/test.gd'

	var DiffTool = GutUtils.DiffTool

	func test_cannot_set_summary():
		var ad = DiffTool.new([], [])
		ad.summary = 'the summary'
		assert_ne(ad.summary,  'the summary')

	func test_summary_prop_returns_summarize():
		var ad =  DiffTool.new([], [1])
		assert_not_null(ad.summary)

	func test_cannot_set_are_equal():
		var ad = DiffTool.new([], [])
		ad.are_equal = 'asdf'
		assert_eq(ad.are_equal, true)

	func test_are_equal_prop_returns_result_of_diff():
		var ad = DiffTool.new([], [])
		assert_eq(ad.are_equal, true)

	func test_get_total_different_returns_correct_count():
		var diff = DiffTool.new([1, 2, 3], [1, 'two', 99, 'four'])
		assert_eq(diff.get_different_count(), 3)

	var _total_count_vars = [
		[[1, 2, 3], [1, 2, 3, 4], 4],
		[[1, 2, 3, 4], [1, 2, 3], 4],
		[[1, 2], [1, 2], 2]
	]
	func test_get_total_count_returns_correct_count(p=use_parameters(_total_count_vars)):
		var diff = DiffTool.new(p[0], p[1])
		assert_eq(diff.get_total_count(), p[2])

	func test_get_short_summary_includes_x_of_y_keys_when_different():
		var diff = DiffTool.new([1, 2, 3, 4], [1, 'a', 'b', 'c', 'd'], GutUtils.DIFF.DEEP)
		assert_string_contains(diff.get_short_summary(), '4 of 5')

	func test_get_short_summary_does_not_include_x_of_y_when_equal():
		var diff = DiffTool.new([], [], GutUtils.DIFF.DEEP)
		assert_eq(diff.get_short_summary().find(' of '), -1, diff.get_short_summary())
		assert_string_contains(diff.get_short_summary(), '==')

	func test_brackets():
		var diff = DiffTool.new([], [])
		assert_eq(diff.get_brackets().open, '[', 'open')
		assert_eq(diff.get_brackets().close, ']', 'close')


class TestArrayDiff:
	extends 'res://addons/gut/test.gd'

	var DiffTool = GutUtils.DiffTool

	func test_can_instantiate_with_two_arrays():
		var ad  = DiffTool.new([], [])
		assert_not_null(ad)

	func test_constructor_defaults_diff_type_to_deep():
		var diff = DiffTool.new([], [])
		assert_eq(diff.get_diff_type(), GutUtils.DIFF.DEEP)

	func test_constructor_sets_diff_type():
		var diff = DiffTool.new([], [], GutUtils.DIFF.SIMPLE)
		assert_eq(diff.get_diff_type(), GutUtils.DIFF.SIMPLE)

	func test_is_equal_is_true_when_all_elements_match():
		var ad = DiffTool.new([1, 2, 3], [1, 2, 3])
		assert_true(ad.are_equal)

	func test_is_equal_returns_false_when_one_element_does_not_match():
		var ad = DiffTool.new([1, 2, 3], [1, 2, 99])
		assert_false(ad.are_equal, 'should be false but is ' + str(ad.are_equal))

	func test_lists_indexes_as_missing_in_first_array():
		var ad = DiffTool.new([1, 2, 3], [1, 2, 3, 4, 5])
		assert_string_contains(ad.summarize(), '<missing index> !=')

	func test_get_summary_text_lists_both_arrays():
		var ad = DiffTool.new([3, 2, 1, 98, 99], [1, 2, 3])
		assert_string_contains(ad.summarize(), '[3, 2, 1, 98, 99] != [1, 2, 3]')

	func test_get_summary_text_lists_differences():
		var ad = DiffTool.new([3, 2, 1, 98, 99], [1, 2, 3])
		assert_string_contains(ad.summarize(), '0:  3 !=')

	func test_when_arrays_are_large_then_summarize_truncates():
		var a1 = []
		var a2 = []
		for i in range(100):
			a1.append(i)
			if(i%2 == 0):
				a2.append(str(i))
			else:
				if(i < 90):
					a2.append(i)

		var ad = DiffTool.new(a1, a2)
		var summary = ad.summarize()
		assert_lt(summary.split("\n").size(), 40, summary)

	func test_works_with_strings_and_numbers():
		var a1 = [0, 1, 2, 3, 4]
		var a2 = [0, 'one', 'two', 'three', '4']
		var ad = DiffTool.new(a1, a2)
		gut.p(ad.summarize())
		pass_test('we got here')

	func test_when_arrays_are_equal_summarize_says_so():
		var ad = DiffTool.new(['a', 'b', 'c'], ['a', 'b', 'c'])
		assert_string_contains(ad.summarize(), ' == ')

	func test_diff_display_with_classes():
		var d_test = double(GutTest).new()
		var a1 = [gut, d_test]
		var a2 = [d_test, gut]
		var ad  = DiffTool.new(a1, a2)
		assert_string_contains(ad.summarize(), '(gut.gd)')
		assert_string_contains(ad.summarize(), 'double of test.gd')

	func test_diff_display_with_classes2():
		var d_test_1 = double(GutTest).new()
		var d_test_2 = double(GutTest).new()
		var a1 = [d_test_1, d_test_2]
		var a2 = [d_test_2, d_test_1]
		var ad  = DiffTool.new(a1, a2)
		assert_string_contains(ad.summarize(), 'double of test.gd')

	func test_dictionaries_in_sub_arrays():
		var a1 = [[{'a': 1}]]
		var a2 = [[{'a': 1}]]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.SIMPLE)
		assert_true(diff.are_equal, diff.summarize())


class TestArrayDeepDiff:
	extends 'res://addons/gut/test.gd'

	var DiffTool = GutUtils.DiffTool

	func test_diff_with_dictionaries_passes_when_not_same_reference_but_same_values():
		var a1 = [{'a':1}, {'b':2}]
		var a2 = [{'a':1}, {'b':2}]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		assert_true(diff.are_equal, diff.summarize())

	func test_diff_with_dictionaries_fails_when_different_values():
		var a1 = [{'a':1}, {'b':1}, {'c':1}, {'d':1}]
		var a2 = [{'a':1}, {'b':2}, {'c':2}, {'d':2}]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		assert_false(diff.are_equal, diff.summarize())

	func test_matching_dictionaries_in_sub_arrays():
		var a1 = [[{'a': 1}]]
		var a2 = [[{'a': 1}]]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		assert_true(diff.are_equal, diff.summarize())

	func test_non_matching_dictionaries_in_sub_arrays():
		var a1 = [[{'a': 1}], [{'b': 1}], [{'c': 1}]]
		var a2 = [[{'a': 1}], [{'b': 2}], [{'c': 2}]]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		assert_false(diff.are_equal, diff.summarize())

	func test_when_deep_compare_non_equal_dictionaries_do_not_contain_disclaimer():
		var a1 = [[{'a': 2}], [{'b': 3}], [{'c': 4}]]
		var a2 = [[{'a': 1}], [{'b': 2}], [{'c': 2}]]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		assert_eq(diff.summary.find('reference'), -1, diff.summary)


class TestDictionaryCompareResultInterace:
	extends 'res://addons/gut/test.gd'

	var DiffTool = GutUtils.DiffTool

	func test_cannot_set_summary():
		var diff = DiffTool.new({},{}, GutUtils.DIFF.DEEP)
		diff.summary = 'the summary'
		assert_ne(diff.summary,  'the summary')

	func test_summary_prop_returns_summarize():
		var diff = DiffTool.new({},{}, GutUtils.DIFF.DEEP)
		assert_not_null(diff.summary)

	func test_cannot_set_are_equal():
		var diff = DiffTool.new({},{}, GutUtils.DIFF.DEEP)
		diff.are_equal = 'asdf'
		assert_eq(diff.are_equal, true)

	func test_are_equal_prop_returns_result_of_diff():
		var diff = DiffTool.new({},{}, GutUtils.DIFF.DEEP)
		assert_eq(diff.are_equal, true)

	func test_get_different_count_returns_correct_number():
		var diff = DiffTool.new({'a':1, 'b':2, 'c':3}, {'a':'a', 'b':2, 'c':'c'}, GutUtils.DIFF.DEEP)
		assert_eq(diff.get_different_count(), 2)

	func test_get_total_count_returns_correct_number():
		var diff = DiffTool.new({'a':1, 'b':2, 'c':3}, {'aa':9, 'b':2, 'cc':10}, GutUtils.DIFF.DEEP)
		assert_eq(diff.get_total_count(), 5)

	func test_get_short_summary_includes_x_of_y_keys_when_different():
		var diff = DiffTool.new({'a':1, 'b':2, 'c':3}, {'aa':9, 'b':2, 'cc':10}, GutUtils.DIFF.DEEP)
		assert_string_contains(diff.get_short_summary(), '4 of 5')

	func test_get_short_summary_does_not_include_x_of_y_when_equal():
		var diff = DiffTool.new({}, {}, GutUtils.DIFF.DEEP)
		assert_eq(diff.get_short_summary().find(' of '), -1, diff.get_short_summary())
		assert_string_contains(diff.get_short_summary(), '==')

	func test_brackets():
		var diff = DiffTool.new({}, {})
		assert_eq(diff.get_brackets().open, '{', 'open')
		assert_eq(diff.get_brackets().close, '}', 'close')


class TestDictionaryDiff:
	extends 'res://addons/gut/test.gd'

	var DiffTool = GutUtils.DiffTool

	func test_can_init_with_two_dictionaries():
		var dd = DiffTool.new({}, {})
		assert_not_null(dd)

	func test_constructor_defaults_diff_type_to_deep():
		var diff = DiffTool.new({}, {})
		assert_eq(diff.get_diff_type(), GutUtils.DIFF.DEEP)

	func test_get_differences_returns_empty_array_when_matching():
		var dd = DiffTool.new({'a':'asdf'}, {'a':'asdf'})
		assert_eq(dd.get_differences().keys(), [])

	func test_get_differences_returns_non_matching_keys():
		var dd = DiffTool.new({'a':'asdf', 'b':1}, {'a':'asdf', 'b':2})
		assert_eq(dd.get_differences().keys(), ['b'])

	func test_get_differetn_keys_returns_missing_indexes_in_d2():
		var dd = DiffTool.new({'a':'asdf', 'b':1}, {'a':'asdf'})
		assert_eq(dd.get_differences().keys(), ['b'])

	func test_get_different_keys_returns_missing_indexes_in_d1():
		var dd = DiffTool.new({'a':'asdf'}, {'a':'asdf', 'b':1})
		assert_eq(dd.get_differences().keys(), ['b'])

	func test_get_differences_works_with_different_datatypes():
		var d1 = {'a':1, 'b':'two', 'c':autofree(Node2D.new())}
		var d2 = {'a':1.0, 'b':2, 'c':GutUtils.Strutils.new()}
		var dd = DiffTool.new(d1, d2)
		assert_eq(dd.get_differences().keys(), ['a', 'b', 'c'])

	func test_are_equal_true_for_matching_dictionaries():
		assert_true(DiffTool.new({}, {}).are_equal, 'empty')
		assert_true(DiffTool.new({'a':1}, {'a':1}).are_equal, 'same')
		assert_true(DiffTool.new({'a':1, 'b':2}, {'b':2, 'a':1}).are_equal, 'different order')


	func test_sub_dictionary_compare_when_equal():
		var d1 = {'a':1, 'b':{'a':99}}
		var d2 = {'a':1, 'b':{'a':99}}
		var dd = DiffTool.new(d1, d2)
		assert_true(dd.are_equal, dd.summarize())

	func test_sub_dictionary_compare_when_not_equal():
		var d1 = {'a':1, 'b':{'c':88, 'd':22, 'f':{'g':1, 'h':200}}, 'z':{}, 'dne_in_d2':'asdf'}
		var d2 = {'a':1, 'b':{'c':99, 'e':'letter e', 'f':{'g':1, 'h':2}}, 'z':{}}
		var dd = DiffTool.new(d1, d2)
		assert_false(dd.are_equal, dd.summarize())
		assert_eq(dd.get_total_count(), 4, 'total key count')
		assert_eq(dd.get_different_count(), 2, 'total different count')

	func test_sub_dictionary_missing_in_other():
		var d1 = {'a': 1, 'dne_in_d2':{'x':'x', 'y':'y', 'z':'z'}, 'r':1}
		var d2 = {'a': 2, 'dne_in_d1':{'xx':'x', 'yy':'y', 'zz':'z'}, 'r':2}
		var diff = DiffTool.new(d1, d2)
		var summary = diff.summarize()
		assert_string_contains(summary, 'key>' + ' !=')
		assert_string_contains(summary, ' != ' + '<missing')


	func test_dictionary_key_and_non_dictionary_key():
		var d1 = {'a':1, 'b':{'c':1}}
		var d2 = {'a':1, 'b':22}
		var diff = DiffTool.new(d1, d2)
		assert_false(diff.are_equal, diff.summarize())

	func test_ditionaries_in_arrays():
		var d1 = {'a':[{'b':1}]}
		var d2 = {'a':[{'b':1}]}
		var diff = DiffTool.new(d1, d2)
		assert_true(diff.are_equal, diff.summarize())


	func test_when_deep_diff_then_different_arrays_contains_DiffTool():
		var d1 = {'a':[1, 2, 3]}
		var d2 = {'a':[3, 4, 5]}
		var diff = DiffTool.new(d1, d2, GutUtils.DIFF.DEEP)
		assert_is(diff.differences['a'], GutUtils.DiffTool)


	func test_large_differences_in_sub_arrays_does_not_exceed_max_differences_shown():
		var d1 = {'a':[], 'b':[]}
		var d2 = {'a':[], 'b':[]}
		for i in range(200):
			d1['a'].append(i)
			d2['a'].append(i + 1)

			d1['b'].append(i)
			d2['b'].append(i + 1)

		var diff = DiffTool.new(d1, d2, GutUtils.DIFF.DEEP)
		diff.max_differences = 10
		assert_lt(diff.summary.split("\n").size(), 50, diff.summary)
		assert_false(diff.are_equal)


