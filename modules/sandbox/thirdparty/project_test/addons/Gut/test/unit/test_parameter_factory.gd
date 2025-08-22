extends GutTest

class TestNamedParameters:
	extends GutTest

	func test_creates_array_of_dictionaries():
		var got = ParameterFactory.named_parameters(['a', 'b'], [[1, 2], [3, 4]])
		assert_typeof(got[0], TYPE_DICTIONARY)

	# spot check values.
	func test_gets_what_you_expect_back():
		var names = ['a', 'b', 'c']
		var vals = [[1, 2, 3], ['one', 'two', 'three'], [4, 'five', 6]]
		var got = ParameterFactory.named_parameters(names, vals)
		assert_eq(got[0].a, 1, '0.a')
		assert_eq(got[1]['c'], 'three', '1.c')
		assert_eq(got[2].b, 'five', '2.b')
		assert_eq(got[2]['a'], 4, '2.a')

	func test_when_less_names_than_values_values_are_ignored():
		var names = ['a', 'b']
		var vals = [[1, 2, 3], [4, 5, 6]]
		var got = ParameterFactory.named_parameters(names, vals)
		assert_eq(got[0].size(), 2)

	func test_when_less_values_then_nulls_are_filled_in():
		var names = ['a', 'b','c']
		var vals = [[1, 2], [3, 4]]
		var got = ParameterFactory.named_parameters(names, vals)
		assert_eq(got[0].size(), 3, 'size')
		assert_null(got[0]['c'], 'c is null')

	func test_fills_in_gaps_when_param_is_not_array():
		var names = ['a', 'b','c']
		var vals = [[1, 2], 'a', [3, 4]]
		var got = ParameterFactory.named_parameters(names, vals)
		assert_eq(got[1].size(), 3, 'size')
		assert_null(got[1].b, 'b is null')
