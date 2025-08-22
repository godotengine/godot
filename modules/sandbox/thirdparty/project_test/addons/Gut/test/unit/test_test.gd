extends GutTest
# ------------------------------------------------------------------------------
# Tests test.gd.  test.gd contains all the asserts and is the class that all
# test scripts inherit from.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class BaseTestClass:
	extends GutInternalTester
	# !! Use this for debugging to see the results of all the subtests that
	# are run using assert_fail_pass, assert_fail and assert_pass that are
	# built into this class
	var _print_all_subtests = false

	# GlobalReset(gr) variables to be used by tests.
	# The values of these are reset in the setup or
	# teardown methods.
	var gr = {
		test = null,
		signal_object = null,
		test_with_gut = null
	}



	# #############
	# Seutp/Teardown
	# #############
	func before_each():
		# Everything in here uses the same logger (the one in `g`) since there
		# should not be any times when they would need to be different and
		# `new_gut` sets up the logger to be more quiet.
		var g = autofree(new_gut(_print_all_subtests))
		g.log_level = 3

		gr.test = Test.new()
		gr.test.set_logger(g.logger)

		gr.test_with_gut = Test.new()
		gr.test_with_gut.gut = g
		gr.test_with_gut.set_logger(g.logger)
		add_child(gr.test_with_gut.gut)

	func after_each():
		gr.test_with_gut.gut.get_spy().clear()

		gr.test.free()
		gr.test = null
		gr.test_with_gut.gut.free()
		gr.test_with_gut.free()


# ------------------------------------------------------------------------------
class TestMiscTests:
	extends BaseTestClass

	func test_script_object_added_to_tree():
		gr.test.assert_ne(get_tree(), null, "The tree should not be null if we are added to it")
		assert_pass(gr.test)

	func test_get_set_logger():
		assert_ne(gr.test.get_logger(), null)
		var dlog = double(GutLogger).new()
		gr.test.set_logger(dlog)
		assert_eq(gr.test.get_logger(), dlog)

	func test_when_leaves_tree_awaiter_is_freed():
		add_child(gr.test)
		remove_child(gr.test)
		await wait_physics_frames(10)
		assert_freed(gr.test._awaiter, 'awaiter')


	# -------
	# Spot check some type comparisons, these were all causing errors.  These
	# were adjusted from issue 510 sample code.
	# -------
	func test_rect2i_basis():
		gr.test.assert_ne(Rect2i(), Basis())
		assert_fail(gr.test)

	func test_vector2i_3i():
		gr.test.assert_eq(Vector2i(), Vector3i())
		assert_fail(gr.test)

	func test_vector4_4i():
		gr.test.assert_gt(Vector4(), Vector4i())
		assert_fail(gr.test)

	func test_transform2d_projection():
		gr.test.assert_ne(Transform2D(), Projection())
		assert_fail(gr.test)

	func test_callable_signal():
		gr.test.assert_ne(Callable(), Signal())
		assert_fail(gr.test)



# ------------------------------------------------------------------------------
class TestAssertEq:
	extends BaseTestClass

	func test_passes_when_integer_equal():
		gr.test.assert_eq(1, 1)
		assert_pass(gr.test)

	func test_fails_when_number_not_equal():
		gr.test.assert_eq(1, 2)
		assert_fail(gr.test, 1, "Should fail.  1 != 2")

	func test_passes_when_float_eq():
		gr.test.assert_eq(1.0, 1.0)
		assert_pass(gr.test)

	func test_fails_when_float_eq_fail():
		gr.test.assert_eq(.19, 1.9)
		assert_fail(gr.test)

	func test_fails_when_comparing_float_cast_as_int():
		# int cast will make it 0
		gr.test.assert_eq(int(0.5), 0.5)
		assert_fail(gr.test)

	func test_passes_when_cast_int_expression_to_float():
		var i = 2
		gr.test.assert_eq(5 / float(i), 2.5)
		assert_pass(gr.test)

	func test_fails_when_string_not_equal():
		gr.test.assert_eq("one", "two", "Should Fail")
		assert_fail(gr.test)

	func test_passes_when_string_equal():
		gr.test.assert_eq("one", "one", "Should Pass")
		assert_pass(gr.test)

	func test_warns_when_comparing_float_and_int():
		gr.test.assert_eq(1.0, 1, 'Should pass and warn')
		assert_warn(gr.test)

	var array_vals = [
		[[1, 2, 3], ['1', '2', '3'], false],
		[[4, 5, 6], [4, 5, 6], true],
		[[10, 20.0, 30], [10.0, 20, 30.0], false],
		[[1, 2], [1, 2, 3, 4, 5], false],
		[[1, 2, 3, 4, 5], [1, 2], false],
		[[{'a':1}], [{'a':1}], true],
		[[[1, 2], [3, 4]], [[5, 6], [7, 8]], false],
		[
			[[1, [2, 3]], [4, [5, 6]]],
			[[1, [2, 'a']], [4, ['b', 6]]],
			false
		]
	]
	func test_with_array(p = use_parameters(array_vals)):
		gr.test.assert_eq(p[0], p[1])
		if(p[2]):
			assert_pass(gr.test)
		else:
			assert_fail(gr.test)

	func test_with_dictionary_references():
		var d = {}
		var d_pointer = d
		gr.test.assert_eq(d, d_pointer)
		assert_pass(gr.test)

	func test_dictionary_are_compared_by_value():
		var d  = {'a':1}
		var d2 = {'a':1}
		gr.test.assert_eq(d, d2)
		assert_pass(gr.test)

	func test_comparing_callable_does_not_error():
		gr.test.assert_eq(test_comparing_callable_does_not_error, '1')
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, 'Cannot compare CALLABLE')



# ------------------------------------------------------------------------------
class TestAssertNe:
	extends BaseTestClass

	func test_passes_with_integers_not_equal():
		gr.test.assert_ne(1, 2)
		assert_pass(gr.test)

	func test_fails_with_integers_equal():
		gr.test.assert_ne(1, 1, "Should fail")
		assert_fail(gr.test, 1, '1 = 1')

	func test_passes_with_floats_not_equal():
		gr.test.assert_ne(0.9, .009)
		assert_pass(gr.test)

	func test_passes_with_strings_not_equal():
		gr.test.assert_ne("one", "two", "Should Pass")
		assert_pass(gr.test)

	func test_fails_with_strings_equal():
		gr.test.assert_ne("one", "one", "Should Fail")
		assert_fail(gr.test)

	var array_vals = [
		[[1, 2, 3], ['1', '2', '3'], true],
		[[1, 2, 3], [1, 2, 3], false],
		[[1, 2.0, 3], [1.0, 2, 3.0], true]
	]
	func test_with_array(p = use_parameters(array_vals)):
		gr.test.assert_ne(p[0], p[1])
		if(p[2]):
			assert_pass(gr.test)
		else:
			assert_fail(gr.test)

	func test_with_dictionary_references():
		var d = {}
		var d_pointer = d
		gr.test.assert_ne(d, d_pointer)
		assert_fail(gr.test)

	func test_dictionary_are_compared_by_value():
		var d  = {'a':2}
		var d2 = {'a':1}
		gr.test.assert_ne(d, d2)
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
class TestAssertAlmostEq:
	extends BaseTestClass

	func test_passes_with_integers_equal():
		gr.test.assert_almost_eq(2, 2, 0, "Should pass, 2 == 2 +/- 0")
		assert_pass(gr.test)

	func test_passes_with_integers_almost_within_range():
		gr.test.assert_almost_eq(1, 2, 1, "Should pass, 1 == 2 +/- 1")
		gr.test.assert_almost_eq(3, 2, 1, "Should pass, 3 == 2 +/- 1")
		assert_pass(gr.test, 2)

	func test_fails_with_integers_outside_range():
		gr.test.assert_almost_eq(0, 2, 1, "Should fail, 0 != 2 +/- 1")
		gr.test.assert_almost_eq(4, 2, 1, "Should fail, 4 != 2 +/- 1")
		assert_fail(gr.test, 2)

	func test_passes_with_floats_within_range():
		gr.test.assert_almost_eq(1.000, 1.000, 0.001, "Should pass, 1.000 == 1.000 +/- 0.001")
		gr.test.assert_almost_eq(1.001, 1.000, 0.001, "Should pass, 1.001 == 1.000 +/- 0.001")
		gr.test.assert_almost_eq(.999, 1.000, 0.001, "Should pass, .999 == 1.000 +/- 0.001")
		assert_pass(gr.test, 3)

	func test_fails_with_floats_outside_range():
		gr.test.assert_almost_eq(2.002, 2.000, 0.001, "Should fail, 2.002 == 2.000 +/- 0.001")
		gr.test.assert_almost_eq(1.998, 2.000, 0.001, "Should fail, 1.998 == 2.000 +/- 0.001")
		assert_fail(gr.test, 2)

	func test_passes_with_integers_within_float_range():
		gr.test.assert_almost_eq(2, 1.9, .5, 'Should pass, 1.5 < 2 < 2.4')
		assert_pass(gr.test)

	func test_passes_with_float_within_integer_range():
		gr.test.assert_almost_eq(2.5, 2, 1, 'Should pass, 1 < 2.5 < 3')
		assert_pass(gr.test)

	func test_passes_with_vector2s_eq():
		gr.test.assert_almost_eq(Vector2(1.0, 1.0), Vector2(1.0, 1.0), Vector2(0.0, 0.0), "Should pass, Vector2(1.0, 1.0) == Vector2(1.0, 1.0) +/- Vector2(0.0, 0.0)")
		assert_pass(gr.test)

	func test_fails_with_vector2s_ne():
		gr.test.assert_almost_eq(Vector2(1.0, 1.0), Vector2(2.0, 2.0), Vector2(0.0, 0.0), "Should fail, Vector2(1.0, 1.0) == Vector2(2.0, 2.0) +/- Vector2(0.0, 0.0)")
		assert_fail(gr.test)

	func test_passes_with_vector2s_almost_eq():
		gr.test.assert_almost_eq(Vector2(1.0, 1.0), Vector2(2.0, 2.0), Vector2(1.0, 1.0), "Should pass, Vector2(1.0, 1.0) == Vector2(2.0, 2.0) +/- Vector2(1.0, 1.0)")
		assert_pass(gr.test)

	func test_fails_with_vector2s_y_ne():
		gr.test.assert_almost_eq(Vector2(1.0, 2.0), Vector2(1.0, 1.0), Vector2(0.0, 0.0), "Should fail, Vector2(1.0, 2.0) == Vector2(1.0, 1.0) +/- Vector2(0.0, 0.0)")
		assert_fail(gr.test)

	func test_fails_with_vector3s_z_ne():
		gr.test.assert_almost_eq(Vector3(1.0, 1.0, 2.0), Vector3(1.0, 1.0, 1.0), Vector3(0.0, 0.0, 0.0), "Should fail, Vector3(1.0, 1.0, 2.0) == Vector3(1.0, 1.0, 1.0) +/- Vector2(0.0, 0.0, 0.0)")
		assert_fail(gr.test)

	func test_fails_with_vector3s_y_z_ne():
		gr.test.assert_almost_eq(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(0.0, 0.0, 0.0), "Should fail, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(0.0, 0.0, 0.0)")
		assert_fail(gr.test)

	func test_fails_with_vector2s_y_outside_range():
		gr.test.assert_almost_eq(Vector2(1.0, 3.0), Vector2(1.0, 1.0), Vector2(1.0, 1.0), "Should fail, Vector2(1.0, 3.0) == Vector2(1.0, 1.0) +/- Vector2(1.0, 1.0)")
		assert_fail(gr.test)

	func test_fails_with_vector3s_z_outside_range():
		gr.test.assert_almost_eq(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(1.0, 1.0, 1.0), "Should fail, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(1.0, 1.0, 1.0)")
		assert_fail(gr.test)

	func test_passes_with_vector3s_y_z_within_range():
		gr.test.assert_almost_eq(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(2.0, 2.0, 2.0), "Should pass, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(2.0, 2.0, 2.0)")
		assert_pass(gr.test)

	func test_passes_with_vector4s_within_range():
		gr.test.assert_almost_eq(Vector4(1.0, 2.0, 3.0, 1.0), Vector4(1.0, 1.0, 1.0, 1.0), Vector4(2.0, 2.0, 2.0, 2.0), "Should pass")
		assert_pass(gr.test)

	func test_fails_with_vector4_outside_range():
		gr.test.assert_almost_eq(Vector4(1.0, 2.0, 3.0, 1.0), Vector4(9.0, 1.0, 1.0, 1.0), Vector4(2.0, 2.0, 2.0, 2.0), "Should fail")
		assert_fail(gr.test)

	func test_fails_when_vector4_y_outside_range():
		gr.test.assert_almost_eq(Vector4(1.0, 2.0, 99.0, 1.0), Vector4(1.0, 1.0, 1.0, 1.0), Vector4(2.0, 2.0, 2.0, 2.0), "Should fail")
		assert_fail(gr.test)

	func test_fail_message_includes_extra_precision_for_floats():
		gr.test.assert_almost_eq(.500000000012300000, .499, .001)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')

	func test_fail_message_includes_extra_precision_for_vector2():
		var got = Vector2(1.0001230, 1.003450)
		var expected = Vector2(.999, .999)
		var pad = Vector2(.001, .001)
		gr.test.assert_almost_eq(got, expected, pad)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')
		assert_fail_msg_contains(gr.test, '03450')

	func test_fail_message_includes_extra_precision_for_vector3():
		var got = Vector3(1.0001230, 1.003450, 1.0032101)
		var expected = Vector3(.999, .999, .999)
		var pad = Vector3(.001, .001, .001)
		gr.test.assert_almost_eq(got, expected, pad)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')
		assert_fail_msg_contains(gr.test, '03450')
		assert_fail_msg_contains(gr.test, '03210')


# ------------------------------------------------------------------------------
class TestAssertAlmostNe:
	extends BaseTestClass

	func test_pass_with_integers_not_equal():
		gr.test.assert_almost_ne(1, 2, 0, "Should pass, 1 != 2 +/- 0")
		assert_pass(gr.test)

	func test_fails_with_integers_equal():
		gr.test.assert_almost_ne(2, 2, 0, "Should fail, 2 == 2 +/- 0")
		assert_fail(gr.test)

	func test_passes_with_integers_outside_range():
		gr.test.assert_almost_ne(1, 3, 1, "Should pass, 1 != 3 +/- 1")
		assert_pass(gr.test)

	func test_fails_with_integers_within_range():
		gr.test.assert_almost_ne(2, 3, 1, "Should fail, 2 == 3 +/- 1")
		assert_fail(gr.test)

	func test_passes_with_floats_outside_range():
		gr.test.assert_almost_ne(1.000, 2.000, 0.001, "Should pass, 1.000 != 2.000 +/- 0.001")
		assert_pass(gr.test)

	func test_fails_with_floats_eq():
		gr.test.assert_almost_ne(1.000, 1.000, 0.001, "Should fail, 1.000 == 1.000 +/- 0.001")
		assert_fail(gr.test)

	func test_fails_with_floats_within_range():
		gr.test.assert_almost_ne(1.000, 2.000, 1.000, "Should fail, 1.000 == 2.000 +/- 1.000")
		assert_fail(gr.test)

	func test_passes_with_vector2s_outside_range():
		gr.test.assert_almost_ne(Vector2(1.0, 1.0), Vector2(2.0, 2.0), Vector2(0.0, 0.0), "Should pass, Vector2(1.0, 1.0) != Vector2(2.0, 2.0) +/- Vector2(0.0, 0.0)")
		assert_pass(gr.test)

	func test_fails_with_vector2s_eq():
		gr.test.assert_almost_ne(Vector2(1.0, 1.0), Vector2(1.0, 1.0), Vector2(0.0, 0.0), "Should fail, Vector2(1.0, 1.0) == Vector2(1.0, 1.0) +/- Vector2(0.0, 0.0)")
		assert_fail(gr.test)

	func test_passes_with_vector2s_almost_outside_range():
		gr.test.assert_almost_ne(Vector2(1.0, 1.0), Vector2(2.0, 2.0), Vector2(0.9, 0.9), "Should pass, Vector2(1.0, 1.0) == Vector2(2.0, 2.0) +/- Vector2(0.9, 0.9)")
		assert_pass(gr.test)

	func test_passes_with_vector2s_y_ne():
		gr.test.assert_almost_ne(Vector2(1.0, 2.0), Vector2(1.0, 1.0), Vector2(0.0, 0.0), "Should pass, Vector2(1.0, 2.0) == Vector2(1.0, 1.0) +/- Vector2(0.0, 0.0)")
		assert_pass(gr.test)

	func test_passes_with_vector3s_z_ne():
		gr.test.assert_almost_ne(Vector3(1.0, 1.0, 2.0), Vector3(1.0, 1.0, 1.0), Vector3(0.0, 0.0, 0.0), "Should pass, Vector3(1.0, 1.0, 2.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(0.0, 0.0, 0.0)")
		assert_pass(gr.test)

	func test_passes_with_vector3s_y_z_ne():
		gr.test.assert_almost_ne(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(0.0, 0.0, 0.0), "Should pass, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(0.0, 0.0, 0.0)")
		assert_pass(gr.test)

	func test_passes_with_vector2s_y_outside_range():
		gr.test.assert_almost_ne(Vector2(1.0, 3.0), Vector2(1.0, 1.0), Vector2(1.0, 1.0), "Should pass, Vector2(1.0, 3.0) == Vector2(1.0, 1.0) +/- Vector2(1.0, 1.0)")
		assert_pass(gr.test)

	func test_passes_with_vector3s_z_outside_range():
		gr.test.assert_almost_ne(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(1.0, 1.0, 1.0), "Should pass, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(1.0, 1.0, 1.0)")
		assert_pass(gr.test)

	func test_fails_with_vector3s_y_z_within_range():
		gr.test.assert_almost_ne(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 1.0, 1.0), Vector3(2.0, 2.0, 2.0), "Should fail, Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 1.0, 1.0) +/- Vector3(2.0, 2.0, 2.0)")
		assert_fail(gr.test)

	func test_fail_message_includes_extra_precision_for_floats():
		gr.test.assert_almost_ne(.500000000012300000, .5, .001)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')

# ------------------------------------------------------------------------------
class TestAssertGt:
	extends BaseTestClass

	# The assert_gt method asserts that `got` is greater than `expected`.
	var test_data = ParameterFactory.named_parameters(
		['got', 'expected', 'pass_fail_gt'],
		[
			[2, 1, SHOULD_PASS],
			[2, 2, SHOULD_FAIL],
			[2, 3, SHOULD_FAIL],
			[2.2, 2.1, SHOULD_PASS],
			[2.2, 2.2, SHOULD_FAIL],
			[2.2, 2.3, SHOULD_FAIL],
			['B', 'A', SHOULD_PASS],
			['B', 'B', SHOULD_FAIL],
			['B', 'C', SHOULD_FAIL]
		])

	func test_assert_gt(p = use_parameters(test_data)):
		gr.test.assert_gt(p.got, p.expected, p.pass_fail_gt)
		if (p.pass_fail_gt == SHOULD_PASS):
			assert_pass(gr.test, 1, str(p.got, ' > ', p.expected))
		else:
			assert_fail(gr.test, 1, str(p.got, ' > ', p.expected))


# ------------------------------------------------------------------------------
class TestAssertGte:
	extends BaseTestClass

	# The assert_gte method asserts that `got` is greater than or equal to `expected`.
	var test_data = ParameterFactory.named_parameters(
		['got', 'expected', 'pass_fail_gte'],
		[
			[2, 1, SHOULD_PASS],
			[2, 2, SHOULD_PASS],
			[2, 3, SHOULD_FAIL],
			[2.2, 2.1, SHOULD_PASS],
			[2.2, 2.2, SHOULD_PASS],
			[2.2, 2.3, SHOULD_FAIL],
			['B', 'A', SHOULD_PASS],
			['B', 'B', SHOULD_PASS],
			['B', 'C', SHOULD_FAIL]
		])

	func test_assert_gte(p = use_parameters(test_data)):
		gr.test.assert_gte(p.got, p.expected, p.pass_fail_gte)
		if (p.pass_fail_gte == SHOULD_PASS):
			assert_pass(gr.test, 1, str(p.got, ' >= ', p.expected))
		else:
			assert_fail(gr.test, 1, str(p.got, ' >= ', p.expected))

# ------------------------------------------------------------------------------
class TestAssertLt:
	extends BaseTestClass

	# The assert_lt method asserts that `got` is less than `expected`.
	var test_data = ParameterFactory.named_parameters(
		['got', 'expected', 'pass_fail_lt'],
		[
			[2, 1, SHOULD_FAIL],
			[2, 2, SHOULD_FAIL],
			[2, 3, SHOULD_PASS],
			[2.2, 2.1, SHOULD_FAIL],
			[2.2, 2.2, SHOULD_FAIL],
			[2.2, 2.3, SHOULD_PASS],
			['B', 'A', SHOULD_FAIL],
			['B', 'B', SHOULD_FAIL],
			['B', 'C', SHOULD_PASS]
		])

	func test_assert_lt(p = use_parameters(test_data)):
		gr.test.assert_lt(p.got, p.expected, p.pass_fail_lt)
		if (p.pass_fail_lt == SHOULD_PASS):
			assert_pass(gr.test, 1, str(p.got, ' < ', p.expected))
		else:
			assert_fail(gr.test, 1, str(p.got, ' < ', p.expected))

# ------------------------------------------------------------------------------
class TestAssertLte:
	extends BaseTestClass

	# The assert_lte method asserts that `got` is less than or equal to `expected`.
	var test_data = ParameterFactory.named_parameters(
		['got', 'expected', 'pass_fail_lte'],
		[
			[2, 1, SHOULD_FAIL],
			[2, 2, SHOULD_PASS],
			[2, 3, SHOULD_PASS],
			[2.2, 2.1, SHOULD_FAIL],
			[2.2, 2.2, SHOULD_PASS],
			[2.2, 2.3, SHOULD_PASS],
			['B', 'A', SHOULD_FAIL],
			['B', 'B', SHOULD_PASS],
			['B', 'C', SHOULD_PASS]
		])

	func test_assert_lte(p = use_parameters(test_data)):
		gr.test.assert_lte(p.got, p.expected, p.pass_fail_lte)
		if (p.pass_fail_lte == SHOULD_PASS):
			assert_pass(gr.test, 1, str(p.got, ' <= ', p.expected))
		else:
			assert_fail(gr.test, 1, str(p.got, ' <= ', p.expected))

# ------------------------------------------------------------------------------
# TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestAssertBetween:
	extends BaseTestClass

	func test_between_with_number_between():
		gr.test.assert_between(2, 1, 3, "Should pass, 2 between 1 and 3")
		assert_pass(gr.test, 1, "Should pass, 2 between 1 and 3")

	func test_between_with_number_lt():
		gr.test.assert_between(0, 1, 3, "Should fail")
		assert_fail(gr.test, 1, '0 not between 1 and 3')

	func test_between_with_number_gt():
		gr.test.assert_between(4, 1, 3, "Should fail")
		assert_fail(gr.test, 1, '4 not between 1 and 3')

	func test_between_with_number_at_high_end():
		gr.test.assert_between(3, 1, 3, "Should pass")
		assert_pass(gr.test, 1, '3 is between 1 and 3')

	func test_between_with_number_at_low_end():
		gr.test.assert_between(1, 1, 3, "Should pass")
		assert_pass(gr.test, 1, '1 between 1 and 3')

	func test_between_with_invalid_number_range():
		gr.test.assert_between(4, 8, 0, "Should fail")
		assert_fail(gr.test, 1, '8 is starting number and is not less than 0')

	func test_between_with_string_between():
		gr.test.assert_between('b', 'a', 'c', "Should pass, 2 between 1 and 3")
		assert_pass(gr.test)

	func test_between_with_string_lt():
		gr.test.assert_between('a', 'b', 'd', "Should fail")
		assert_fail(gr.test)

	func test_between_with_string_gt():
		gr.test.assert_between('z', 'a', 'c', "Should fail")
		assert_fail(gr.test)

	func test_between_with_string_at_high_end():
		gr.test.assert_between('c', 'a', 'c', "Should pass")
		assert_pass(gr.test)

	func test_between_with_string_at_low_end():
		gr.test.assert_between('a', 'a', 'c', "Should pass")
		assert_pass(gr.test)

	func test_between_with_invalid_string_range():
		gr.test.assert_between('q', 'z', 'a', "Should fail")
		assert_fail(gr.test)

	func test_fail_message_includes_extra_precision_for_floats():
		gr.test.assert_between(.50000000012300100, .498, .5)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')


# ------------------------------------------------------------------------------
class TestAssertNotBetween:
	extends BaseTestClass

	func test_with_number_lt():
		gr.test.assert_not_between(1, 2, 3, "Should pass, 1 not between 2 and 3")
		assert_pass(gr.test)

	func test_with_number_gt():
		gr.test.assert_not_between(4, 1, 3, "Should pass, 4 not between 1 and 3")
		assert_pass(gr.test, 1, '4 not between 1 and 3')

	func test_with_number_at_low_end():
		gr.test.assert_not_between(1, 1, 3, "Should pass: exclusive not between")
		assert_pass(gr.test, 1, '1 not between 1 and 3, exclusively')

	func test_with_number_at_high_end():
		gr.test.assert_not_between(3, 1, 3, "Should pass: exclusive not between")
		assert_pass(gr.test, 1, '3 not between 1 and 3, exclusively')

	func test_with_invalid_number_range():
		gr.test.assert_not_between(4, 8, 0, "Should fail")
		assert_fail(gr.test, 1, '8 is starting number and is not less than 0')

	func test_with_string_between():
		gr.test.assert_not_between('b', 'a', 'c', "Should fail, b is between a and c")
		assert_fail(gr.test)

	func test_with_string_lt():
		gr.test.assert_not_between('a', 'b', 'd', "Should pass")
		assert_pass(gr.test)

	func test_with_string_gt():
		gr.test.assert_not_between('z', 'a', 'c', "Should pass")
		assert_pass(gr.test)

	func test_with_string_at_high_end():
		gr.test.assert_not_between('c', 'a', 'c', "Should pass: exclusive not between")
		assert_pass(gr.test)

	func test_with_string_at_low_end():
		gr.test.assert_not_between('a', 'a', 'c', "Should pass: exclusive not between")
		assert_pass(gr.test)

	func test_with_invalid_string_range():
		gr.test.assert_not_between('q', 'z', 'a', "Should fail: Invalid range")
		assert_fail(gr.test)

	func test_fail_message_includes_extra_precision_for_floats():
		gr.test.assert_not_between(.50000000012300100, .498, .51)
		assert_fail(gr.test)
		assert_fail_msg_contains(gr.test, '01230')


# ------------------------------------------------------------------------------
class TestAssertTrue:
	extends BaseTestClass

	func test_passes_with_true():
		gr.test.assert_true(true, "Should pass, true is true")
		assert_pass(gr.test)

	func test_fails_with_false():
		gr.test.assert_true(false, "Should fail")
		assert_fail(gr.test)

	func test_text_is_optional():
		gr.test.assert_true(true)
		assert_pass(gr.test)

	func test_fails_with_non_bools():
		gr.test.assert_true('asdf')
		gr.test.assert_true(1)
		assert_fail(gr.test, 2)


# ------------------------------------------------------------------------------
class TestAssertFalse:
	extends BaseTestClass

	func test_text_is_optional():
		gr.test.assert_false(false)
		assert_pass(gr.test)

	func test_fails_with_true():
		gr.test.assert_false(true, "Should fail")
		assert_fail(gr.test)

	func test_passes_with_false():
		gr.test.assert_false(false, "Should pass")
		assert_pass(gr.test)

	func test_fails_with_non_bools():
		gr.test.assert_false(null)
		gr.test.assert_false(0)
		assert_fail(gr.test, 2)


# ------------------------------------------------------------------------------
class TestAssertHas:
	extends BaseTestClass

	func test_passes_when_array_has_element():
		var array = [0]
		gr.test.assert_has(array, 0, 'It should have zero')
		assert_pass(gr.test)

	func test_fails_when_it_does_not_have_element():
		var array = [0]
		gr.test.assert_has(array, 1, 'Should not have it')
		assert_fail(gr.test)

	func test_assert_not_have_passes_when_not_in_there():
		var array = [0, 3, 5]
		gr.test.assert_does_not_have(array, 2, 'Should not have it.')
		assert_pass(gr.test)

	func test_assert_not_have_fails_when_in_there():
		var array = [1, 10, 20]
		gr.test.assert_does_not_have(array, 20, 'Should not have it.')
		assert_fail(gr.test)


# ------------------------------------------------------------------------------
class TestFailingDatatypeChecks:
	extends BaseTestClass

	func test_dt_string_number_eq():
		gr.test.assert_eq('1', 1)
		assert_fail(gr.test)

	func test_dt_string_number_ne():
		gr.test.assert_ne('2', 1)
		assert_fail(gr.test)

	func test_dt_string_number_assert_gt():
		gr.test.assert_gt('3', 1)
		assert_fail(gr.test)

	func test_dt_string_number_func_assert_lt():
		gr.test.assert_lt('1', 3)
		assert_fail(gr.test)

	func test_dt_string_number_func_assert_between():
		gr.test.assert_between('a', 5, 6)
		gr.test.assert_between(1, 2, 'c')
		assert_fail(gr.test, 2)

	func test_dt_can_compare_to_null():
		gr.test.assert_ne(autofree(Node2D.new()), null)
		gr.test.assert_ne(null, autofree(Node2D.new()))
		assert_pass(gr.test, 2)


# ------------------------------------------------------------------------------
class TestPending:
	extends BaseTestClass

	func test_pending_increments_pending_count():
		gr.test.pending()
		assert_eq(gr.test.get_pending_count(), 1, 'One test should have been marked as pending')

	func test_pending_accepts_text():
		pending("This is a pending test.  You should see this text in the results.")

	func test_pending_does_not_increment_passed():
		gr.test.pending()
		assert_eq(gr.test.get_pass_count(), 0)


# ------------------------------------------------------------------------------
class TestAssertHasMethod:
	extends BaseTestClass

	class NoWantedMethod:
		func irrelevant_method():
			pass

	class HasWantedMethod:
		func wanted_method():
			pass

	func test_fail_if_is_lacking_method():
		var obj = NoWantedMethod.new()
		gr.test.assert_has_method(obj, 'wanted_method')
		assert_fail(gr.test)

	func test_pass_if_has_correct_method():
		var obj = HasWantedMethod.new()
		gr.test.assert_has_method(obj, 'wanted_method')
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
class TestAccessorAsserts:
	extends BaseTestClass

	class NoGetNoSet:
		var _thing = 'nothing'

	class HasGetNotSet:
		func get_thing():
			pass

	class HasGetAndSetThatDontWork:
		func get_thing():
			pass
		func set_thing(new_thing):
			pass

	class HasGetSetThatWorks:
		var _thing = 'something'

		func get_thing():
			return _thing
		func set_thing(new_thing):
			_thing = new_thing

	class HasIsGetter:
		var _flagged = true
		func is_flagged():
			return _flagged
		func set_flagged(isit):
			_flagged = isit

	func test_fail_if_get_set_not_defined():
		var obj = NoGetNoSet.new()
		gr.test.assert_accessors(obj, 'thing', 'something', 'another thing')
		assert_fail(gr.test, 2)

	func test_fail_if_has_get_and_not_set():
		var obj = HasGetNotSet.new()
		gr.test.assert_accessors(obj, 'thing', 'something', 'another thing')
		assert_fail_pass(gr.test, 1, 1)

	func test_fail_if_default_wrong_and_get_dont_work():
		var obj = HasGetAndSetThatDontWork.new()
		gr.test.assert_accessors(obj, 'thing', 'something', 'another thing')
		assert_fail_pass(gr.test, 2, 2)

	func test_fail_if_default_wrong():
		var obj = HasGetSetThatWorks.new()
		gr.test.assert_accessors(obj, 'thing', 'not the right default', 'another thing')
		assert_fail_pass(gr.test, 1, 3)

	func test_pass_if_all_get_sets_are_aligned():
		var obj = HasGetSetThatWorks.new()
		gr.test.assert_accessors(obj, 'thing', 'something', 'another thing')
		assert_pass(gr.test, 4)

	func test_finds_getters_that_start_with_is():
		var obj = HasIsGetter.new()
		gr.test.assert_accessors(obj, 'flagged', true, false)
		assert_pass(gr.test, 4)


# ------------------------------------------------------------------------------
class TestAssertExports:
	extends BaseTestClass

	func should_skip_script():
		return 'Not implemented in 4.0'

	class NoProperty:
		func _unused():
			pass

	class NotEditorProperty:
		var some_property = 1

	class HasCorrectEditorPropertyAndExplicitType:
		@export var int_property: int

	class HasCorrectEditorPropertyAndImplicitType:
		@export var vec2_property = Vector2(0.0, 0.0)

	class HasCorrectEditorPropertyNotType:
		@export var bool_property: bool

	class HasObjectDerivedPropertyType:
		@export var scene_property: PackedScene

	func test_fail_if_property_not_found():
		var obj = NoProperty.new()
		gr.test.assert_exports(obj, "some_property", TYPE_BOOL)
		assert_fail(gr.test)

	func test_fail_if_not_editor_property():
		var obj = NotEditorProperty.new()
		gr.test.assert_exports(obj, "some_property", TYPE_INT)
		assert_fail(gr.test)

	func test_pass_if_editor_property_present_with_correct_explicit_type():
		var obj = HasCorrectEditorPropertyAndExplicitType.new()
		gr.test.assert_exports(obj, "int_property", TYPE_INT)
		assert_pass(gr.test)

	func test_pass_if_editor_property_present_with_correct_implicit_type():
		var obj = HasCorrectEditorPropertyAndImplicitType.new()
		gr.test.assert_exports(obj, "vec2_property", TYPE_VECTOR2)
		assert_pass(gr.test)

	func test_fail_if_editor_property_present_with_incorrect_type():
		var obj = HasCorrectEditorPropertyNotType.new()
		gr.test.assert_exports(obj, "bool_property", TYPE_FLOAT)
		assert_fail(gr.test)

	func test__object_derived_type__exported_as_object_type():
		var obj = HasObjectDerivedPropertyType.new()
		gr.test.assert_exports(obj, "scene_property", TYPE_OBJECT)
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
# TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestAssertFileExists:
	extends BaseTestClass

	func test__assert_file_exists__with_file_dne():
		gr.test_with_gut.assert_file_exists('user://file_dne.txt')
		assert_fail(gr.test_with_gut)

	func test__assert_file_exists__with_file_exists():
		var path = 'user://gut_test_file.txt'
		FileAccess.open(path, FileAccess.WRITE)
		gr.test_with_gut.assert_file_exists(path)
		assert_pass(gr.test_with_gut)


# ------------------------------------------------------------------------------
# TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestAssertFileDne:
	extends BaseTestClass

	func test__assert_file_dne__with_file_dne():
		gr.test_with_gut.assert_file_does_not_exist('user://file_dne.txt')
		assert_pass(gr.test_with_gut)

	func test__assert_file_dne__with_file_exists():
		var path = 'user://gut_test_file2.txt'
		FileAccess.open(path, FileAccess.WRITE)
		gr.test_with_gut.assert_file_does_not_exist(path)
		assert_fail(gr.test_with_gut)


# ------------------------------------------------------------------------------
# # TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestAssertFileEmpty:
	extends BaseTestClass

	func test__assert_file_empty__with_empty_file():
		var path = 'user://gut_test_empty.txt'
		FileAccess.open(path, FileAccess.WRITE)
		gr.test_with_gut.assert_file_empty(path)
		assert_pass(gr.test_with_gut)

	func test__assert_file_empty__with_not_empty_file():
		var path = 'user://gut_test_empty2.txt'
		var f = FileAccess.open(path, FileAccess.WRITE)
		f.store_8(1)
		f.flush()
		gr.test_with_gut.assert_file_empty(path)
		assert_fail(gr.test_with_gut)

	func test__assert_file_empty__fails_when_file_dne():
		var path = 'user://file_dne.txt'
		gr.test_with_gut.assert_file_empty(path)
		assert_fail(gr.test_with_gut)


# ------------------------------------------------------------------------------
# # TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestAssertFileNotEmpty:
	extends BaseTestClass

	func test__assert_file_not_empty__with_empty_file():
		var path = 'user://gut_test_empty3.txt'
		var f = FileAccess.open(path, FileAccess.WRITE)
		gr.test_with_gut.assert_file_not_empty(path)
		assert_fail(gr.test_with_gut)

	func test__assert_file_not_empty__with_populated_file():
		var path = 'user://gut_test_empty4.txt'
		var f = FileAccess.open(path, FileAccess.WRITE)
		f.store_8(1)
		f.flush()
		gr.test_with_gut.assert_file_not_empty(path)
		assert_pass(gr.test_with_gut)

	func test__assert_file_not_empty__fails_when_file_dne():
		var path = 'user://file_dne.txt'
		gr.test_with_gut.assert_file_not_empty(path)
		assert_fail(gr.test_with_gut)



# ------------------------------------------------------------------------------
class TestExtendAsserts:
	extends BaseTestClass

	class BaseClass:
		extends Node2D

	class ExtendsBaseClass:
		extends BaseClass

	class HasSubclass1:
		class SubClass:
			var a = 1

	class HasSubclass2:
		class SubClass:
			var a = 2

	func test_passes_when_class_extends_parent():
		var node2d = autofree(Node2D.new())
		gr.test.assert_is(node2d, Node2D)
		assert_pass(gr.test)

	func test_fails_when_class_does_not_extend_parent():
		var lbl = autofree(Label.new())
		gr.test.assert_is(lbl, TextEdit)
		assert_fail(gr.test)

	func test_fails_with_primitves_and_classes():
		gr.test.assert_is([], Node2D)
		assert_fail(gr.test)

	func test_fails_when_compareing_object_to_primitives():
		gr.test.assert_is(autofree(Node2D.new()), [])
		gr.test.assert_is(autofree(TextEdit.new()), {})
		assert_fail(gr.test, 2)

	func test_fails_with_another_instance():
		var node1 = autofree(Node2D.new())
		var node2 = autofree(Node2D.new())
		gr.test.assert_is(node1, node2)
		assert_fail(gr.test)

	func test_passes_with_deeper_inheritance():
		var eb =autofree(ExtendsBaseClass.new())
		gr.test.assert_is(eb, Node2D)
		assert_pass(gr.test)

	func test_fails_when_class_names_match_but_inheritance_does_not():
		var a = HasSubclass1.SubClass.new()
		var b = HasSubclass2.SubClass.new()
		gr.test.assert_is(a, b)
		assert_fail(gr.test)

	func test_fails_when_class_names_match_but_inheritance_does_not__with_class():
		var a = HasSubclass1.SubClass.new()
		gr.test.assert_is(a, HasSubclass2.SubClass)
		# created bug https://github.com/godotengine/godot/issues/27111 for 3.1
		# TODO remove_at comment after awhile, this appears fixed in 3.2
		assert_fail(gr.test, 1, 'Fails in 3.1, bug has been created.')

	func test_assrt_is_does_not_free_references():
		var ref = RefCounted.new()
		gr.test.assert_is(ref, RefCounted)
		assert_pass(gr.test)

	func test_works_with_resources():
		var res = autofree(Resource.new())
		gr.test.assert_is(res, Resource)
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
class TestAssertTypeOf:
	extends BaseTestClass

	func test_passes_when_object_is_of_type():
		var c = Color(1, 1, 1, 1)
		gr.test.assert_typeof(c, TYPE_COLOR)
		assert_pass(gr.test)

	func test_fails_when_it_is_not():
		var c = Color(1, 1, 1, 1)
		gr.test.assert_typeof(c, TYPE_INT)
		assert_fail(gr.test)

	func test_not_fails_when_object_is_of_type():
		var c = Color(1, 1, 1, 1)
		gr.test.assert_not_typeof(c, TYPE_COLOR)
		assert_fail(gr.test)

	func test_not_passes_when_it_is_not():
		var c = Color(1, 1, 1, 1)
		gr.test.assert_not_typeof(c, TYPE_INT)
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
class TestAssertStringContains:
	extends BaseTestClass

	func test_fails_when_text_is_empty():
		gr.test.assert_string_contains('', 'walrus')
		assert_fail_msg_contains(gr.test, 'Expected text and search strings to be non-empty. You passed "" and "walrus".')

	func test_fails_when_search_string_is_empty():
		gr.test.assert_string_contains('This is a test.', '')
		assert_fail_msg_contains(gr.test, 'Expected text and search strings to be non-empty. You passed "This is a test." and "".')

	func test_fails_when_case_sensitive_search_not_found():
		gr.test.assert_string_contains('This is a test.', 'TeSt', true)
		assert_fail_msg_contains(gr.test, 'Expected \'This is a test.\' to contain \'TeSt\', match_case=true')

	func test_fails_when_case_insensitive_search_not_found():
		gr.test.assert_string_contains('This is a test.', 'penguin', false)
		assert_fail_msg_contains(gr.test, 'Expected \'This is a test.\' to contain \'penguin\', match_case=false')

	func test_passes_when_case_sensitive_search_is_found():
		gr.test.assert_string_contains('This is a test.', 'is a ', true)
		assert_pass(gr.test)

	func test_passes_when_case_insensitive_search_is_found():
		gr.test.assert_string_contains('This is a test.', 'this ', false)
		assert_pass(gr.test)

	func test_fails_when_text_is_null():
		gr.test.assert_string_contains(null, 'whatever', false)
		assert_fail_msg_contains(gr.test, 'Expected text and search to both be strings.  You passed <null> and "whatever".')

	func test_fails_when_search_is_null():
		gr.test.assert_string_contains('hello', null, false)
		assert_fail_msg_contains(gr.test, 'Expected text and search to both be strings.  You passed "hello" and <null>.')

	func test_fails_when_text_is_int():
		gr.test.assert_string_contains(123, '2', false)
		assert_fail_msg_contains(gr.test, 'Expected text and search to both be strings.  You passed 123 and "2".')

	func test_fails_when_search_is_int():
		gr.test.assert_string_contains('2', 123, false)
		assert_fail_msg_contains(gr.test, 'Expected text and search to both be strings.  You passed "2" and 123.')


# ------------------------------------------------------------------------------
# TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestStringStartsWith:
	extends BaseTestClass

	func test__assert_string_starts_with__fails_when_text_is_empty():
		gr.test.assert_string_starts_with('', 'The')
		assert_fail(gr.test)

	func test__assert_string_starts_with__fails_when_search_string_is_empty():
		gr.test.assert_string_starts_with('This is a test.', '')
		assert_fail(gr.test)

	func test__assert_string_starts_with__fails_when_case_sensitive_search_not_at_start():
		gr.test.assert_string_starts_with('This is a test.', 'thi', true)
		assert_fail(gr.test)

	func test__assert_string_starts_with__fails_when_case_insensitive_search_not_at_start():
		gr.test.assert_string_starts_with('This is a test.', 'puffin', false)
		assert_fail(gr.test)

	func test__assert_string_starts_with__passes_when_case_sensitive_search_at_start():
		gr.test.assert_string_starts_with('This is a test.', 'This ', true)
		assert_pass(gr.test)

	func test__assert_string_starts_with__passes_when_case_insensitive_search_at_start():
		gr.test.assert_string_starts_with('This is a test.', 'tHI', false)
		assert_pass(gr.test)


# ------------------------------------------------------------------------------
# TODO rename tests since they are now in an inner class.  See NOTE at top about naming.
class TestStringEndsWith:
	extends BaseTestClass

	func test__assert_string_ends_with__fails_when_text_is_empty():
		gr.test.assert_string_ends_with('', 'seal')
		assert_fail(gr.test)

	func test__assert_string_ends_with__fails_when_search_string_is_empty():
		gr.test.assert_string_ends_with('This is a test.', '')
		assert_fail(gr.test)

	func test__assert_string_ends_with__fails_when_case_sensitive_search_not_at_end():
		gr.test.assert_string_ends_with('This is a test.', 'TEST.', true)
		assert_fail(gr.test)

	func test__assert_string_ends_with__fails_when_case_insensitive_search_not_at_end():
		gr.test.assert_string_ends_with('This is a test.', 'orca', false)
		assert_fail(gr.test)

	func test__assert_string_ends_with__passes_when_case_sensitive_search_at_end():
		gr.test.assert_string_ends_with('This is a test.', 'est.', true)
		assert_pass(gr.test)

	func test__assert_string_ends_with__passes_when_case_insensitive_search_at_end():
		gr.test.assert_string_ends_with('This is a test.', 'A teSt.', false)
		assert_pass(gr.test)



# ------------------------------------------------------------------------------
class TestAssertNull:
	extends BaseTestClass

	func test_when_null_assert_passes():
		gr.test.assert_null(null)
		assert_pass(gr.test)

	func test_when_not_null_assert_fails():
		gr.test.assert_null('a')
		assert_fail(gr.test)

	func test_accepts_text():
		gr.test.assert_null('a', 'a is not null')
		assert_fail(gr.test)

	func test_does_not_blow_up_on_different_kinds_of_input():
		gr.test.assert_null(autofree(Node2D.new()))
		gr.test.assert_null(1)
		gr.test.assert_null([])
		gr.test.assert_null({})
		gr.test.assert_null(Color(1,1,1,1))
		assert_fail(gr.test, 5)


# ------------------------------------------------------------------------------
class TestAssertNotNull:
	extends BaseTestClass

	func test_when_null_assert_fails():
		gr.test.assert_not_null(null)
		assert_fail(gr.test)

	func test_when_not_null_assert_passes():
		gr.test.assert_not_null('a')
		assert_pass(gr.test)

	func test_accepts_text():
		gr.test.assert_not_null('a', 'a is not null')
		assert_pass(gr.test)

	func test_does_not_blow_up_on_different_kinds_of_input():
		gr.test.assert_not_null(autofree(Node2D.new()))
		gr.test.assert_not_null(1)
		gr.test.assert_not_null([])
		gr.test.assert_not_null({})
		gr.test.assert_not_null(Color(1,1,1,1))
		assert_pass(gr.test, 5)


# ------------------------------------------------------------------------------
class TestReplaceNode:
	extends BaseTestClass

	# The get methods in this scene use paths and $ to get to various resources
	# in the scene and return them.
	var Arena = load('res://test/resources/replace_node_scenes/Arena.tscn')
	var _arena = null

	func before_each():
		super.before_each()
		_arena = autofree(Arena.instantiate())

	func after_each():
		# Things get queue_free in these tests and show up as orphans when they
		# actually aren't, so wait for them to free.
		await wait_physics_frames(10)
		super.after_each()

	func test_can_replace_node():
		var replacement = autofree(Node2D.new())
		gr.test.replace_node(_arena, 'Player1/Sword', replacement)
		assert_eq(_arena.get_sword(), replacement)


	func test_when_node_does_not_exist_error_is_generated():
		var replacement = autofree(Node2D.new())
		gr.test.replace_node(_arena, 'DoesNotExist', replacement)
		assert_errored(gr.test)

	func test_replacement_works_with_dollar_sign_references():
		var replacement = autofree(Node2D.new())
		gr.test.replace_node(_arena, 'Player1', replacement)
		assert_eq(_arena.get_player1_ds(), replacement)

	func test_replacement_works_with_dollar_sign_references_2():
		var replacement = autofree(Node2D.new())
		gr.test.replace_node(_arena, 'Player1/Sword', replacement)
		assert_eq(_arena.get_sword_ds(), replacement)

	func test_replaced_node_is_freed():
		var replacement = autofree(Node2D.new())
		var old = _arena.get_sword()
		gr.test.replace_node(_arena, 'Player1/Sword', replacement)
		# object is freed using queue_free, so we have to wait for it to go away
		await wait_physics_frames(20)
		assert_true(GutUtils.is_freed(old))

	func test_replaced_node_retains_groups():
		var replacement = autofree(Node2D.new())
		var old = _arena.get_sword()
		old.add_to_group('Swords')
		gr.test.replace_node(_arena, 'Player1/Sword', replacement)
		assert_true(replacement.is_in_group('Swords'))

	func test_works_with_node_and_not_path():
		var replacement = autofree(Node2D.new())
		var old = _arena.get_sword_ds()
		gr.test.replace_node(_arena, old, replacement)
		assert_eq(_arena.get_sword(), replacement)

	func test_generates_error_if_base_node_does_not_have_node_to_replace():
		var replacement = autofree(Node2D.new())
		var old = autofree(Node2D.new())
		gr.test.replace_node(_arena, old, replacement)
		assert_errored(gr.test)


# ------------------------------------------------------------------------------
class TestAssertIsFreed:
	extends BaseTestClass

	func test_object_is_freed_should_pass():
		var obj = Node.new()
		obj.free()
		gr.test.assert_freed(obj, "Object1")
		assert_pass(gr.test)

	func test_object_is_freed_should_fail():
		var obj = Node.new()
		gr.test.assert_freed(obj, "Object2")
		assert_fail(gr.test)
		obj.free()

	func test_object_is_not_freed_should_pass():
		var obj = Node.new()
		gr.test.assert_not_freed(obj, "Object3")
		assert_pass(gr.test)
		obj.free()

	func test_object_is_not_freed_should_fail():
		var obj = Node.new()
		obj.free()
		gr.test.assert_not_freed(obj, "Object4")
		assert_fail(gr.test)

	func test_queued_free_is_not_freed():
		var obj = Node.new()
		add_child(obj)
		obj.queue_free()
		gr.test.assert_not_freed(obj, "Object4")
		assert_pass(gr.test)

	func test_assert_not_freed_title_is_optional():
		var obj = Node.new()
		gr.test.assert_not_freed(obj)
		pass_test("we got here")




# ------------------------------------------------------------------------------
class TestParameterizedTests:
	extends BaseTestClass

	func test_first_call_to_use_parameters_returns_first_index_of_params():
		var result = gr.test_with_gut.use_parameters([1, 2, 3])
		assert_eq(result, 1)

	func test_when_use_parameters_is_called_it_populates_guts_parameter_handler():
		gr.test_with_gut.use_parameters(['a'])
		assert_not_null(gr.test_with_gut.gut.parameter_handler)

	func test_prameter_handler_has_logger_set_to_guts_logger():
		gr.test_with_gut.use_parameters(['a'])
		var ph = gr.test_with_gut.gut.parameter_handler
		assert_eq(ph.get_logger(), gr.test_with_gut.gut.logger)

	func test_when_gut_already_has_parameter_handler_it_does_not_make_a_new_one():
		gr.test_with_gut.use_parameters(['a', 'b', 'c', 'd'])
		var ph = gr.test_with_gut.gut.parameter_handler
		gr.test_with_gut.use_parameters(['a', 'b', 'c', 'd'])
		assert_eq(gr.test_with_gut.gut.parameter_handler, ph)

	func test_when_parameterized_test_does_not_assert_a_warning_is_generated(p=use_parameters([1, 2, 3])):
		if(p == 1):
			gut.p("WATCH THIS, SHOULD GENERATE 2 WARNINGS")
		if(p == 2):
			pass_test('passing')


# ------------------------------------------------------------------------------
class TestMemoryMgmt:
	extends GutInternalTester

	var _gut = null

	func before_each():
		# verbose = true
		_gut = add_child_autofree(new_gut(verbose))

	func test_passes_when_no_orphans_introduced():
		var d = DynamicGutTest.new()
		d.add_source("""
		func test_assert_no_orphans():
			assert_no_new_orphans()
		""")
		var results = d.run_test_in_gut(_gut)
		assert_eq(results.passing, 1)

	func test_failing_orphan_assert_marks_test_as_failing():
		var d = DynamicGutTest.new()
		d.add_source("""
		func test_assert_no_orphans():
			var n2d = Node2D.new()
			assert_no_new_orphans('SHOULD FAIL')
			assert_true(is_failing(), 'this test should be failing')
			n2d.free()
		""")
		var results = d.run_test_in_gut(_gut)
		assert_eq(results.failing, 1)


	func test_passes_when_orphans_released():
		var n2d = Node2D.new()
		n2d.free()
		assert_no_new_orphans()
		assert_true(is_passing(), 'this should be passing')

	func test_passes_with_queue_free():
		var n2d = Node2D.new()
		n2d.queue_free()
		await wait_seconds(.5, 'must wait for queue_free to take hold')
		assert_no_new_orphans()
		assert_true(is_passing(), 'this should be passing')

	func test_autofree_children():
		var n = Node.new()
		add_child_autofree(n)
		assert_eq(n.get_parent(), self, 'added as child')
		gut.get_autofree().free_all()
		assert_freed(n, 'node')
		assert_no_new_orphans()

	func test_autoqfree_children():
		var n = Node.new()
		add_child_autoqfree(n)
		assert_eq(n.get_parent(), self, 'added as child')
		gut.get_autofree().free_all()
		assert_not_freed(n, 'node') # should not be freed until we wait
		await wait_physics_frames(10)
		assert_freed(n, 'node')
		assert_no_new_orphans()


# ------------------------------------------------------------------------------
class TestTestStateChecking:
	extends GutInternalTester

	var _gut = null

	func before_each():
		super.before_each()
		_gut = new_gut()
		_gut.logger._indent_level = 3
		add_child_autoqfree(_gut)
		_gut.add_script('res://test/resources/state_check_tests.gd')

	func _same_name():
		return gut.get_current_test_object().name

	# Well, old me was insane.  This runs a test in the _gut variable created
	# before each test.  The script is state_check_tests.gd.  The name of the
	# test is either passed in, OR it's the SAME NAME as the test that is
	# currently being run.  So, when you see five tests in a row that look
	# like they are doing the same thing and asserting different things...THIS
	# is why they are not testing the same thing.  Crazy.
	func _run_test(inner_class, name=_same_name()):
		_gut.inner_class_name = inner_class
		_gut.unit_test_name = name
		_gut.test_scripts()

	func _assert_pass_fail_count(passing, failing):
		assert_eq(_gut.get_pass_count(), passing, 'Pass count does not match')
		assert_eq(_gut.get_fail_count(), failing, 'Failing count does not match')

	func test_is_passing_returns_true_when_test_is_passing():
		_run_test('TestIsPassing')
		_assert_pass_fail_count(2, 0)

	func test_is_passing_returns_false_when_test_is_failing():
		_run_test('TestIsPassing')
		_assert_pass_fail_count(1, 1)

	func test_is_passing_false_by_default():
		_run_test('TestIsPassing')
		_assert_pass_fail_count(1, 0)

	func  test_is_passing_returns_true_before_test_fails():
		_run_test('TestIsPassing')
		_assert_pass_fail_count(2, 1)

	func test_is_failing_returns_true_when_failing():
		_run_test('TestIsFailing')
		_assert_pass_fail_count(1, 1)

	func test_is_failing_returns_false_when_passing():
		_run_test('TestIsFailing')
		_assert_pass_fail_count(2, 0)

	func test_is_failing_returns_false_by_default():
		_run_test('TestIsFailing')
		_assert_pass_fail_count(1, 0)

	func test_is_failing_returns_false_before_test_passes():
		_run_test('TestIsFailing')
		_assert_pass_fail_count(2, 0)

	func test_error_generated_when_using_is_passing_in_before_all():
		_run_test('TestUseIsPassingInBeforeAll', 'test_nothing')
		assert_errored(_gut, 1)

	func test_error_generated_when_using_is_passing_in_after_all():
		_run_test('TestUseIsPassingInAfterAll', 'test_nothing')
		assert_errored(_gut, 1)

	func test_error_generated_when_using_is_failing_in_before_all():
		_run_test('TestUseIsFailingInBeforeAll', 'test_nothing')
		assert_errored(_gut, 1)

	func test_error_generated_when_using_is_failing_in_after_all():
		_run_test('TestUseIsFailingInAfterAll', 'test_nothing')
		assert_errored(_gut, 1)


# ------------------------------------------------------------------------------
class TestPassFailTestMethods:
	extends BaseTestClass

	func test_pass_test_passes_ups_pass_count():
		gr.test_with_gut.pass_test('pass this')
		assert_eq(gr.test_with_gut.get_pass_count(), 1, 'test count')

	func test_fail_test_ups_fail_count():
		gr.test_with_gut.fail_test('fail this')
		assert_eq(gr.test_with_gut.get_fail_count(), 1, 'test count')


# ------------------------------------------------------------------------------
class TestCompareDeepShallow:
	extends BaseTestClass

	func test_compare_deep_uses_compare():
		var d_compare = double(GutUtils.Comparator).new()
		gr.test._compare = d_compare
		var result = gr.test.compare_deep([], [])
		assert_called(d_compare, 'deep')

	func test_compare_deep_sets_max_differences():
		var result = gr.test.compare_deep([], [], 10)
		assert_eq(result.max_differences, 10)

	func test_assert_eq_deep_pass_with_same():
		gr.test.assert_eq_deep({'a':1}, {'a':1})
		assert_pass(gr.test)

	func test_assert_eq_deep_fails_with_different():
		gr.test.assert_eq_deep({'a':12}, {'a':1})
		assert_fail(gr.test)

	func test_assert_ne_deep_passes_with_different():
		gr.test.assert_ne_deep({'a':12}, {'a':1})
		assert_pass(gr.test)

	func test_assert_ne_deep_fails_with_same():
		gr.test.assert_ne_deep({'a':1}, {'a':1})
		assert_fail(gr.test)

	func test_assert_shallow_fails_due_to_removed():
		gr.test.assert_eq_shallow({'a':1}, {'a':1})
		assert_fail(gr.test)

	func test_assert_ne_shallow_fails_due_to_removed():
		gr.test.assert_ne_shallow({'a':1}, {'a':2})
		assert_fail(gr.test)

	func test_compare_shallow_results_in_fail_and_warning():
		gr.test.compare_shallow([], [])
		assert_fail(gr.test)


# ------------------------------------------------------------------------------
class TestAssertProperty:
	extends BaseTestClass

	const TestNode = preload("res://test/resources/test_assert_setget_test_objects/test_node.gd")
	const TestScene = preload("res://test/resources/test_assert_setget_test_objects/TestScene.tscn")
	# the number of asserts performed by assert_property.
	# 1 for setter method
	# 1 for getter method
	# 1 for default value
	# 1 for setter
	# 1 for the total test fail count != 0
	const SUB_ASSERT_COUNT = 4

	func test_has_assert_property():
		assert_has_method(gr.test, "assert_property")

	func test_passes_if_given_input_is_valid():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property(test_node, "has_both", 4, 0)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_passes_if_instance_is_script():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property(test_node, "has_both", 4, 0)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_passes_if_instance_is_packed_scene():
		var new_node_child_mock = autofree(TestNode.new())
		add_child_autofree(new_node_child_mock)
		gr.test_with_gut.assert_property(autofree(TestScene.instantiate()), "node_with_setter_getter", null, new_node_child_mock)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_passes_if_instance_is_obj_from_script():
		var node_child_mock = autofree(TestNode.new())
		add_child_autofree(node_child_mock)
		gr.test_with_gut.assert_property(node_child_mock, "has_both", 4, 5)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_passes_if_instance_is_obj_from_packed_scene():
		var scene_mock = autofree(TestScene.instantiate())
		add_child_autoqfree(scene_mock)
		var dflt_node_with_setter = scene_mock.node_with_setter_getter
		var new_node_child_mock = autofree(TestNode.new())
		add_child_autofree(new_node_child_mock)
		gr.test_with_gut.assert_property(scene_mock, "node_with_setter_getter", dflt_node_with_setter, new_node_child_mock)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_fails_if_getter_does_not_exist():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property(test_node, 'has_setter', 2, 0)
		assert_fail_pass(gr.test_with_gut, 1, 1)

	func test_fails_if_obj_is_something_unexpected():
		var inst = "asdf"
		gr.test_with_gut.assert_property(inst, "current_dir", "", "new_dir")
		assert_fail_pass(gr.test_with_gut, 1, 0)

	func test_other_fails_do_not_cause_false_negative():
		gr.test_with_gut.fail_test('fail')
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property(test_node, "has_both", 4, 0)
		assert_fail_pass(gr.test_with_gut, 1, SUB_ASSERT_COUNT)


# ------------------------------------------------------------------------------
class TestAssertBackedProperty:
	extends BaseTestClass

	const TestNode = preload("res://test/resources/test_assert_setget_test_objects/test_node.gd")
	const TestScene = preload("res://test/resources/test_assert_setget_test_objects/TestScene.tscn")
	# the number of asserts performed by assert_property.
	# 1 for expected backing variable name
	# 1 for setter method
	# 1 for getter method
	# 1 for default value
	# 1 for setter
	# 1 for setting backed variable
	# 1 for getting backed variable
	const SUB_ASSERT_COUNT = 7

	func test_has_method():
		assert_has_method(gr.test, "assert_property_with_backing_variable")

	func test_all_pass_when_everything_is_setup_right():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property_with_backing_variable(test_node, "backed_property", 10, 0)
		assert_pass(gr.test_with_gut, SUB_ASSERT_COUNT)

	func test_fails_when_getter_does_not_return_backing_var():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property_with_backing_variable(test_node, "backed_get_broke", 11, 0)
		assert_fail_pass(gr.test_with_gut, 1, SUB_ASSERT_COUNT - 1)

	func test_fails_when_setter_does_not_set_backing_var():
		var test_node = autofree(TestNode.new())
		gr.test_with_gut.assert_property_with_backing_variable(test_node, "backed_set_broke", 12, 0)
		assert_fail_pass(gr.test_with_gut, 2, SUB_ASSERT_COUNT - 2)


# ------------------------------------------------------------------------------
class TestAssertSameAndAssertNotSame:
	extends BaseTestClass

	var _a1 = []
	var _a2 = []
	var _d1 = {}
	var _d2 = {}
	var _same_values = [
		[1, 1],
		['a', 'a'],
		[Vector3(1, 2, 3), Vector3(1, 2, 3)],
		[_a1, _a1],
		[_d1, _d1]
	]

	var _not_same_values = [
		[1, 2],
		['a', 'b'],
		[Vector3(1, 1, 1), Vector3(5, 5, 5)],
		[_a1, _a2],
		[_d1, _d2],
		[1, _a1],
		['b', _d2]
	]

	func test_assert_same_passes_when_values_are_same(p = use_parameters(_same_values)):
		gr.test_with_gut.assert_same(p[0], p[1])
		assert_pass(gr.test_with_gut)

	func test_assert_same_fails_when_values_are_not_the_same(p = use_parameters(_not_same_values)):
		gr.test_with_gut.assert_same(p[0], p[1])
		assert_fail(gr.test_with_gut)

	func test_assert_not_same_fails_when_values_are_same(p = use_parameters(_same_values)):
		gr.test_with_gut.assert_not_same(p[0], p[1])
		assert_fail(gr.test_with_gut)

	func test_assert_not_same_fails_when_values_are_not_the_same(p = use_parameters(_not_same_values)):
		gr.test_with_gut.assert_not_same(p[0], p[1])
		assert_pass(gr.test_with_gut)
