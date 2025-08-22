extends "res://addons/gut/test.gd"

func test_has_one_defaulted_parameter(p=use_parameters(['a'])):
	assert_true(true, 'this one passes')

func test_has_two_parameters(p1=null, p2=null):
	assert_true(false, 'Should not see this.  This should not be run.')

func test_no_parameters():
	assert_true(true, 'this one passes')

func test_has_three_values_for_parameters(p=use_parameters([['a', 'a'], ['b', 'b'], ['c', 'c']])):
	assert_eq(p[0], p[1])

func test_does_not_use_use_parameters(p=null):
	assert_true(true, 'this passes but should never be called more than once.')

func test_three_values_and_a_yield(p=use_parameters([['a', 'a'], ['b', 'b'], ['c', 'c']])):
	await wait_seconds(.2)
	assert_eq(p[0], p[1])


class TestInnerClass:
	extends "res://addons/gut/test.gd"
	func test_inner_has_one_defaulted_parameter(p=null):
		assert_true(true, 'this one passes')

	func test_inner_has_two_parameters(p1=null, p2=null):
		assert_true(false, 'Should not see this.  This should not be run.')

	func test_inner_no_parameters():
		assert_true(true, 'this one passes')


class TestWithBeforeEach:
	extends "res://addons/gut/test.gd"

	var before_count = 0
	var func_params = [1, 2, 3]

	func before_each():
		before_count += 1

	func test_run(p = use_parameters(func_params)):
		assert_eq(before_count, p)

class TestWithAfterEach:
	extends "res://addons/gut/test.gd"

	var after_count = 0
	var func_params = [0, 1, 2]

	func after_each():
		after_count += 1

	func test_run(p = use_parameters(func_params)):
		assert_eq(after_count, p)
