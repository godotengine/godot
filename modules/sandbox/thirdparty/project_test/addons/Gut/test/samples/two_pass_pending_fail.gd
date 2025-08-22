#A sample script for illustrating multiple scripts and what it looks like
#when all tests pass.
extends "res://addons/gut/test.gd"

func test_pass_one():
	assert_true(true, 'This is true')

func test_pass_two():
	assert_eq(1, 1, 'one is one in test two')

func test_fail_one():
	assert_true(false, 'false is not true')

func test_fail_two():
	assert_eq(3, 1, 'three is not one in test two')

func test_pending_one():
	pending('This one is pending')

func test_pending_two():
	pending('This is also pending')
