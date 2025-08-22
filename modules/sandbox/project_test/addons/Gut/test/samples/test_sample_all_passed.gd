#A sample script for illustrating multiple scripts and what it looks like
#when all tests pass.
extends GutTest

func test_works():
	assert_true(true, 'This is true')

func test_two():
	assert_eq("two", "two", "This is also true")

func test_3():
	assert_ne("one", "two", "This is yet again true")

