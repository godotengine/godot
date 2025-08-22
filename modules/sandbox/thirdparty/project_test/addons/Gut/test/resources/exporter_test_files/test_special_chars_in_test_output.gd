extends GutTest


func test_pending_with_null_message():
	pending("The < and > in this message mess up the XML")

func test_failure_message_includes_str_of_null_in_message_1():
	assert_not_null(null)

func test_failure_message_includes_str_of_null_in_message_2():
	assert_eq('null', str(null))

func test_failure_message_includes_str_of_null_in_message_3():
	assert_eq('a', 'b', "The < and > in this message mess up the XML")

class TestSkipWithNullMessage:
	extends GutTest

	func should_skip():
		return null

	func test_something_that_is_skipped():
		assert_true(false, 'hello world')


