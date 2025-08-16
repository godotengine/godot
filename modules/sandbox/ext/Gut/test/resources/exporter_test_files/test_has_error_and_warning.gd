extends GutTest

func test_manual_error():
	gut.logger.error("This is a manual error")
	pass_test('we did it')

func test_manual_warning():
	gut.logger.warn("This is a manual warning")
	pass_test('we did it')
