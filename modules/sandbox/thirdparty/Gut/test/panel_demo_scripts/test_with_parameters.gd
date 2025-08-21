extends GutTest

func test_all_passing(p = use_parameters([1, 2, 3, 4, 5, 6, 7, 8])):
	assert_true(true, str('this always passes ', p))

func test_all_failing(p = use_parameters([1, 2, 3, 4, 5, 6, 7, 8])):
	assert_true(false, str('this always passes ', p))

func test_some_failing(p = use_parameters([1, 2, 3, 4, 5, 6, 7, 8])):
	assert_true(p%2 == 0, str('is even ', p))

func test_has_a_pending_in_middle(p = use_parameters([1, 2, 3, 4, 5, 6, 7, 8])):
	if(p == 5):
		pending("five is pending")
	else:
		assert_true(p != 3, str('this always passes ', p))
