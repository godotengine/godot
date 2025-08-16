extends GutTest


func test_this_passes():
    pass_test('passing')

func test_this_also_passes():
    assert_eq(1, 1, 'one')

func test_this_is_pending():
    pending('')

func test_this_does_nothing():
    pass

func test_another_pass():
    assert_true(true, 'true')