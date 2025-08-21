extends GutTest

func test_this_causes_an_error_and_passes():
    _lgr.error('There was an error here')
    pass_test('passes')

func test_this_causes_an_error_and_fails():
    _lgr.error('another error here')
    fail_test('this fails')

func test_this_causes_an_orphan_and_passes():
    var n = Node.new()
    pass_test('passes')

func test_this_causes_a_warning_and_passes():
    _lgr.warn('A new warning')
    pass_test('passing')

func test_this_causes_a_warning_and_fails():
    _lgr.warn('Another warning')
    fail_test('failing')

func test_this_causes_a_deprecation_and_passes():
    _lgr.deprecated('this thing should not be used anymore')
    pass_test('passing')

func test_this_causes_a_deprecation_and_fails():
    _lgr.deprecated('this thing should not be used anymore')
    fail_test('passing')