extends GutTest

func test_can_make_one():
	var c  = CompareResult.new()
	assert_not_null(c)

func test_get_set_equal():
	var c = CompareResult.new()
	assert_accessors(c, 'are_equal', false, true)

func test_get_set_summary():
	var  c = CompareResult.new()
	assert_accessors(c, 'summary', null, 'asdf')

func test_get_short_summary_returns_summary():
	var c = CompareResult.new()
	c.set_summary('adsf')
	assert_eq(c.get_short_summary(), 'adsf')

func test_get_set_max_differences():
	var c = CompareResult.new()
	assert_accessors(c, 'max_differences', 30, 40)
