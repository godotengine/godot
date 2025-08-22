extends GutInternalTester

var Summary = load('res://addons/gut/test_collector.gd')
const PARSING_AND_LOADING = 'res://test/resources/parsing_and_loading_samples'
const SUMMARY_SCRIPTS = 'res://test/resources/summary_test_scripts'


func test_can_make_one():
	var s = Summary.new()
	assert_not_null(s)

func test_can_make_one_with_a_test_colletor():
	var s = Summary.new(autofree(new_gut()))
	assert_not_null(s)


