extends GutTest

var Gut = load('res://addons/gut/gut.gd')
var JunitExporter = GutUtils.JunitXmlExport
var GutLogger = GutUtils.GutLogger

var _test_gut = null

const RESULT_XML_VALID_TAGS := {
	"testsuite": [ "name", "tests", "failures", "skipped", "time" ],
	"testcase": [ "name", "assertions", "status", "classname", "time" ],
	"testsuites": [ "name", "failures", "tests" ],
	"failure": [ "message" ],
	"skipped": [ "message" ]
}

# Returns a new gut object, all setup for testing.
func get_a_gut():
	var g = Gut.new()
	g.log_level = g.LOG_LEVEL_ALL_ASSERTS
	g.logger = GutUtils.GutLogger.new()
	g.logger.disable_printer('terminal', true)
	g.logger.disable_printer('gui', true)
	g.logger.disable_printer('console', true)
	return g


func run_scripts(g, one_or_more):
	var scripts = one_or_more
	if(typeof(scripts) != TYPE_ARRAY):
		scripts = [scripts]
	for s in scripts:
		g.add_script(export_script(s))
	g.test_scripts()


# Very simple xml validator.  Matches closing tags to opening tags as they
# are encountered and any validation provided by XMLParser (which is very
# little).  Does not catch malformed attributes among other things probably.
# Additionally checks for valid tags as defined by RESULT_XML_VALID_TAGS
func assert_is_valid_xml(xml : String)->void:
	var tags = []
	var pba = xml.to_utf8_buffer()
	var parser = XMLParser.new()
	var result = parser.open_buffer(pba)

	while(result == OK):
		if(parser.get_node_type() == parser.NODE_ELEMENT):
			var tag_name := parser.get_node_name()
			tags.push_back(tag_name)

			if (tag_name in RESULT_XML_VALID_TAGS):
				# check for required attributes
				var required_attributes : Array = RESULT_XML_VALID_TAGS[tag_name].duplicate()
				var missing_attributes := required_attributes.filter(func(attribute): return !parser.has_attribute(attribute))
				assert_eq(missing_attributes, [], str(tag_name, ":  Required attribute(s) missing ", missing_attributes))

				# check for unexpected attributes
				var unexpected_attributes : Array[String] = []
				for attribute_index : int in parser.get_attribute_count():
					var attribute_name := parser.get_attribute_name(attribute_index)
					if not attribute_name in RESULT_XML_VALID_TAGS[tag_name]:
						unexpected_attributes.push_back(attribute_name)
				assert_eq(unexpected_attributes, [], str(tag_name, " Unexpected attribute(s) ", unexpected_attributes))
			else:
				fail_test("%s is not one of the expected tags: %s" % [tag_name, RESULT_XML_VALID_TAGS.keys()])

		elif(parser.get_node_type() == parser.NODE_ELEMENT_END):
			var last_tag = tags.pop_back()
			if(last_tag != parser.get_node_name()):
				var msg = str("End tag does not match.  Expected:  ", last_tag, ', got:  ', parser.get_node_name())
				push_error(msg)
				result = -1

		if(result != -1):
			result = parser.read()

	assert_eq(result, ERR_FILE_EOF, 'Parsing xml should reach EOF')


func export_script(name):
	return str('res://test/resources/exporter_test_files/', name)

func before_all():
	GutUtils._test_mode = true

func before_each():
	_test_gut = get_a_gut()
	add_child_autoqfree(_test_gut)


func test_can_make_one():
	assert_not_null(JunitExporter.new())

func test_no_tests_returns_valid_xml():
	_test_gut.test_scripts()
	var re = JunitExporter.new()
	var result = re.get_results_xml(_test_gut)
	assert_is_valid_xml(result)

func test_spot_check():
	run_scripts(_test_gut, ['test_simple_2.gd', 'test_simple.gd', 'test_with_inner_classes.gd', 'test_special_chars_in_test_output.gd'])
	var re = JunitExporter.new()
	var result = re.get_results_xml(_test_gut)
	assert_is_valid_xml(result)

func test_res_removed_from_classname_path():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = JunitExporter.new()
	var result = re.get_results_xml(_test_gut)
	assert_false(result.contains("classname=\"res://test/resources/exporter_test_files/test_simple_2.gd\""))
	assert_string_contains(result, "classname=\"test/resources/exporter_test_files/test_simple_2.gd\"")

func test_write_file_creates_file():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var fname = "user://test_junit_exporter.xml"
	var re = JunitExporter.new()
	var result = re.write_file(_test_gut, fname)
	assert_file_not_empty(fname)
	gut.file_delete(fname)

func test_xml_is_valid_when_test_skip_message_is_null():
	run_scripts(_test_gut, ['test_special_chars_in_test_output.gd'])
	var re = JunitExporter.new()
	var result = re.get_results_xml(_test_gut)
	assert_is_valid_xml(result)
