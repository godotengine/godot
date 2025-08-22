class_name DynamicGutTest
# ------------------------------------------------------------------------------
# Used to create dynamic GutTest scripts for integration tests that require
# running tests through GUT.  This makes it easier to keep the source of the
# test scripts created in tests to test the tests closer to the tests that
# test the test.
# ------------------------------------------------------------------------------
static var should_print_source = true
var source_entries = []
var lambdas = []


func make_source():
	var src = "extends GutTest\n"
	for e in source_entries:
		src += str(e, "\n")

	return src


func make_script():
	return GutUtils.create_script_from_source(make_source())


func make_new():
	var to_return = make_script().new()
	if(should_print_source):
		print(to_return.get_script().resource_path)
		print_source()

	return to_return


func add_source(p1='', p2='', p3='', p4='', p5='', p6='', p7='', p8='', p9='', p10=''):
	var source = str(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
	source_entries.append(source.dedent())
	return self


func add_lambda_test(lambda, test_name=null):
	var idx = lambdas.size()
	var func_name = test_name
	if(func_name == null):
		func_name = str("test_run_lambda_", idx)
	lambdas.append(lambda)
	add_source("func ", func_name, "():\n",
		"\tinstance_from_id(", get_instance_id(), ").lambdas[", idx, "].call(self)")
	return self


func add_as_test_to_gut(which):
	var dyn = make_script()
	if(should_print_source):
		print(dyn.resource_path)
		print_source()

	which.get_test_collector().add_script(dyn.resource_path)


func run_test_in_gut(which):
	add_as_test_to_gut(which)
	which.run_tests()
	var s = GutUtils.Summary.new()
	return s.get_totals(which)


func print_source():
	print(GutUtils.add_line_numbers(make_source()))

