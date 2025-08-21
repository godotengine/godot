extends GutTest

var CollectedTest = GutUtils.CollectedTest
var CollectedScript = GutUtils.CollectedScript



func test_get_risky_count_counts_all_tests_that_were_run():
	var c_script = CollectedScript.new()

	for i in range(3):
		var c_test = CollectedTest.new()
		c_script.tests.append(c_test)
		c_test.was_run = true

	assert_eq(c_script.get_risky_count(), 3)


func test_get_risky_count_does_not_count_scripts_not_run():
	var c_script = CollectedScript.new()

	for i in range(3):
		var c_test = CollectedTest.new()
		c_script.tests.append(c_test)
		c_test.was_run = false

	assert_eq(c_script.get_risky_count(), 0)


func test_get_ran_test_count_only_returns_tests_that_were_run():
	var c_script = CollectedScript.new()

	var c_test = CollectedTest.new()
	c_script.tests.append(c_test)
	c_test.was_run = true

	c_test = CollectedTest.new()
	c_script.tests.append(c_test)
	c_test.was_run = true

	c_test = CollectedTest.new()
	c_script.tests.append(c_test)
	c_test.was_run = false

	assert_eq(c_script.get_ran_test_count(), 2)