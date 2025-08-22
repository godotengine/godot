# ------------------------------------------------------------------------------
# Prints things, mostly.  Knows too much about gut.gd, but it's only supposed to
# work with gut.gd, so I'm fine with that.
# ------------------------------------------------------------------------------
# a _test_collector to use when one is not provided.
var _gut = null


func _init(gut=null):
	_gut = gut

# ---------------------
# Private
# ---------------------
func _log_end_run_header(gut):
	var lgr = gut.get_logger()
	lgr.log("\n\n\n")
	lgr.log('==============================================', lgr.fmts.yellow)
	lgr.log("= Run Summary", lgr.fmts.yellow)
	lgr.log('==============================================', lgr.fmts.yellow)


func _log_what_was_run(gut):
	if(!GutUtils.is_null_or_empty(gut._select_script)):
		gut.p('Ran Scripts matching "' + gut._select_script + '"')
	if(!GutUtils.is_null_or_empty(gut._unit_test_name)):
		gut.p('Ran Tests matching "' + gut._unit_test_name + '"')
	if(!GutUtils.is_null_or_empty(gut._inner_class_name)):
		gut.p('Ran Inner Classes matching "' + gut._inner_class_name + '"')


func _log_orphans_and_disclaimer(gut):
	var lgr = gut.get_logger()
	if(!lgr.is_type_enabled('orphan')):
		return

	var counter = gut.get_orphan_counter()
	# Do not count any of the test scripts since these will be released when GUT
	# is released.
	var do_not_count_orphans = counter.get_count("pre_run") + gut.get_test_script_count()
	var total_run_orphans = counter.orphan_count() - do_not_count_orphans

	if(total_run_orphans > 0):
		lgr.orphan(str("Total orphans in run ", total_run_orphans))
		gut.p("Note:  This count does not include GUT objects that will be freed upon exit.")
		gut.p("       It also does not include any orphans created by global scripts")
		gut.p("       loaded before tests were ran.")
		gut.p(str("Total orphans = ", counter.orphan_count()))
		gut.p('')


func _total_fmt(text, value):
	var space = 18
	if(str(value) == '0'):
		value = 'none'
	return str(text.rpad(space), value)


func _log_non_zero_total(text, value, lgr):
	if(str(value) != '0'):
		lgr.log(_total_fmt(text, value))
		return 1
	else:
		return 0


func _log_totals(gut, totals):
	var lgr = gut.get_logger()
	lgr.log()

	lgr.log("---- Totals ----")
	var issue_count = 0
	issue_count += _log_non_zero_total('Errors', totals.errors, lgr)
	issue_count += _log_non_zero_total('Warnings', totals.warnings, lgr)
	issue_count += _log_non_zero_total('Deprecated', totals.deprecated, lgr)
	if(issue_count > 0):
		lgr.log("")

	lgr.log(_total_fmt( 'Scripts', totals.scripts))
	lgr.log(_total_fmt( 'Tests', gut.get_test_collector().get_ran_test_count()))
	lgr.log(_total_fmt( '  Passing', totals.passing_tests))
	_log_non_zero_total('  Failing', totals.failing_tests, lgr)
	_log_non_zero_total('  Risky/Pending', totals.risky + totals.pending, lgr)
	lgr.log(_total_fmt( 'Asserts', totals.passing + totals.failing))
	lgr.log(_total_fmt( 'Time', str(gut.get_elapsed_time(), 's')))

	return totals


func _log_nothing_run(gut):
	var lgr = gut.get_logger()
	lgr.error("Nothing was run.")
	lgr.log('On the one hand nothing failed, on the other hand nothing did anything.')


# ---------------------
# Public
# ---------------------
func log_all_non_passing_tests(gut=_gut):
	var test_collector = gut.get_test_collector()
	var lgr = gut.get_logger()

	var to_return = {
		passing = 0,
		non_passing = 0
	}

	for test_script in test_collector.scripts:
		lgr.set_indent_level(0)

		if(test_script.was_skipped or test_script.get_fail_count() > 0 or test_script.get_pending_count() > 0):
			lgr.log("\n" + test_script.get_full_name(), lgr.fmts.underline)

		if(test_script.was_skipped):
			lgr.inc_indent()
			var skip_msg = str('[Risky] Script was skipped:  ', test_script.skip_reason)
			lgr.log(skip_msg, lgr.fmts.yellow)
			lgr.dec_indent()

		for test in test_script.tests:
			if(test.was_run):
				if(test.is_passing()):
					to_return.passing += 1
				else:
					to_return.non_passing += 1
					lgr.log(str('- ', test.name))
					lgr.inc_indent()

					for i in range(test.fail_texts.size()):
						lgr.failed(test.fail_texts[i])
					for i in range(test.pending_texts.size()):
						lgr.pending(test.pending_texts[i])
					if(test.is_risky()):
						lgr.risky('Did not assert')
					lgr.dec_indent()

	return to_return


func log_the_final_line(totals, gut):
	var lgr = gut.get_logger()
	var grand_total_text = ""
	var grand_total_fmt = lgr.fmts.none
	if(totals.failing_tests > 0):
		grand_total_text = str(totals.failing_tests, " failing tests")
		grand_total_fmt = lgr.fmts.red
	elif(totals.risky > 0 or totals.pending > 0):
		grand_total_text = str(totals.risky + totals.pending, " pending/risky tests.")
		grand_total_fmt = lgr.fmts.yellow
	else:
		grand_total_text = "All tests passed!"
		grand_total_fmt = lgr.fmts.green

	lgr.log(str("---- ", grand_total_text, " ----"), grand_total_fmt)


func log_totals(gut, totals):
	var lgr = gut.get_logger()
	var orig_indent = lgr.get_indent_level()
	lgr.set_indent_level(0)
	_log_totals(gut, totals)
	lgr.set_indent_level(orig_indent)


func get_totals(gut=_gut):
	var tc = gut.get_test_collector()
	var lgr = gut.get_logger()

	var totals = {
		failing = 0,
		failing_tests = 0,
		passing = 0,
		passing_tests = 0,
		pending = 0,
		risky = 0,
		scripts = tc.get_ran_script_count(),
		tests = 0,

		deprecated = lgr.get_deprecated().size(),
		errors = lgr.get_errors().size(),
		warnings = lgr.get_warnings().size(),
	}

	for s in tc.scripts:
		# assert totals
		totals.passing += s.get_pass_count()
		totals.pending += s.get_pending_count()
		totals.failing += s.get_fail_count()

		# test totals
		totals.tests += s.get_ran_test_count()
		totals.passing_tests += s.get_passing_test_count()
		totals.failing_tests += s.get_failing_test_count()
		totals.risky += s.get_risky_count()

	return totals


func log_end_run(gut=_gut):
	var totals = get_totals(gut)
	if(totals.tests == 0):
		_log_nothing_run(gut)
		return

	_log_end_run_header(gut)
	var lgr = gut.get_logger()

	log_all_non_passing_tests(gut)
	log_totals(gut, totals)
	lgr.log("\n")

	_log_orphans_and_disclaimer(gut)
	_log_what_was_run(gut)
	log_the_final_line(totals, gut)
	lgr.log("")
