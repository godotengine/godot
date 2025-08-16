# ------------------------------------------------------------------------------
# Used to keep track of info about each test ran.
# ------------------------------------------------------------------------------
# the name of the function
var name = ""

# flag to know if the name has been printed yet.  Used by the logger.
var has_printed_name = false

# the number of arguments the method has
var arg_count = 0

# the time it took to execute the test in seconds
var time_taken : float = 0

# The number of asserts in the test.  Converted to a property for backwards
# compatibility.  This now reflects the text sizes instead of being a value
# that can be altered externally.
var assert_count = 0 :
	get: return pass_texts.size() + fail_texts.size()
	set(val): pass

# Converted to propety for backwards compatibility.  This now cannot be set
# externally
var pending = false :
	get: return is_pending()
	set(val): pass

# the line number when the test fails
var line_number = -1

# Set internally by Gut using whatever reason Gut wants to use to set this.
# Gut will skip these marked true and the test will be listed as risky.
var should_skip = false  # -- Currently not used by GUT don't believe ^

var pass_texts = []
var fail_texts = []
var pending_texts = []
var orphans = 0

var was_run = false


func did_pass():
	return is_passing()


func add_fail(fail_text):
	fail_texts.append(fail_text)


func add_pending(pending_text):
	pending_texts.append(pending_text)


func add_pass(passing_text):
	pass_texts.append(passing_text)


# must have passed an assert and not have any other status to be passing
func is_passing():
	return pass_texts.size() > 0 and fail_texts.size() == 0 and pending_texts.size() == 0


# failing takes precedence over everything else, so any failures makes the
# test a failure.
func is_failing():
	return fail_texts.size() > 0


# test is only pending if pending was called and the test is not failing.
func is_pending():
	return pending_texts.size() > 0 and fail_texts.size() == 0


func is_risky():
	return should_skip or (was_run and !did_something())


func did_something():
	return is_passing() or is_failing() or is_pending()


func get_status_text():
	var to_return = GutUtils.TEST_STATUSES.NO_ASSERTS

	if(should_skip):
		to_return = GutUtils.TEST_STATUSES.SKIPPED
	elif(!was_run):
		to_return = GutUtils.TEST_STATUSES.NOT_RUN
	elif(pending_texts.size() > 0):
		to_return = GutUtils.TEST_STATUSES.PENDING
	elif(fail_texts.size() > 0):
		to_return = GutUtils.TEST_STATUSES.FAILED
	elif(pass_texts.size() > 0):
		to_return = GutUtils.TEST_STATUSES.PASSED

	return to_return


# Deprecated
func get_status():
	return get_status_text()


func to_s():
	var pad = '     '
	var to_return = str(name, "[", get_status_text(), "]\n")

	for i in range(fail_texts.size()):
		to_return += str(pad, 'Fail:  ', fail_texts[i])
	for i in range(pending_texts.size()):
		to_return += str(pad, 'Pending:  ', pending_texts[i], "\n")
	for i in range(pass_texts.size()):
		to_return += str(pad, 'Pass:  ', pass_texts[i], "\n")
	return to_return


