class_name GutTest
extends Node
## This is the base class for your GUT test scripts.[br]
## [br]
## GUT Wiki:  [url=https://gut.readthedocs.io]https://gut.readthedocs.io[/url]
## [br]
## Simple Example
## [codeblock]
##    extends GutTest
##
##    func before_all():
##        gut.p("before_all called"
##
##    func before_each():
##        gut.p("before_each called")
##
##    func after_each():
##        gut.p("after_each called")
##
##    func after_all():
##        gut.p("after_all called")
##
##    func test_assert_eq_letters():
##        assert_eq("asdf", "asdf", "Should pass")
##
##    func test_assert_eq_number_not_equal():
##        assert_eq(1, 2, "Should fail.  1 != 2")
## [/codeblock]


# Normalizes p1 and p2 into object/signal_name/signal_ref(sig).  Additional
# parameters are optional and will be placed into the others array.  This
# class is used in refactoring signal methods to accept a reference to the
# signal instead an object and the signal name.
class SignalAssertParameters:
	var object = null
	var signal_name = null
	var sig = null
	var others := []

	func _init(p1, p2, p3=null, p4=null, p5=null, p6=null):
		others = [p3, p4, p5, p6]
		if(p1 is Signal):
			object = p1.get_object()
			signal_name = p1.get_name()
			others.push_front(p2)
			sig = p1
		else:
			object = p1
			signal_name = p2
			sig = object.get(signal_name)


const EDITOR_PROPERTY = PROPERTY_USAGE_SCRIPT_VARIABLE | PROPERTY_USAGE_DEFAULT
const VARIABLE_PROPERTY = PROPERTY_USAGE_SCRIPT_VARIABLE
# Convenience copy of GutUtils.DOUBLE_STRATEGY
var DOUBLE_STRATEGY = GutUtils.DOUBLE_STRATEGY

## Reference to [addons/gut/parameter_factory.gd] script.
var ParameterFactory = GutUtils.ParameterFactory
## @ignore
var CompareResult = GutUtils.CompareResult
## Reference to [GutInputFactory] class that was originally used to reference
## the Input Factory before the class_name was introduced.
var InputFactory = GutInputFactory
## Reference to [GutInputSender].  This was the way you got to the [GutInputSender]
## before it was given a [code]class_name[/code]
var InputSender = GutUtils.InputSender

# Need a reference to the instance that is running the tests.  This
# is set by the gut class when it runs the test script.
var gut: GutMain = null


var _compare = GutUtils.Comparator.new()
var _disable_strict_datatype_checks = false
# Holds all the text for a test's fail/pass.  This is used for testing purposes
# to see the text of a failed sub-test in test_test.gd
var _fail_pass_text = []
# Summary counts for the test.
var _summary = {
	asserts = 0,
	passed = 0,
	failed = 0,
	tests = 0,
	pending = 0
}
# This is used to watch signals so we can make assertions about them.
var _signal_watcher = load('res://addons/gut/signal_watcher.gd').new()
var _lgr = GutUtils.get_logger()
var _strutils = GutUtils.Strutils.new()
var _awaiter = null
var _was_ready_called = false

# I haven't decided if we should be using _ready or not.  Right now gut.gd will
# call this if _ready was not called (because it was overridden without a super
# call).  Maybe gut.gd should just call _do_ready_stuff (after we rename it to
# something better).  I'm leaving all this as it is until it bothers me more.
func _do_ready_stuff():
	_awaiter = GutUtils.Awaiter.new()
	add_child(_awaiter)
	_was_ready_called = true


func _ready():
	_do_ready_stuff()


func _notification(what):
	# Tests are never expected to re-enter the tree.  Tests are removed from the
	# tree after they are run.
	if(what == NOTIFICATION_EXIT_TREE):
		_awaiter.queue_free()


#region Private
# ----------------


func _str(thing):
	return _strutils.type2str(thing)


func _str_precision(value, precision):
	var to_return = _str(value)
	var format = str('%.', precision, 'f')
	if(typeof(value) == TYPE_FLOAT):
		to_return = format % value
	elif(typeof(value) == TYPE_VECTOR2):
		to_return = str('VECTOR2(', format % value.x, ', ', format %value.y, ')')
	elif(typeof(value) == TYPE_VECTOR3):
		to_return = str('VECTOR3(', format % value.x, ', ', format %value.y, ', ', format % value.z, ')')

	return to_return


# Fail an assertion.  Causes test and script to fail as well.
func _fail(text):
	_summary.asserts += 1
	_summary.failed += 1
	_fail_pass_text.append('failed:  ' + text)
	if(gut):
		_lgr.failed(gut.get_call_count_text() + text)
		gut._fail(text)


# Pass an assertion.
func _pass(text):
	_summary.asserts += 1
	_summary.passed += 1
	_fail_pass_text.append('passed:  ' + text)
	if(gut):
		_lgr.passed(text)
		gut._pass(text)


# Checks if the datatypes passed in match.  If they do not then this will cause
# a fail to occur.  If they match then TRUE is returned, FALSE if not.  This is
# used in all the assertions that compare values.
func _do_datatypes_match__fail_if_not(got, expected, text):
	var did_pass = true

	if(!_disable_strict_datatype_checks):
		var got_type = typeof(got)
		var expect_type = typeof(expected)
		if(got_type != expect_type and got != null and expected != null):
			# If we have a mismatch between float and int (types 2 and 3) then
			# print out a warning but do not fail.
			if([2, 3].has(got_type) and [2, 3].has(expect_type)):
				_lgr.warn(str('Warn:  Float/Int comparison.  Got ', _strutils.types[got_type],
					' but expected ', _strutils.types[expect_type]))
			elif([TYPE_STRING, TYPE_STRING_NAME].has(got_type) and [TYPE_STRING, TYPE_STRING_NAME].has(expect_type)):
				pass
			else:
				_fail('Cannot compare ' + _strutils.types[got_type] + '[' + _str(got) + '] to ' + \
					_strutils.types[expect_type] + '[' + _str(expected) + '].  ' + text)
				did_pass = false

	return did_pass


# Create a string that lists all the methods that were called on an spied
# instance.
func _get_desc_of_calls_to_instance(inst):
	var BULLET = '  * '
	var calls = gut.get_spy().get_call_list_as_string(inst)
	# indent all the calls
	calls = BULLET + calls.replace("\n", "\n" + BULLET)
	# remove_at trailing newline and bullet
	calls = calls.substr(0, calls.length() - BULLET.length() - 1)
	return "Calls made on " + str(inst) + "\n" + calls



# Signal assertion helper.  Do not call directly, use _can_make_signal_assertions
func _fail_if_does_not_have_signal(object, signal_name):
	var did_fail = false
	if(!_signal_watcher.does_object_have_signal(object, signal_name)):
		_fail(str('Object ', object, ' does not have the signal [', signal_name, ']'))
		did_fail = true
	return did_fail


# Signal assertion helper.  Do not call directly, use _can_make_signal_assertions
func _fail_if_not_watching(object):
	var did_fail = false
	if(!_signal_watcher.is_watching_object(object)):
		_fail(str('Cannot make signal assertions because the object ', object, \
				' is not being watched.  Call watch_signals(some_object) to be able to make assertions about signals.'))
		did_fail = true
	return did_fail


# Returns text that contains original text and a list of all the signals that
# were emitted for the passed in object.
func _get_fail_msg_including_emitted_signals(text, object):
	return str(text," (Signals emitted: ", _signal_watcher.get_signals_emitted(object), ")")


# This validates that parameters is an array and generates a specific error
# and a failure with a specific message
func _fail_if_parameters_not_array(parameters):
	var invalid = parameters != null and typeof(parameters) != TYPE_ARRAY
	if(invalid):
		_lgr.error('The "parameters" parameter must be an array of expected parameter values.')
		_fail('Cannot compare parameter values because an array was not passed.')
	return invalid


# A bunch of common checkes used when validating a double/method pair.  If
# everything is ok then an empty string is returned, otherwise the message
# is returned.
func _get_bad_double_or_method_message(inst, method_name, what_you_cant_do):
	var to_return = ''

	if(!GutUtils.is_double(inst)):
		to_return = str("An instance of a Double was expected, you passed:  ", _str(inst))
	elif(!inst.has_method(method_name)):
		to_return = str("You cannot ", what_you_cant_do, " [", method_name, "] because the method does not exist.  ",
			"This can happen if the method is virtual and not overloaded (i.e. _ready) ",
			"or you have mistyped the name of the method.")
	elif(!inst.__gutdbl_values.doubled_methods.has(method_name)):
		to_return = str("You cannot ", what_you_cant_do, " [", method_name, "] because ",
			_str(inst), ' does not overload it or it was ignored with ',
			'ignore_method_when_doubling.  See Doubling ',
			'Strategy in the wiki for details on including non-overloaded ',
			'methods in a double.')

	return to_return


func _fail_if_not_double_or_does_not_have_method(inst, method_name):
	var to_return = OK

	var msg = _get_bad_double_or_method_message(inst, method_name, 'spy on')
	if(msg != ''):
		_fail(msg)
		to_return = ERR_INVALID_DATA

	return to_return


func _create_obj_from_type(type):
	var obj = null
	if type.is_class("PackedScene"):
		obj = type.instantiate()
		add_child(obj)
	else:
		obj = type.new()
	return obj


# Converts a Callabe passed through inst or inst/method_name/parameters into a
# hash so that methods that interact with Spy can accept both more easily.
func _convert_spy_args(inst, method_name, parameters):
	var to_return = {
		'object':inst,
		'method_name':method_name,
		'arguments':parameters,
		'invalid_message':'ok'
	}

	if(inst is Callable):
		if(parameters != null):
			to_return.invalid_message =\
				"3rd parameter to assert_called not supported when using a Callable."
		elif(method_name != null):
			to_return.invalid_message =\
				"2nd parameter to assert_called not supported when using a Callable."
		else:
			if(inst.get_bound_arguments_count() > 0):
				to_return.arguments = inst.get_bound_arguments()
			to_return.method_name = inst.get_method()
			to_return.object = inst.get_object()

	return to_return


func _get_typeof_string(the_type):
	var to_return = ""
	if(_strutils.types.has(the_type)):
		to_return += str(the_type, '(',  _strutils.types[the_type], ')')
	else:
		to_return += str(the_type)
	return to_return


# Validates the singleton_name is a string and exists.  Errors when conditions
# are not met.  Returns true/false if singleton_name is valid or not.
func _validate_singleton_name(singleton_name):
	var is_valid = true
	if(typeof(singleton_name) != TYPE_STRING):
		_lgr.error("double_singleton requires a Godot singleton name, you passed " + _str(singleton_name))
		is_valid = false
	# Sometimes they have underscores in front of them, sometimes they do not.
	# The doubler is smart enought of ind the right thing, so this has to be
	# that smart as well.
	elif(!ClassDB.class_exists(singleton_name) and !ClassDB.class_exists('_' + singleton_name)):
		var txt = str("The singleton [", singleton_name, "] could not be found.  ",
					"Check the GlobalScope page for a list of singletons.")
		_lgr.error(txt)
		is_valid = false
	return is_valid


# Checks the object for 'get_' and 'set_' methods for the specified property.
# If found a warning is generated.
func _warn_for_public_accessors(obj, property_name):
	var public_accessors = []
	var accessor_names = [
		str('get_', property_name),
		str('is_', property_name),
		str('set_', property_name)
	]

	for acc in accessor_names:
		if(obj.has_method(acc)):
			public_accessors.append(acc)

	if(public_accessors.size() > 0):
		_lgr.warn (str('Public accessors ', public_accessors, ' found for property ', property_name))


func _smart_double(thing, double_strat, partial):
	var override_strat = GutUtils.nvl(double_strat, gut.get_doubler().get_strategy())
	var to_return = null

	if(thing is PackedScene):
		if(partial):
			to_return =  gut.get_doubler().partial_double_scene(thing, override_strat)
		else:
			to_return =  gut.get_doubler().double_scene(thing, override_strat)

	elif(GutUtils.is_native_class(thing)):
		if(partial):
			to_return = gut.get_doubler().partial_double_gdnative(thing)
		else:
			to_return = gut.get_doubler().double_gdnative(thing)

	elif(thing is GDScript):
		if(partial):
			to_return = gut.get_doubler().partial_double(thing, override_strat)
		else:
			to_return = gut.get_doubler().double(thing, override_strat)

	return to_return


# This is here to aid in the transition to the new doubling sytnax.  Once this
# has been established it could be removed.  We must keep the is_instance check
# going forward though.
func _are_double_parameters_valid(thing, p2, p3):
	var bad_msg = ""
	if(p3 != null or typeof(p2) == TYPE_STRING):
		bad_msg += "Doubling using a subpath is not supported.  Call register_inner_class and then pass the Inner Class to double().\n"

	if(typeof(thing) == TYPE_STRING):
		bad_msg += "Doubling using the path to a script or scene is no longer supported.  Load the script or scene and pass that to double instead.\n"

	if(GutUtils.is_instance(thing)):
		bad_msg += "double requires a script, you passed an instance:  " + _str(thing)

	if(bad_msg != ""):
		_lgr.error(bad_msg)

	return bad_msg == ""

# ----------------
#endregion
#region Virtual Methods
# ----------------

## Virtual Method.  This is run after the script has been prepped for execution, but before `before_all` is executed.  If you implement this method and return `true` or a `String` (the string is displayed in the log) then GUT will stop executing the script and mark it as risky.  You might want to do this because:
## - You are porting tests from 3.x to 4.x and you don't want to comment everything out.[br]
## - Skipping tests that should not be run when in `headless` mode such as input testing that does not work in headless.[br]
## [codeblock]
##    func should_skip_script():
##        if DisplayServer.get_name() == "headless":
##            return "Skip Input tests when running headless"
## [/codeblock]
## - If you have tests that would normally cause the debugger to break on an error, you can skip the script if the debugger is enabled so that the run is not interrupted.[br]
## [codeblock]
##    func should_skip_script():
##        return EngineDebugger.is_active()
## [/codeblock]
func should_skip_script():
	return false


## Virtual method.  Run once before anything else in the test script is run.
func before_all():
	pass


## Virtual method.  Run before each test is executed
func before_each():
	pass

## Virtual method.  Run after each test is executed.
func after_each():
	pass


## Virtual method.  Run after all tests have been run.
func after_all():
	pass

# ----------------
#endregion
#region Misc Public
# ----------------
## Mark the current test as pending.
func pending(text=""):
	_summary.pending += 1
	if(gut):
		_lgr.pending(text)
		gut._pending(text)


## Returns true if the test is passing as of the time of this call.  False if not.
func is_passing():
	if(gut.get_current_test_object() != null and
		!['before_all', 'after_all'].has(gut.get_current_test_object().name)):
		return gut.get_current_test_object().is_passing() and \
			gut.get_current_test_object().assert_count > 0
	else:
		_lgr.error('No current test object found.  is_passing must be called inside a test.')
		return null


## Returns true if the test is failing as of the time of this call.  False if not.
func is_failing():
	if(gut.get_current_test_object() != null and
		!['before_all', 'after_all'].has(gut.get_current_test_object().name)):

		return gut.get_current_test_object().is_failing()
	else:
		_lgr.error('No current test object found.  is_failing must be called inside a test.')
		return null


## Marks the test as passing.  Does not override any failing asserts or calls to
## fail_test.  Same as a passing assert.
func pass_test(text):
	_pass(text)


## Marks the test as failing.  Same as a failing assert.
func fail_test(text):
	_fail(text)

## @internal
func clear_signal_watcher():
	_signal_watcher.clear()


## Returns the current double strategy.
func get_double_strategy():
	return gut.get_doubler().get_strategy()


## Sets the double strategy for all tests in the script.  This should usually
## be done in [method before_all].  The double strtegy can be set per
## run/script/double.  See [wiki]Double-Strategy[/wiki]
func set_double_strategy(double_strategy):
	gut.get_doubler().set_strategy(double_strategy)


## This method will cause Gut to pause before it moves on to the next test.
## This is useful for debugging, for instance if you want to investigate the
## screen or anything else after a test has finished executing.
## [br]
## Sometimes you get lazy, and you don't remove calls to
## [code skip-lint]pause_before_teardown[/code] after you are done with them.  You can
## tell GUT to ignore calls to this method through the panel or
## the command line.  Setting this in your `.gutconfig.json` file is recommended
## for CI/CD Pipelines.
func pause_before_teardown():
	gut.pause_before_teardown()


## @internal
func get_logger():
	return _lgr

## @internal
func set_logger(logger):
	_lgr = logger


## This must be called in order to make assertions based on signals being
## emitted.  __Right now, this only supports signals that are emitted with 9 or
## less parameters.__  This can be extended but nine seemed like enough for now.
## The Godot documentation suggests that the limit is four but in my testing
## I found you can pass more.
## [br]
## This must be called in each test in which you want to make signal based
## assertions in.  You can call it multiple times with different objects.
## You should not call it multiple times with the same object in the same test.
## The objects that are watched are cleared after each test (specifically right
## before `teardown` is called).  Under the covers, Gut will connect to all the
## signals an object has and it will track each time they fire.  You can then
## use the following asserts and methods to verify things are acting correct.
func watch_signals(object):
	_signal_watcher.watch_signals(object)


## This will return the number of times a signal was fired.  This gives you
## the freedom to make more complicated assertions if the spirit moves you.
## This will return -1 if the signal was not fired or the object was not being
## watched, or if the object does not have the signal.
## [br][br]
## Accepts either the object and the signal name or the signal.
func get_signal_emit_count(p1, p2=null):
	var sp = SignalAssertParameters.new(p1, p2)
	return _signal_watcher.get_emit_count(sp.object, sp.signal_name)


## If you need to inspect the parameters in order to make more complicate assertions, then this will give you access to
## the parameters of any watched signal.  This works the same way that
## [code skip-lint]assert_signal_emitted_with_parameters[/code] does.  It takes an object, signal name, and an optional
## index.  If the index is not specified then the parameters from the most recent emission will be returned.  If the
## object is not being watched, the signal was not fired, or the object does not have the signal then `null` will be
## returned.
##
## [br][br]
## [b]Signatures:[/b][br]
## - get_signal_parameters([param p1]:Signal, [param p2]:parameter-index (optional))[br]
## - get_signal_parameters([param p1]:object, [param p2]:signal name, [param p3]:parameter-index (optional)) [br]
## [br]
## [b]Examples:[/b]
## [codeblock]
## class SignalObject:
##     signal some_signal
##     signal other_signal
##
##
## func test_get_signal_parameters():
##     var obj = SignalObject.new()
##     watch_signals(obj)
##     obj.some_signal.emit(1, 2, 3)
##     obj.some_signal.emit('a', 'b', 'c')
##
##     # -- Passing --
##     # passes because get_signal_parameters returns the most recent emission
##     # by default
##     assert_eq(get_signal_parameters(obj, 'some_signal'), ['a', 'b', 'c'])
##     assert_eq(get_signal_parameters(obj.some_signal), ['a', 'b', 'c'])
##
##     assert_eq(get_signal_parameters(obj, 'some_signal', 0), [1, 2, 3])
##     assert_eq(get_signal_parameters(obj.some_signal, 0), [1, 2, 3])
##
##     # if the signal was not fired null is returned
##     assert_null(get_signal_parameters(obj, 'other_signal'))
##     # if the signal does not exist or isn't being watched null is returned
##     assert_null(get_signal_parameters(obj, 'signal_dne'))
##
##     # -- Failing --
##     assert_eq(get_signal_parameters(obj, 'some_signal'), [1, 2, 3])
##     assert_eq(get_signal_parameters(obj.some_signal, 0), ['a', 'b', 'c'])
## [/codeblock]
func get_signal_parameters(p1, p2=null, p3=-1):
	var sp := SignalAssertParameters.new(p1, GutUtils.nvl(p2, -1), p3)
	return _signal_watcher.get_signal_parameters(sp.object, sp.signal_name, sp.others[0])


## Get the parameters for a method call to a doubled object.  By default it will
## return the most recent call.  You can optionally specify an index for which
## call you want to get the parameters for.
##
## Can be called using a Callable for the first parameter instead of specifying
## an object and method name.  When you do this, the seoncd parameter is used
## as the index.
##
## Returns:
## * an array of parameter values if a call the method was found
## * null when a call to the method was not found or the index specified was
##   invalid.
func get_call_parameters(object, method_name_or_index = -1, idx=-1):
	var to_return = null
	var index = idx
	if(object is Callable):
		index = method_name_or_index
		method_name_or_index = null
	var converted = _convert_spy_args(object, method_name_or_index, null)

	if(GutUtils.is_double(converted.object)):
		to_return = gut.get_spy().get_call_parameters(
			converted.object, converted.method_name, index)
	else:
		_lgr.error('You must pass a doulbed object to get_call_parameters.')

	return to_return


## Returns the call count for a method with optional paramter matching.
##
## Can be called with a Callable instead of an object, method_name, and
## parameters.  Bound arguments will be used to match call arguments.
func get_call_count(object, method_name=null, parameters=null):
	var converted = _convert_spy_args(object, method_name, parameters)
	return gut.get_spy().call_count(converted.object, converted.method_name, converted.arguments)


## Simulate a number of frames by calling '_process' and '_physics_process' (if
## the methods exist) on an object and all of its descendents. The specified frame
## time, 'delta', will be passed to each simulated call.
##
## NOTE: Objects can disable their processing methods using 'set_process(false)' and
## 'set_physics_process(false)'. This is reflected in the 'Object' methods
## 'is_processing()' and 'is_physics_processing()', respectively. To make 'simulate'
## respect this status, for example if you are testing an object which toggles
## processing, pass 'check_is_processing' as 'true'.
func simulate(obj, times, delta, check_is_processing: bool = false):
	gut.simulate(obj, times, delta, check_is_processing)


# ------------------------------------------------------------------------------
## Replace the node at base_node.get_node(path) with with_this.  All references
## to the node via $ and get_node(...) will now return with_this.  with_this will
## get all the groups that the node that was replaced had.
## [br]
## The node that was replaced is queued to be freed.
## [br]
## TODO see replace_by method, this could simplify the logic here.
# ------------------------------------------------------------------------------
func replace_node(base_node, path_or_node, with_this):
	var path = path_or_node

	if(typeof(path_or_node) != TYPE_STRING):
		# This will cause an engine error if it fails.  It always returns a
		# NodePath, even if it fails.  Checking the name count is the only way
		# I found to check if it found something or not (after it worked I
		# didn't look any farther).
		path = base_node.get_path_to(path_or_node)
		if(path.get_name_count() == 0):
			_lgr.error('You passed an object that base_node does not have.  Cannot replace node.')
			return

	if(!base_node.has_node(path)):
		_lgr.error(str('Could not find node at path [', path, ']'))
		return

	var to_replace = base_node.get_node(path)
	var parent = to_replace.get_parent()
	var replace_name = to_replace.get_name()

	parent.remove_child(to_replace)
	parent.add_child(with_this)
	with_this.set_name(replace_name)
	with_this.set_owner(parent)

	var groups = to_replace.get_groups()
	for i in range(groups.size()):
		with_this.add_to_group(groups[i])

	to_replace.queue_free()


## Use this as the default value for the first parameter to a test to create
## a parameterized test.  See also the ParameterFactory and Parameterized Tests.
## [br][br]
## [b]Example[/b]
## [codeblock]
##    func test_with_parameters(p = use_parameters([1, 2, 3])):
## [/codeblock]
func use_parameters(params):
	var ph = gut.parameter_handler
	if(ph == null):
		ph = GutUtils.ParameterHandler.new(params)
		gut.parameter_handler = ph

	# DO NOT use gut.gd's get_call_count_text here since it decrements the
	# get_call_count value.  This method increments the call count in its
	# return statement.
	var output = str('- params[', ph.get_call_count(), ']','(', ph.get_current_parameters(), ')')
	gut.p(output, gut.LOG_LEVEL_TEST_AND_FAILURES)

	return ph.next_parameters()


## @internal
## When used as the default for a test method parameter, it will cause the test
## to be run x times.
##
## I Hacked this together to test a method that was occassionally failing due to
## timing issues.  I don't think it's a great idea, but you be the judge.  If
## you find a good use for it, let me know and I'll make it a legit member
## of the api.
func run_x_times(x):
	var ph = gut.parameter_handler
	if(ph == null):
		_lgr.warn(
			str("This test uses run_x_times and you really should not be ",
			"using it.  I don't think it's a good thing, but I did find it ",
			"temporarily useful so I left it in here and didn't document it.  ",
			"Well, you found it, might as well open up an issue and let me ",
			"know why you're doing this."))
		var params = []
		for i in range(x):
			params.append(i)

		ph = GutUtils.ParameterHandler.new(params)
		gut.parameter_handler = ph
	return ph.next_parameters()


## Checks the passed in version string (x.x.x) against the engine version to see
## if the engine version is less than the expected version.  If it is then the
## test is mareked as passed (for a lack of anything better to do).  The result
## of the check is returned.
## [br][br]
## [b]Example[/b]
## [codeblock]
##    if(skip_if_godot_version_lt('3.5.0')):
##        return
## [/codeblock]
func skip_if_godot_version_lt(expected):
	var should_skip = !GutUtils.is_godot_version_gte(expected)
	if(should_skip):
		_pass(str('Skipping: ', GutUtils.godot_version_string(), ' is less than ', expected))
	return should_skip


## Checks if the passed in version matches the engine version.  The passed in
## version can contain just the major, major.minor or major.minor.path.  If
## the version is not the same then the test is marked as passed.  The result of
## the check is returned.
## [br][br]
## [b]Example[/b]
## [codeblock]
##     if(skip_if_godot_version_ne('3.4')):
##        return
## [/codeblock]
func skip_if_godot_version_ne(expected):
	var should_skip = !GutUtils.is_godot_version(expected)
	if(should_skip):
		_pass(str('Skipping: ', GutUtils.godot_version_string(), ' is not ', expected))
	return should_skip


## Registers all the inner classes in a script with the doubler.  This is required
## before you can double any inner class.
func register_inner_classes(base_script):
	gut.get_doubler().inner_class_registry.register(base_script)


## Peforms a deep compare on both values, a CompareResult instnace is returned.
## The optional max_differences paramter sets the max_differences to be displayed.
func compare_deep(v1, v2, max_differences=null):
	var result = _compare.deep(v1, v2)
	if(max_differences != null):
		result.max_differences = max_differences
	return result


# ----------------
#endregion
#region Asserts
# ----------------

## Asserts that the expected value equals the value got.
## assert got == expected and prints optional text.  See [wiki]Comparing-Things[/wiki]
## for information about comparing dictionaries and arrays.
## [br]
## See also: [method assert_ne], [method assert_same], [method assert_not_same]
## [codeblock]
##    var one = 1
##    var node1 = Node.new()
##    var node2 = node1
##
##    # Passing
##    assert_eq(one, 1, 'one should equal one')
##    assert_eq('racecar', 'racecar')
##    assert_eq(node2, node1)
##    assert_eq([1, 2, 3], [1, 2, 3])
##    var d1_pass = {'a':1}
##    var d2_pass = d1_pass
##    assert_eq(d1_pass, d2_pass)
##
##    # Failing
##    assert_eq(1, 2) # FAIL
##    assert_eq('hello', 'world')
##    assert_eq(self, node1)
##    assert_eq([1, 'two', 3], [1, 2, 3, 4])
##    assert_eq({'a':1}, {'a':1})
## [/codeblock]
func assert_eq(got, expected, text=""):

	if(_do_datatypes_match__fail_if_not(got, expected, text)):
		var disp = "[" + _str(got) + "] expected to equal [" + _str(expected) + "]:  " + text
		var result = null

		result = _compare.simple(got, expected)

		if(typeof(got) in [TYPE_ARRAY, TYPE_DICTIONARY]):
			disp = str(result.summary, '  ', text)
			_lgr.info('Array/Dictionary compared by value.  Use assert_same to compare references.  Use assert_eq_deep to see diff when failing.')

		if(result.are_equal):
			_pass(disp)
		else:
			_fail(disp)


## asserts got != expected and prints optional text.  See
## [wiki]Comparing-Things[/wiki] for information about comparing dictionaries
## and arrays.
##[br]
## See also: [method assert_eq], [method assert_same], [method assert_not_same]
## [codeblock]
##    var two = 2
##    var node1 = Node.new()
##
##    # Passing
##    assert_ne(two, 1, 'Two should not equal one.')
##    assert_ne('hello', 'world')
##    assert_ne(self, node1)
##
##    # Failing
##    assert_ne(two, 2)
##    assert_ne('one', 'one')
##    assert_ne('2', 2)
## [/codeblock]
func assert_ne(got, not_expected, text=""):
	if(_do_datatypes_match__fail_if_not(got, not_expected, text)):
		var disp = "[" + _str(got) + "] expected to not equal [" + _str(not_expected) + "]:  " + text
		var result = null

		result = _compare.simple(got, not_expected)

		if(typeof(got) in [TYPE_ARRAY, TYPE_DICTIONARY]):
			disp = str(result.summary, '  ', text)
			_lgr.info('Array/Dictionary compared by value.  Use assert_not_same to compare references.  Use assert_ne_deep to see diff.')

		if(result.are_equal):
			_fail(disp)
		else:
			_pass(disp)


## Asserts that [param got] is within the range of [param expected] +/- [param error_interval].
## The upper and lower bounds are included in the check.  Verified to work with
## integers, floats, and Vector2.  Should work with anything that can be
## added/subtracted.
##
## [codeblock]
##    # Passing
##    assert_almost_eq(0, 1, 1, '0 within range of 1 +/- 1')
##    assert_almost_eq(2, 1, 1, '2 within range of 1 +/- 1')
##    assert_almost_eq(1.2, 1.0, .5, '1.2 within range of 1 +/- .5')
##    assert_almost_eq(.5, 1.0, .5, '.5 within range of 1 +/- .5')
##    assert_almost_eq(Vector2(.5, 1.5), Vector2(1.0, 1.0), Vector2(.5, .5))
##    assert_almost_eq(Vector2(.5, 1.5), Vector2(1.0, 1.0), Vector2(.25, .25))
##
##    # Failing
##    assert_almost_eq(1, 3, 1, '1 outside range of 3 +/- 1')
##    assert_almost_eq(2.6, 3.0, .2, '2.6 outside range of 3 +/- .2')
## [/codeblock]
func assert_almost_eq(got, expected, error_interval, text=''):
	var disp = "[" + _str_precision(got, 20) + "] expected to equal [" + _str(expected) + "] +/- [" + str(error_interval) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, expected, text) and _do_datatypes_match__fail_if_not(got, error_interval, text)):
		if not _is_almost_eq(got, expected, error_interval):
			_fail(disp)
		else:
			_pass(disp)


## This is the inverse of [method assert_almost_eq].  This will pass if [param got] is
## outside the range of [param not_expected] +/- [param error_interval].
func assert_almost_ne(got, not_expected, error_interval, text=''):
	var disp = "[" + _str_precision(got, 20) + "] expected to not equal [" + _str(not_expected) + "] +/- [" + str(error_interval) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, not_expected, text) and _do_datatypes_match__fail_if_not(got, error_interval, text)):
		if _is_almost_eq(got, not_expected, error_interval):
			_fail(disp)
		else:
			_pass(disp)

# ------------------------------------------------------------------------------
# Helper function compares a value against a expected and a +/- range.  Compares
# all components of Vector2, Vector3, and Vector4 as well.
# ------------------------------------------------------------------------------
func _is_almost_eq(got, expected, error_interval) -> bool:
	var result = false
	var upper = expected + error_interval
	var lower = expected - error_interval

	if typeof(got) in [TYPE_VECTOR2, TYPE_VECTOR3, TYPE_VECTOR4]:
		result = got.clamp(lower, upper) == got
	else:
		result = got >= (lower) and got <= (upper)

	return(result)

## assserts got > expected
## [codeblock]
##    var bigger = 5
##    var smaller = 0
##
##    # Passing
##    assert_gt(bigger, smaller, 'Bigger should be greater than smaller')
##    assert_gt('b', 'a')
##    assert_gt('a', 'A')
##    assert_gt(1.1, 1)
##
##    # Failing
##    assert_gt('a', 'a')
##    assert_gt(1.0, 1)
##    assert_gt(smaller, bigger)
## [/codeblock]
func assert_gt(got, expected, text=""):
	var disp = "[" + _str(got) + "] expected to be > than [" + _str(expected) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, expected, text)):
		if(got > expected):
			_pass(disp)
		else:
			_fail(disp)


## Asserts got is greater than or equal to expected.
## [codeblock]
##    var bigger = 5
##    var smaller = 0
##
##    # Passing
##    assert_gte(bigger, smaller, 'Bigger should be greater than or equal to smaller')
##    assert_gte('b', 'a')
##    assert_gte('a', 'A')
##    assert_gte(1.1, 1)
##    assert_gte('a', 'a')
##
##    # Failing
##    assert_gte(0.9, 1.0)
##    assert_gte(smaller, bigger)
## [/codeblock]
func assert_gte(got, expected, text=""):
	var disp = "[" + _str(got) + "] expected to be >= than [" + _str(expected) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, expected, text)):
		if(got >= expected):
			_pass(disp)
		else:
			_fail(disp)

## Asserts [param got] is less than [param expected]
## [codeblock]
##    var bigger = 5
##    var smaller = 0
##
##    # Passing
##    assert_lt(smaller, bigger, 'Smaller should be less than bigger')
##    assert_lt('a', 'b')
##    assert_lt(99, 100)
##
##    # Failing
##    assert_lt('z', 'x')
##    assert_lt(-5, -5)
## [/codeblock]
func assert_lt(got, expected, text=""):
	var disp = "[" + _str(got) + "] expected to be < than [" + _str(expected) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, expected, text)):
		if(got < expected):
			_pass(disp)
		else:
			_fail(disp)


## Asserts got is less than or equal to expected
func assert_lte(got, expected, text=""):
	var disp = "[" + _str(got) + "] expected to be <= than [" + _str(expected) + "]:  " + text
	if(_do_datatypes_match__fail_if_not(got, expected, text)):
		if(got <= expected):
			_pass(disp)
		else:
			_fail(disp)


## asserts that got is true.  Does not assert truthiness, only boolean values
## will pass.
func assert_true(got, text=""):
	if(typeof(got) == TYPE_BOOL):
		if(got):
			_pass(text)
		else:
			_fail(text)
	else:
		var msg = str("Cannot convert ", _strutils.type2str(got), " to boolean")
		_fail(msg)


## Asserts that got is false.  Does not assert truthiness, only boolean values
## will pass.
func assert_false(got, text=""):
	if(typeof(got) == TYPE_BOOL):
		if(got):
			_fail(text)
		else:
			_pass(text)
	else:
		var msg = str("Cannot convert ", _strutils.type2str(got), " to boolean")
		_fail(msg)


## Asserts value is between (inclusive) the two expected values.[br]
## got >= expect_low and <= expect_high
## [codeblock]
##    # Passing
##    assert_between(5, 0, 10, 'Five should be between 0 and 10')
##    assert_between(10, 0, 10)
##    assert_between(0, 0, 10)
##    assert_between(2.25, 2, 4.0)
##
##    # Failing
##    assert_between('a', 'b', 'c')
##    assert_between(1, 5, 10)
## [/codeblock]
func assert_between(got, expect_low, expect_high, text=""):
	var disp = "[" + _str_precision(got, 20) + "] expected to be between [" + _str(expect_low) + "] and [" + str(expect_high) + "]:  " + text

	if(_do_datatypes_match__fail_if_not(got, expect_low, text) and _do_datatypes_match__fail_if_not(got, expect_high, text)):
		if(expect_low > expect_high):
			disp = "INVALID range.  [" + str(expect_low) + "] is not less than [" + str(expect_high) + "]"
			_fail(disp)
		else:
			if(got < expect_low or got > expect_high):
				_fail(disp)
			else:
				_pass(disp)


## Asserts value is not between (exclusive) the two expected values.[br]
## asserts that got <= expect_low or got >=  expect_high.
## [codeblock]
##    # Passing
##    assert_not_between(1, 5, 10)
##    assert_not_between('a', 'b', 'd')
##    assert_not_between('d', 'b', 'd')
##    assert_not_between(10, 0, 10)
##    assert_not_between(-2, -2, 10)
##
##    # Failing
##    assert_not_between(5, 0, 10, 'Five shouldnt be between 0 and 10')
##    assert_not_between(0.25, -2.0, 4.0)
## [/codeblock]
func assert_not_between(got, expect_low, expect_high, text=""):
	var disp = "[" + _str_precision(got, 20) + "] expected not to be between [" + _str(expect_low) + "] and [" + str(expect_high) + "]:  " + text

	if(_do_datatypes_match__fail_if_not(got, expect_low, text) and _do_datatypes_match__fail_if_not(got, expect_high, text)):
		if(expect_low > expect_high):
			disp = "INVALID range.  [" + str(expect_low) + "] is not less than [" + str(expect_high) + "]"
			_fail(disp)
		else:
			if(got > expect_low and got < expect_high):
				_fail(disp)
			else:
				_pass(disp)


## Uses the 'has' method of the object passed in to determine if it contains
## the passed in element.
## [codeblock]
##    var an_array = [1, 2, 3, 'four', 'five']
##    var a_hash = { 'one':1, 'two':2, '3':'three'}
##
##    # Passing
##    assert_has(an_array, 'four') # PASS
##    assert_has(an_array, 2) # PASS
##    # the hash's has method checks indexes not values
##    assert_has(a_hash, 'one') # PASS
##    assert_has(a_hash, '3') # PASS
##
##    # Failing
##    assert_has(an_array, 5) # FAIL
##    assert_has(an_array, self) # FAIL
##    assert_has(a_hash, 3) # FAIL
##    assert_has(a_hash, 'three') # FAIL
## [/codeblock]
func assert_has(obj, element, text=""):
	var disp = str('Expected [', _str(obj), '] to contain value:  [', _str(element), ']:  ', text)
	if(obj.has(element)):
		_pass(disp)
	else:
		_fail(disp)


## The inverse of assert_has.
func assert_does_not_have(obj, element, text=""):
	var disp = str('Expected [', _str(obj), '] to NOT contain value:  [', _str(element), ']:  ', text)
	if(obj.has(element)):
		_fail(disp)
	else:
		_pass(disp)


## asserts a file exists at the specified path
## [codeblock]
##    func before_each():
##        gut.file_touch('user://some_test_file')
##
##    func after_each():
##        gut.file_delete('user://some_test_file')
##
##    func test_assert_file_exists():
##        # Passing
##        assert_file_exists('res://addons/gut/gut.gd')
##        assert_file_exists('user://some_test_file')
##
##        # Failing
##        assert_file_exists('user://file_does_not.exist')
##        assert_file_exists('res://some_dir/another_dir/file_does_not.exist')
## [/codeblock]
func assert_file_exists(file_path):
	var disp = 'expected [' + file_path + '] to exist.'
	if(FileAccess.file_exists(file_path)):
		_pass(disp)
	else:
		_fail(disp)


## asserts a file does not exist at the specified path
## [codeblock]
##    func before_each():
##        gut.file_touch('user://some_test_file')
##
##    func after_each():
##        gut.file_delete('user://some_test_file')
##
##    func test_assert_file_does_not_exist():
##        # Passing
##        assert_file_does_not_exist('user://file_does_not.exist')
##        assert_file_does_not_exist('res://some_dir/another_dir/file_does_not.exist')
##
##        # Failing
##        assert_file_does_not_exist('res://addons/gut/gut.gd')
## [/codeblock]
func assert_file_does_not_exist(file_path):
	var disp = 'expected [' + file_path + '] to NOT exist'
	if(!FileAccess.file_exists(file_path)):
		_pass(disp)
	else:
		_fail(disp)


## asserts the specified file is empty
## [codeblock]
##    func before_each():
##        gut.file_touch('user://some_test_file')
##
##    func after_each():
##        gut.file_delete('user://some_test_file')
##
##    func test_assert_file_empty():
##        # Passing
##        assert_file_empty('user://some_test_file')
##
##        # Failing
##        assert_file_empty('res://addons/gut/gut.gd')
## [/codeblock]
func assert_file_empty(file_path):
	var disp = 'expected [' + file_path + '] to be empty'
	if(FileAccess.file_exists(file_path) and gut.is_file_empty(file_path)):
		_pass(disp)
	else:
		_fail(disp)


## Asserts the specified file is not empty
## [codeblock]
##    func before_each():
##        gut.file_touch('user://some_test_file')
##
##    func after_each():
##        gut.file_delete('user://some_test_file')
##
##    func test_assert_file_not_empty():
##        # Passing
##        assert_file_not_empty('res://addons/gut/gut.gd') # PASS
##
##        # Failing
##        assert_file_not_empty('user://some_test_file') # FAIL
## [/codeblock]
func assert_file_not_empty(file_path):
	var disp = 'expected [' + file_path + '] to contain data'
	if(!gut.is_file_empty(file_path)):
		_pass(disp)
	else:
		_fail(disp)


## Asserts that the passed in object has a method named [param method].
func assert_has_method(obj, method, text=''):
	var disp = _str(obj) + ' should have method: ' + method
	if(text != ''):
		disp = _str(obj) + ' ' + text
	assert_true(obj.has_method(method), disp)


## This is meant to make testing public get/set methods for a member variable.  This was originally created for early Godot 3.x setter and getter methods.  See [method assert_property] for verifying Godot 4.x accessors.  This makes multiple assertions to verify:
## [br]
## [li]The object has a method called [code]get_<PROPERTY_NAME>[/code][/li]
## [li]The object has a method called [code]set_<PROPERTY_NAME>[/code][/li]
## [li]The method [code]get_<PROPERTY_NAME>[/code] returns the expected default value when first called.[/li]
## [li]Once you set the property, the [code]get_<PROPERTY_NAME>[/code] returns the new value.[/li]
## [br]
func assert_accessors(obj, property, default, set_to):
	var fail_count = _summary.failed
	var get_func = 'get_' + property
	var set_func = 'set_' + property

	if(obj.has_method('is_' + property)):
		get_func = 'is_' + property

	assert_has_method(obj, get_func, 'should have getter starting with get_ or is_')
	assert_has_method(obj, set_func)
	# SHORT CIRCUIT
	if(_summary.failed > fail_count):
		return
	assert_eq(obj.call(get_func), default, 'It should have the expected default value.')
	obj.call(set_func, set_to)
	assert_eq(obj.call(get_func), set_to, 'The set value should have been returned.')


# Property search helper.  Used to retrieve Dictionary of specified property
# from passed object. Returns null if not found.
# If provided, property_usage constrains the type of property returned by
# passing either:
# EDITOR_PROPERTY for properties defined as: export var some_value: int
# VARIABLE_PROPERTY for properties defined as: var another_value
func _find_object_property(obj, property_name, property_usage=null):
	var result = null
	var found = false
	var properties = obj.get_property_list()

	while !found and !properties.is_empty():
		var property = properties.pop_back()
		if property['name'] == property_name:
			if property_usage == null or property['usage'] == property_usage:
				result = property
				found = true
	return result


## Asserts that [param obj] exports a property with the name
## [param property_name] and a type of [param type].  The [param type] must be
## one of the various Godot built-in [code]TYPE_[/code] constants.
## [codeblock]
##    class ExportClass:
##        export var some_number = 5
##        export(PackedScene) var some_scene
##        var some_variable = 1
##
##    func test_assert_exports():
##        var obj = ExportClass.new()
##
##        # Passing
##        assert_exports(obj, "some_number", TYPE_INT)
##        assert_exports(obj, "some_scene", TYPE_OBJECT)
##
##        # Failing
##        assert_exports(obj, 'some_number', TYPE_VECTOR2)
##        assert_exports(obj, 'some_scene', TYPE_AABB)
##        assert_exports(obj, 'some_variable', TYPE_INT)
## [/codeblock]
func assert_exports(obj, property_name, type):
	var disp = 'expected %s to have editor property [%s]' % [_str(obj), property_name]
	var property = _find_object_property(obj, property_name, EDITOR_PROPERTY)
	if property != null:
		disp += ' of type [%s]. Got type [%s].' % [_strutils.types[type], _strutils.types[property['type']]]
		if property['type'] == type:
			_pass(disp)
		else:
			_fail(disp)
	else:
		_fail(disp)


# Signal assertion helper.
#
# Verifies that the object and signal are valid for making signal assertions.
# This will fail with specific messages that indicate why they are not valid.
# This returns true/false to indicate if the object and signal are valid.
func _can_make_signal_assertions(object, signal_name):
	return !(_fail_if_not_watching(object) or _fail_if_does_not_have_signal(object, signal_name))


# Check if an object is connected to a signal on another object. Returns True
# if it is and false otherwise
func _is_connected(signaler_obj, connect_to_obj, signal_name, method_name=""):
	if(method_name != ""):
		return signaler_obj.is_connected(signal_name,Callable(connect_to_obj,method_name))
	else:
		var connections = signaler_obj.get_signal_connection_list(signal_name)
		for conn in connections:
			if(conn['signal'].get_name() == signal_name and conn['callable'].get_object() == connect_to_obj):
				return true
		return false


## Asserts that `signaler_obj` is connected to `connect_to_obj` on signal `signal_name`.  The method that is connected is optional.  If `method_name` is supplied then this will pass only if the signal is connected to the  method.  If it is not provided then any connection to the signal will cause a pass.
## [br][br]
## [b]Signatures:[/b][br]
## - assert_connected([param p1]:Signal, [param p2]:connected-object)[br]
## - assert_connected([param p1]:Signal, [param p2]:connected-method)[br]
## - assert_connected([param p1]:object, [param p2]:connected-object, [param p3]:signal-name, [param p4]: connected-method-name <optional>)
## [br][br]
## [b]Examples:[/b]
## [codeblock]
## class Signaler:
##     signal the_signal
##
## class Connector:
##     func connect_this():
##         pass
##     func  other_method():
##         pass
##
## func test_assert_connected():
##     var signaler = Signaler.new()
##     var connector  = Connector.new()
##     signaler.the_signal.connect(connector.connect_this)
##
##     # Passing
##     assert_connected(signaler.the_signal, connector.connect_this)
##     assert_connected(signaler.the_signal, connector)
##     assert_connected(signaler, connector, 'the_signal')
##     assert_connected(signaler, connector, 'the_signal', 'connect_this')
##
##     # Failing
##     assert_connected(signaler.the_signal, connector.other_method)
##
##     var foo = Connector.new()
##     assert_connected(signaler,  connector, 'the_signal', 'other_method')
##     assert_connected(signaler, connector, 'other_signal')
##     assert_connected(signaler, foo, 'the_signal')
## [/codeblock]
func assert_connected(p1, p2, p3=null, p4=""):
	var sp := SignalAssertParameters.new(p1, p3)
	var connect_to_obj = p2
	var method_name = p4

	if(connect_to_obj is  Callable):
		method_name = connect_to_obj.get_method()
		connect_to_obj = connect_to_obj.get_object()

	var method_disp = ''
	if (method_name != ""):
		method_disp = str(' using method: [', method_name, '] ')
	var disp = str('Expected object ', _str(sp.object),\
		' to be connected to signal: [', sp.signal_name, '] on ',\
		_str(connect_to_obj), method_disp)
	if(_is_connected(sp.object, connect_to_obj, sp.signal_name, method_name)):
		_pass(disp)
	else:
		_fail(disp)


## The inverse of [method assert_connected].  See [method assert_connected] for parameter syntax.
## [br]
## This will fail with specific messages if the target object is connected to the specified signal on the source object.
func assert_not_connected(p1, p2, p3=null, p4=""):
	var sp := SignalAssertParameters.new(p1, p3)
	var connect_to_obj = p2
	var method_name = p4

	if(connect_to_obj is  Callable):
		method_name = connect_to_obj.get_method()
		connect_to_obj = connect_to_obj.get_object()

	var method_disp = ''
	if (method_name != ""):
		method_disp = str(' using method: [', method_name, '] ')
	var disp = str('Expected object ', _str(sp.object),\
		' to not be connected to signal: [', sp.signal_name, '] on ',\
		_str(sp.object), method_disp)
	if(_is_connected(sp.object, connect_to_obj, sp.signal_name, method_name)):
		_fail(disp)
	else:
		_pass(disp)


## Assert that the specified object emitted the named signal.  You must call
## [method watch_signals] and pass it the object that you are making assertions about.
## This will fail if the object is not being watched or if the object does not
## have the specified signal.  Since this will fail if the signal does not
## exist, you can often skip using [method assert_has_signal].
## [br][br]
## [b]Signatures:[/b][br]
## - assert_signal_emitted([param p1]:Signal, [param p2]: text <optional>)[br]
## - assert_signal_emitted([param p1]:object, [param p2]:signal-name, [param p3]: text <optional>)
## [br][br]
## [b]Examples:[/b]
## [codeblock]
## class SignalObject:
##     signal some_signal
##     signal other_signal
##
##
## func test_assert_signal_emitted():
##     var obj = SignalObject.new()
##
##     watch_signals(obj)
##     obj.emit_signal('some_signal')
##
##     ## Passing
##     assert_signal_emitted(obj, 'some_signal')
##     assert_signal_emitted(obj.some_signal)
##
##     ## Failing
##     # Fails with specific message that the object does not have the signal
##     assert_signal_emitted(obj, 'signal_does_not_exist')
##     # Fails because the object passed is not being watched
##     assert_signal_emitted(SignalObject.new(), 'some_signal')
##     # Fails because the signal was not emitted
##     assert_signal_emitted(obj, 'other_signal')
##     assert_signal_emitted(obj.other_signal)
## [/codeblock]
func assert_signal_emitted(p1, p2='', p3=""):
	var sp := SignalAssertParameters.new(p1, p2, p3)
	var disp = str('Expected object ', _str(sp.object), ' to have emitted signal [', sp.signal_name, ']:  ', sp.others[0])
	if(_can_make_signal_assertions(sp.object, sp.signal_name)):
		if(_signal_watcher.did_emit(sp.object, sp.signal_name)):
			_pass(disp)
		else:
			_fail(_get_fail_msg_including_emitted_signals(disp, sp.object))


## This works opposite of `assert_signal_emitted`.  This will fail if the object
## is not being watched or if the object does not have the signal.
## [br][br]
## [b]Signatures:[/b][br]
## - assert_signal_not_emitted([param p1]:Signal, [param p2]: text <optional>)[br]
## - assert_signal_not_emitted([param p1]:object, [param p2]:signal-name, [param p3]: text <optional>)
## [br][br]
## [b]Examples:[/b]
## [codeblock]
##    class SignalObject:
##        signal some_signal
##        signal other_signal
##
##    func test_assert_signal_not_emitted():
##        var obj = SignalObject.new()
##
##        watch_signals(obj)
##        obj.emit_signal('some_signal')
##
##        # Passing
##        assert_signal_not_emitted(obj, 'other_signal')
##        assert_signal_not_emitted(obj.other_signal)
##
##        # Failing
##        # Fails with specific message that the object does not have the signal
##        assert_signal_not_emitted(obj, 'signal_does_not_exist')
##        # Fails because the object passed is not being watched
##        assert_signal_not_emitted(SignalObject.new(), 'some_signal')
##        # Fails because the signal was emitted
##        assert_signal_not_emitted(obj, 'some_signal')
## [/codeblock]
func assert_signal_not_emitted(p1, p2='', p3=''):
	var sp := SignalAssertParameters.new(p1, p2, p3)
	var disp = str('Expected object ', _str(sp.object), ' to NOT emit signal [', sp.signal_name, ']:  ', sp.others[0])
	if(_can_make_signal_assertions(sp.object, sp.signal_name)):
		if(_signal_watcher.did_emit(sp.object, sp.signal_name)):
			_fail(disp)
		else:
			_pass(disp)


## Asserts that a signal was fired with the specified parameters.  The expected
## parameters should be passed in as an array.  An optional index can be passed
## when a signal has fired more than once.  The default is to retrieve the most
## recent emission of the signal.
## [br]
## This will fail with specific messages if the object is not being watched or
## the object does not have the specified signal
## [br][br]
## [b]Signatures:[/b][br]
## - assert_signal_emitted_with_parameters([param p1]:Signal, [param p2]:expected-parameters, [param p3]: index <optional>)[br]
## - assert_signal_emitted_with_parameters([param p1]:object, [param p2]:signal-name, [param p3]:expected-parameters, [param p4]: index <optional>)
## [br][br]
## [b]Examples:[/b]
## [codeblock]
## class SignalObject:
##     signal some_signal
##     signal other_signal
##
## func test_assert_signal_emitted_with_parameters():
##     var obj = SignalObject.new()
##
##     watch_signals(obj)
##     # emit the signal 3 times to illustrate how the index works in
##     # assert_signal_emitted_with_parameters
##     obj.emit_signal('some_signal', 1, 2, 3)
##     obj.emit_signal('some_signal', 'a', 'b', 'c')
##     obj.emit_signal('some_signal', 'one', 'two', 'three')
##
##     # Passing
##     # Passes b/c the default parameters to check are the last emission of
##     # the signal
##     assert_signal_emitted_with_parameters(obj, 'some_signal', ['one', 'two', 'three'])
##     assert_signal_emitted_with_parameters(obj.some_signal, ['one', 'two', 'three'])
##
##     # Passes because the parameters match the specified emission based on index.
##     assert_signal_emitted_with_parameters(obj, 'some_signal', [1, 2, 3], 0)
##     assert_signal_emitted_with_parameters(obj.some_signal, [1, 2, 3], 0)
##
##     # Failing
##     # Fails with specific message that the object does not have the signal
##     assert_signal_emitted_with_parameters(obj, 'signal_does_not_exist', [])
##     # Fails because the object passed is not being watched
##     assert_signal_emitted_with_parameters(SignalObject.new(), 'some_signal', [])
##     # Fails because parameters do not match latest emission
##     assert_signal_emitted_with_parameters(obj, 'some_signal', [1, 2, 3])
##     # Fails because the parameters for the specified index do not match
##     assert_signal_emitted_with_parameters(obj, 'some_signal', [1, 2, 3], 1)
## [/codeblock]
func assert_signal_emitted_with_parameters(p1, p2, p3=-1, p4=-1):
	var sp := SignalAssertParameters.new(p1, p2, p3, p4)
	var parameters = sp.others[0]
	var index = sp.others[1]

	if(typeof(parameters) != TYPE_ARRAY):
		_lgr.error("The expected parameters must be wrapped in an array, you passed:  " + _str(parameters))
		_fail("Bad Parameters")
		return

	var disp = str('Expected object ', _str(sp.object), ' to emit signal [', sp.signal_name, '] with parameters ', parameters, ', got ')
	if(_can_make_signal_assertions(sp.object, sp.signal_name)):
		if(_signal_watcher.did_emit(sp.object, sp.signal_name)):
			var parms_got = _signal_watcher.get_signal_parameters(sp.object, sp.signal_name, index)
			var diff_result = _compare.deep(parameters, parms_got)
			if(diff_result.are_equal):
				_pass(str(disp, parms_got))
			else:
				_fail(str('Expected object ', _str(sp.object), ' to emit signal [', sp.signal_name, '] with parameters ', diff_result.summarize()))
		else:
			var text = str('Object ', sp.object, ' did not emit signal [', sp.signal_name, ']')
			_fail(_get_fail_msg_including_emitted_signals(text, sp.object))


## Asserts that a signal fired a specific number of times.
## [br][br]
## [b]Signatures:[/b][br]
## - assert_signal_emit_count([param p1]:Signal, [param p2]:expected-count, [param p3]: text <optional>)[br]
## - assert_signal_emit_count([param p1]:object, [param p2]:signal-name, [param p3]:expected-count, [param p4]: text <optional>)
## [br][br]
## [b]Examples:[/b]
## [codeblock]
## class SignalObject:
##     signal some_signal
##     signal other_signal
##
##
## func test_assert_signal_emit_count():
##     var obj_a = SignalObject.new()
##     var obj_b = SignalObject.new()
##
##     watch_signals(obj_a)
##     watch_signals(obj_b)
##
##     obj_a.emit_signal('some_signal')
##     obj_a.emit_signal('some_signal')
##
##     obj_b.emit_signal('some_signal')
##     obj_b.emit_signal('other_signal')
##
##     # Passing
##     assert_signal_emit_count(obj_a, 'some_signal', 2, 'passes')
##     assert_signal_emit_count(obj_a.some_signal, 2, 'passes')
##
##     assert_signal_emit_count(obj_a, 'other_signal', 0)
##     assert_signal_emit_count(obj_a.other_signal, 0)
##
##     assert_signal_emit_count(obj_b, 'other_signal', 1)
##
##     # Failing
##     # Fails with specific message that the object does not have the signal
##     assert_signal_emit_count(obj_a, 'signal_does_not_exist', 99)
##     # Fails because the object passed is not being watched
##     assert_signal_emit_count(SignalObject.new(), 'some_signal', 99)
##     # The following fail for obvious reasons
##     assert_signal_emit_count(obj_a, 'some_signal', 0)
##     assert_signal_emit_count(obj_b, 'other_signal', 283)
## [/codeblock]
func assert_signal_emit_count(p1, p2, p3=0, p4=""):
	var sp := SignalAssertParameters.new(p1, p2, p3, p4)
	var times = sp.others[0]
	var text = sp.others[1]

	if(_can_make_signal_assertions(sp.object, sp.signal_name)):
		var count = _signal_watcher.get_emit_count(sp.object, sp.signal_name)
		var disp = str('Expected the signal [', sp.signal_name, '] emit count of [', count, '] to equal [', times, ']: ', text)
		if(count== times):
			_pass(disp)
		else:
			_fail(_get_fail_msg_including_emitted_signals(disp, sp.object))


## Asserts the passed in object has a signal with the specified name.  It
## should be noted that all the asserts that verify a signal was/wasn't emitted
## will first check that the object has the signal being asserted against.  If
## it does not, a specific failure message will be given.  This means you can
## usually skip the step of specifically verifying that the object has a signal
## and move on to making sure it emits the signal correctly.
## [codeblock]
##    class SignalObject:
##        signal some_signal
##        signal other_signal
##
##    func test_assert_has_signal():
##        var obj = SignalObject.new()
##
##        ## Passing
##        assert_has_signal(obj, 'some_signal')
##        assert_has_signal(obj, 'other_signal')
##
##        ## Failing
##        assert_has_signal(obj, 'not_a real SIGNAL')
##        assert_has_signal(obj, 'yea, this one doesnt exist either')
##        # Fails because the signal is not a user signal.  Node2D does have the
##        # specified signal but it can't be checked this way.  It could be watched
##        # and asserted that it fired though.
##        assert_has_signal(Node2D.new(), 'exit_tree')
## [/codeblock]
func assert_has_signal(object, signal_name, text=""):
	var disp = str('Expected object ', _str(object), ' to have signal [', signal_name, ']:  ', text)
	if(_signal_watcher.does_object_have_signal(object, signal_name)):
		_pass(disp)
	else:
		_fail(disp)


## Asserts that [param object] extends [param a_class].  object must be an instance of an
## object.  It cannot be any of the built in classes like Array or Int or Float.
## [param a_class] must be a class, it can be loaded via load, a GDNative class such as
## Node or Label or anything else.
## [codeblock]
##    # Passing
##    assert_is(Node2D.new(), Node2D)
##    assert_is(Label.new(), CanvasItem)
##    assert_is(SubClass.new(), BaseClass)
##    # Since this is a test script that inherits from test.gd, so
##    # this passes.  It's not obvious w/o seeing the whole script
##    # so I'm telling you.  You'll just have to trust me.
##    assert_is(self, load('res://addons/gut/test.gd'))
##
##    var Gut = load('res://addons/gut/gut.gd')
##    var a_gut = Gut.new()
##    assert_is(a_gut, Gut)
##
##    # Failing
##    assert_is(Node2D.new(), Node2D.new())
##    assert_is(BaseClass.new(), SubClass)
##    assert_is('a', 'b')
##    assert_is([], Node)
## [/codeblock]
func assert_is(object, a_class, text=''):
	var disp  = ''#var disp = str('Expected [', _str(object), '] to be type of [', a_class, ']: ', text)
	var bad_param_2 = 'Parameter 2 must be a Class (like Node2D or Label).  You passed '

	if(typeof(object) != TYPE_OBJECT):
		_fail(str('Parameter 1 must be an instance of an object.  You passed:  ', _str(object)))
	elif(typeof(a_class) != TYPE_OBJECT):
		_fail(str(bad_param_2, _str(a_class)))
	else:
		var a_str = _str(a_class)
		disp = str('Expected [', _str(object), '] to extend [', a_str, ']: ', text)
		if(!GutUtils.is_native_class(a_class) and !GutUtils.is_gdscript(a_class)):
			_fail(str(bad_param_2, a_str))
		else:
			if(is_instance_of(object, a_class)):
				_pass(disp)
			else:
				_fail(disp)


## Asserts that [param object] is the the [param type] specified.  [param type]
## should be one of the Godot [code]TYPE_[/code] constants.
## [codeblock]
##    # Passing
##    var c = Color(1, 1, 1, 1)
##    gr.test.assert_typeof(c, TYPE_COLOR)
##    assert_pass(gr.test)
##
##    # Failing
##    gr.test.assert_typeof('some string', TYPE_INT)
##    assert_fail(gr.test)
## [/codeblock]
func assert_typeof(object, type, text=''):
	var disp = str('Expected [typeof(', object, ') = ')
	disp += _get_typeof_string(typeof(object))
	disp += '] to equal ['
	disp += _get_typeof_string(type) +  ']'
	disp += '.  ' + text
	if(typeof(object) == type):
		_pass(disp)
	else:
		_fail(disp)


## The inverse of [method assert_typeof]
func assert_not_typeof(object, type, text=''):
	var disp = str('Expected [typeof(', object, ') = ')
	disp += _get_typeof_string(typeof(object))
	disp += '] to not equal ['
	disp += _get_typeof_string(type) +  ']'
	disp += '.  ' + text
	if(typeof(object) != type):
		_pass(disp)
	else:
		_fail(disp)


## Assert that `text` contains `search`.  Can perform case insensitive search
## by passing false for `match_case`.
## [codeblock]
##    # Passing
##    assert_string_contains('abc 123', 'a')
##    assert_string_contains('abc 123', 'BC', false)
##    assert_string_contains('abc 123', '3')
##
##    # Failing
##    assert_string_contains('abc 123', 'A')
##    assert_string_contains('abc 123', 'BC')
##    assert_string_contains('abc 123', '012')
## [/codeblock]
func assert_string_contains(text, search, match_case=true):
	const empty_search = 'Expected text and search strings to be non-empty. You passed %s and %s.'
	const non_strings = 'Expected text and search to both be strings.  You passed %s and %s.'
	var disp = 'Expected \'%s\' to contain \'%s\', match_case=%s' % [text, search, match_case]
	if(typeof(text) != TYPE_STRING or typeof(search) != TYPE_STRING):
		_fail(non_strings % [_str(text), _str(search)])
	elif(text == '' or search == ''):
		_fail(empty_search % [_str(text), _str(search)])
	elif(match_case):
		if(text.find(search) == -1):
			_fail(disp)
		else:
			_pass(disp)
	else:
		if(text.to_lower().find(search.to_lower()) == -1):
			_fail(disp)
		else:
			_pass(disp)


## Assert that text starts with search.  Can perform case insensitive check
## by passing false for match_case
## [codeblock]
##    # Passing
##    assert_string_starts_with('abc 123', 'a')
##    assert_string_starts_with('abc 123', 'ABC', false)
##    assert_string_starts_with('abc 123', 'abc 123')
##
##    ## Failing
##    assert_string_starts_with('abc 123', 'z')
##    assert_string_starts_with('abc 123', 'ABC')
##    assert_string_starts_with('abc 123', 'abc 1234')
## [/codeblock]
func assert_string_starts_with(text, search, match_case=true):
	var empty_search = 'Expected text and search strings to be non-empty. You passed \'%s\' and \'%s\'.'
	var disp = 'Expected \'%s\' to start with \'%s\', match_case=%s' % [text, search, match_case]
	if(text == '' or search == ''):
		_fail(empty_search % [text, search])
	elif(match_case):
		if(text.find(search) == 0):
			_pass(disp)
		else:
			_fail(disp)
	else:
		if(text.to_lower().find(search.to_lower()) == 0):
			_pass(disp)
		else:
			_fail(disp)


## Assert that [param text] ends with [param search].  Can perform case insensitive check by passing false for [param match_case]
## [codeblock]
##    ## Passing
##    assert_string_ends_with('abc 123', '123')
##    assert_string_ends_with('abc 123', 'C 123', false)
##    assert_string_ends_with('abc 123', 'abc 123')
##
##    ## Failing
##    assert_string_ends_with('abc 123', '1234')
##    assert_string_ends_with('abc 123', 'C 123')
##    assert_string_ends_with('abc 123', 'nope')
## [/codeblock]
func assert_string_ends_with(text, search, match_case=true):
	var empty_search = 'Expected text and search strings to be non-empty. You passed \'%s\' and \'%s\'.'
	var disp = 'Expected \'%s\' to end with \'%s\', match_case=%s' % [text, search, match_case]
	var required_index = len(text) - len(search)
	if(text == '' or search == ''):
		_fail(empty_search % [text, search])
	elif(match_case):
		if(text.find(search) == required_index):
			_pass(disp)
		else:
			_fail(disp)
	else:
		if(text.to_lower().find(search.to_lower()) == required_index):
			_pass(disp)
		else:
			_fail(disp)


# ------------------------------------------------------------------------------
## Assert that a method was called on an instance of a doubled class.  If
## parameters are supplied then the params passed in when called must match.
##
## Can be called with a Callabe instead of specifying the object, method_name,
## and parameters.  The Callable's object must be a double.  Bound arguments
## will be used to match calls based on values passed to the method.
## [br]
## See also: [wiki]Doubles[/wiki], [wiki]Spies[/wiki]
## [br][br]
## [b]Examples[/b]
## [codeblock]
##    var my_double = double(Foobar).new()
##    ...
##    assert_called(my_double, 'foo')
##    assert_called(my_double.foo)
##    assert_called(my_double, 'foo', [1, 2, 3])
##    assert_called(my_double.foo.bind(1, 2, 3))
## [/codeblock]
func assert_called(inst, method_name=null, parameters=null):

	if(_fail_if_parameters_not_array(parameters)):
		return

	var converted = _convert_spy_args(inst, method_name, parameters)
	if(converted.invalid_message != 'ok'):
		fail_test(converted.invalid_message)
		return

	var disp = str('Expected [',converted.method_name,'] to have been called on ',_str(converted.object))
	if(converted.arguments != null):
		disp += str(' with parameters ', converted.arguments)

	if(_fail_if_not_double_or_does_not_have_method(converted.object, converted.method_name) == OK):
		if(gut.get_spy().was_called(
			converted.object, converted.method_name, converted.arguments)):
			_pass(disp)
		else:
			_fail(str(disp, "\n", _get_desc_of_calls_to_instance(converted.object)))


# ------------------------------------------------------------------------------
## Assert that a method was not called on an instance of a doubled class.  If
## parameters are specified then this will only fail if it finds a call that was
## sent matching parameters.
##
## Can be called with a Callabe instead of specifying the object, method_name,
## and parameters.  The Callable's object must be a double.  Bound arguments
## will be used to match calls based on values passed to the method.
## [br]
## See also: [wiki]Doubles[/wiki], [wiki]Spies[/wiki]
## [br][br]
## [b]Examples[/b]
## [codeblock]
##    assert_not_called(my_double, 'foo')
##    assert_not_called(my_double.foo)
##    assert_not_called(my_double, 'foo', [1, 2, 3])
##    assert_not_called(my_double.foo.bind(1, 2, 3))
## [/codeblock]
func assert_not_called(inst, method_name=null, parameters=null):

	if(_fail_if_parameters_not_array(parameters)):
		return

	var converted = _convert_spy_args(inst, method_name, parameters)
	if(converted.invalid_message != 'ok'):
		fail_test(converted.invalid_message)
		return

	var disp = str('Expected [', converted.method_name, '] to NOT have been called on ', _str(converted.object))

	if(_fail_if_not_double_or_does_not_have_method(converted.object, converted.method_name) == OK):
		if(gut.get_spy().was_called(
			converted.object, converted.method_name, converted.arguments)):
			if(converted.arguments != null):
				disp += str(' with parameters ', converted.arguments)
			_fail(str(disp, "\n", _get_desc_of_calls_to_instance(converted.object)))
		else:
			_pass(disp)


## Asserts the the method of a double was called an expected number of times.
## If any arguments are bound to the callable then only calls with matching
## arguments will be counted.
## [br]
## See also: [wiki]Doubles[/wiki], [wiki]Spies[/wiki]
## [br][br]
## [b]Examples[/b]
## [codeblock]
##    # assert foo was called on my_double 5 times
##    assert_called_count(my_double.foo, 5)
##    # assert foo, with parameters [1,2,3], was called on my_double 4 times.
##    assert_called_count(my_double.foo.bind(1, 2, 3), 4)
## [/codeblock]
func assert_called_count(callable : Callable, expected_count : int):
	var converted = _convert_spy_args(callable, null, null)
	var count = gut.get_spy().call_count(converted.object, converted.method_name, converted.arguments)

	var param_text = ''
	if(callable.get_bound_arguments_count() > 0):
		param_text = ' with parameters ' + str(callable.get_bound_arguments())
	var disp = 'Expected [%s] on %s to be called [%s] times%s.  It was called [%s] times.'
	disp = disp % [converted.method_name, _str(converted.object), expected_count, param_text, count]


	if(_fail_if_not_double_or_does_not_have_method(converted.object, converted.method_name) == OK):
		if(count == expected_count):
			_pass(disp)
		else:
			_fail(str(disp, "\n", _get_desc_of_calls_to_instance(converted.object)))


## Asserts the passed in value is null
func assert_null(got, text=''):
	var disp = str('Expected [', _str(got), '] to be NULL:  ', text)
	if(got == null):
		_pass(disp)
	else:
		_fail(disp)


## Asserts the passed in value is not null.
func assert_not_null(got, text=''):
	var disp = str('Expected [', _str(got), '] to be anything but NULL:  ', text)
	if(got == null):
		_fail(disp)
	else:
		_pass(disp)


## Asserts that the passed in object has been freed.  This assertion requires
## that  you pass in some text in the form of a title since, if the object is
## freed, we won't have anything to convert to a string to put in the output
## statement.
## [br]
## [b]Note[/b] that this currently does not detect if a node has been queued free.
## [codeblock]
##    var obj = Node.new()
##    obj.free()
##    test.assert_freed(obj, "New Node")
## [/codeblock]
func assert_freed(obj, title='something'):
	var disp = title
	if(is_instance_valid(obj)):
		disp = _strutils.type2str(obj) + title
	assert_true(not is_instance_valid(obj), "Expected [%s] to be freed" % disp)


## The inverse of [method assert_freed]
func assert_not_freed(obj, title='something'):
	var disp = title
	if(is_instance_valid(obj)):
		disp = _strutils.type2str(obj) + title
	assert_true(is_instance_valid(obj), "Expected [%s] to not be freed" % disp)


## This method will assert that no orphaned nodes have been introduced by the
## test when the assert is executed.  See the [wiki]Memory-Management[/wiki]
## page for more information.
func assert_no_new_orphans(text=''):
	var count = gut.get_orphan_counter().get_orphans_since('test')
	var msg = ''
	if(text != ''):
		msg = ':  ' + text
	# Note that get_counter will return -1 if the counter does not exist.  This
	# can happen with a misplaced assert_no_new_orphans.  Checking for > 0
	# ensures this will not cause some weird failure.
	if(count > 0):
		_fail(str('Expected no orphans, but found ', count, msg))
	else:
		_pass('No new orphans found.' + msg)


## @ignore
func assert_set_property(obj, property_name, new_value, expected_value):
	pending("this hasn't been implemented yet")


## @ignore
func assert_readonly_property(obj, property_name, new_value, expected_value):
	pending("this hasn't been implemented yet")


## Assumes backing varible with be _<property_name>.  This will perform all the
## asserts of assert_property.  Then this will set the value through the setter
## and check the backing variable value.  It will then reset throught the setter
## and set the backing variable and check the getter.
func assert_property_with_backing_variable(obj, property_name, default_value, new_value, backed_by_name=null):
	var setter_name = str('@', property_name, '_setter')
	var getter_name = str('@', property_name, '_getter')
	var backing_name = GutUtils.nvl(backed_by_name, str('_', property_name))
	var pre_fail_count = get_fail_count()

	var props = obj.get_property_list()
	var found = false
	var idx = 0
	while(idx < props.size() and !found):
		found = props[idx].name == backing_name
		idx += 1

	assert_true(found, str(obj, ' has ', backing_name, ' variable.'))
	assert_true(obj.has_method(setter_name), str('There should be a setter for ', property_name))
	assert_true(obj.has_method(getter_name), str('There should be a getter for ', property_name))

	if(pre_fail_count == get_fail_count()):
		var call_setter = Callable(obj, setter_name)
		var call_getter = Callable(obj, getter_name)

		assert_eq(obj.get(backing_name), default_value, str('Variable ', backing_name, ' has default value.'))
		assert_eq(call_getter.call(), default_value, 'Getter returns default value.')
		call_setter.call(new_value)
		assert_eq(call_getter.call(), new_value, 'Getter returns value from Setter.')
		assert_eq(obj.get(backing_name), new_value, str('Variable ', backing_name, ' was set'))

	_warn_for_public_accessors(obj, property_name)


## This will verify that the method has a setter and getter for the property.
## It will then use the getter to check the default.  Then use the
## setter with new_value and verify the getter returns the same value.
func assert_property(obj, property_name, default_value, new_value) -> void:
	var pre_fail_count = get_fail_count()

	var setter_name = str('@', property_name, '_setter')
	var getter_name = str('@', property_name, '_getter')

	if(typeof(obj) != TYPE_OBJECT):
		_fail(str(_str(obj), ' is not an object'))
		return

	assert_has_method(obj, setter_name)
	assert_has_method(obj, getter_name)

	if(pre_fail_count == get_fail_count()):
		var call_setter = Callable(obj, setter_name)
		var call_getter = Callable(obj, getter_name)

		assert_eq(call_getter.call(), default_value, 'Default value')
		call_setter.call(new_value)
		assert_eq(call_getter.call(), new_value, 'Getter gets Setter value')

	_warn_for_public_accessors(obj, property_name)


## Performs a deep comparison between two arrays or dictionaries and asserts
## they are equal.  If they are not equal then a formatted list of differences
## are displayed.  See [wiki]Comparing-Things[/wiki] for more information.
func assert_eq_deep(v1, v2):
	var result = compare_deep(v1, v2)
	if(result.are_equal):
		_pass(result.get_short_summary())
	else:
		_fail(result.summary)


## Performs a deep comparison of two arrays or dictionaries and asserts they
## are not equal.  See [wiki]Comparing-Things[/wiki] for more information.
func assert_ne_deep(v1, v2):
	var result = compare_deep(v1, v2)
	if(!result.are_equal):
		_pass(result.get_short_summary())
	else:
		_fail(result.get_short_summary())


## Assert v1 and v2 are the same using [code]is_same[/code].  See @GlobalScope.is_same.
func assert_same(v1, v2, text=''):
	var disp = "[" + _str(v1) + "] expected to be same as  [" + _str(v2) + "]:  " + text
	if(is_same(v1, v2)):
		_pass(disp)
	else:
		_fail(disp)


## Assert using v1 and v2 are not the same using [code]is_same[/code].  See @GlobalScope.is_same.
func assert_not_same(v1, v2, text=''):
	var disp = "[" + _str(v1) + "] expected to not be same as  [" + _str(v2) + "]:  " + text
	if(is_same(v1, v2)):
		_fail(disp)
	else:
		_pass(disp)


# ----------------
#endregion
#region Await Helpers
# ----------------


## Use with await to wait an amount of time in seconds.  The optional message
## will be printed when the await starts.[br]
## See [wiki]Awaiting[/wiki]
func wait_seconds(time, msg=''):
	_lgr.yield_msg(str('-- Awaiting ', time, ' second(s) -- ', msg))
	_awaiter.wait_seconds(time)
	return _awaiter.timeout


## Use with await to wait for a signal to be emitted or a maximum amount of
## time.  Returns true if the signal was emitted, false if not.[br]
## See [wiki]Awaiting[/wiki]
func wait_for_signal(sig : Signal, max_wait, msg=''):
	watch_signals(sig.get_object())
	_lgr.yield_msg(str('-- Awaiting signal "', sig.get_name(), '" or for ', max_wait, ' second(s) -- ', msg))
	_awaiter.wait_for_signal(sig, max_wait)
	await _awaiter.timeout
	return !_awaiter.did_last_wait_timeout


## @deprecated
## Use wait_physics_frames or wait_process_frames
## See [wiki]Awaiting[/wiki]
func wait_frames(frames : int, msg=''):
	_lgr.deprecated("wait_frames has been replaced with wait_physics_frames which is counted in _physics_process.  " +
		"wait_process_frames has also been added which is counted in _process.")
	return wait_physics_frames(frames, msg)


## This returns a signal that is emitted after [param x] physics frames have
## elpased.  You can await this method directly to pause execution for [param x]
## physics frames.  The frames are counted prior to _physics_process being called
## on any node (when [signal SceneTree.physics_frame] is emitted).  This means the
## signal is emitted after [param x] frames and just before the x + 1 frame starts.
## [codeblock]
## await wait_physics_frames(10)
## [/codeblock]
## See [wiki]Awaiting[/wiki]
func wait_physics_frames(x :int , msg=''):
	if(x <= 0):
		var text = str('wait_physics_frames:  frames must be > 0, you passed  ', x, '.  1 frames waited.')
		_lgr.error(text)
		x = 1

	_lgr.yield_msg(str('-- Awaiting ', x, ' physics frame(s) -- ', msg))
	_awaiter.wait_physics_frames(x)
	return _awaiter.timeout


## Alias for [method GutTest.wait_process_frames]
func wait_idle_frames(x : int, msg=''):
	return wait_process_frames(x, msg)


## This returns a signal that is emitted after [param x] process/idle frames have
## elpased.  You can await this method directly to pause execution for [param x]
## process/idle frames.  The frames are counted prior to _process being called
## on any node (when [signal SceneTree.process_frame] is emitted).  This means the
## signal is emitted after [param x] frames and just before the x + 1 frame starts.
## [codeblock]
## await wait_process_frames(10)
## # wait_idle_frames is an alias of wait_process_frames
## await wait_idle_frames(10)
## [/codeblock]
## See [wiki]Awaiting[/wiki]
func wait_process_frames(x : int, msg=''):
	if(x <= 0):
		var text = str('wait_process_frames:  frames must be > 0, you passed  ', x, '.  1 frames waited.')
		_lgr.error(text)
		x = 1

	_lgr.yield_msg(str('-- Awaiting ', x, ' idle frame(s) -- ', msg))
	_awaiter.wait_process_frames(x)
	return _awaiter.timeout


## Use with await to wait for [param callable] to return the boolean value
## [code]true[/code] or a maximum amount of time.  All values that are not the
## boolean value [code]true[/code] are ignored.  [param callable] is called
## every [code]_physics_process[/code] tick unless an optional time between
## calls is specified.[br]
## [param p3] can be the optional message or an amount of time to wait between calls.[br]
## [param p4] is the optional message if you have specified an amount of time to
## wait between calls.[br]
## Returns [code]true[/code] if [param callable] returned true before the timeout, false if not.
##[br]
##[codeblock]
## var foo = 1
## func test_example():
##     var foo_func = func():
##         foo += 1
##         return foo == 10
##     foo = 1
##     wait_until(foo_func, 5, 'optional message')
##     # or give it a time between
##     foo = 1
##     wait_until(foo_func, 5, 1,
##         'this will timeout because we call it every second and are waiting a max of 10 seconds')
##
##[/codeblock]
## See also [method wait_while][br]
## See [wiki]Awaiting[/wiki]
func wait_until(callable, max_wait, p3='', p4=''):
	var time_between = 0.0
	var message = p4
	if(typeof(p3) != TYPE_STRING):
		time_between = p3
	else:
		message = p3

	_lgr.yield_msg(str("--Awaiting callable to return TRUE or ", max_wait, "s.  ", message))
	_awaiter.wait_until(callable, max_wait, time_between)
	await _awaiter.timeout
	return !_awaiter.did_last_wait_timeout


## This is the inverse of [method wait_until].  This will continue to wait while
## [param callable] returns the boolean value [code]true[/code].  If [b]ANY[/b]
## other value is is returned then the wait will end.
## Returns [code]true[/code] if [param callable] returned a value other than
## [code]true[/code] before the timeout, [code]false[/code] if not.
##[codeblock]
## var foo = 1
## func test_example():
##     var foo_func = func():
##         foo += 1
##         if(foo < 10):
##             return true
##         else:
##             return 'this is not a boolean'
##     foo = 1
##     wait_while(foo_func, 5, 'optional message')
##     # or give it a time between
##     foo = 1
##     wait_while(foo_func, 5, 1,
##         'this will timeout because we call it every second and are waiting a max of 10 seconds')
##
##[/codeblock]
## See [wiki]Awaiting[/wiki]
func wait_while(callable, max_wait, p3='', p4=''):
	var time_between = 0.0
	var message = p4
	if(typeof(p3) != TYPE_STRING):
		time_between = p3
	else:
		message = p3

	_lgr.yield_msg(str("--Awaiting callable to return FALSE or ", max_wait, "s.  ", message))
	_awaiter.wait_while(callable, max_wait, time_between)
	await _awaiter.timeout
	return !_awaiter.did_last_wait_timeout



## Returns whether the last wait_* method timed out.  This is always true if
## the last method was wait_xxx_frames or wait_seconds.  It will be false when
## using wait_for_signal and wait_until if the timeout occurs before what
## is being waited on.  The wait_* methods return this value so you should be
## able to avoid calling this directly, but you can.
func did_wait_timeout():
	return _awaiter.did_last_wait_timeout

# ----------------
#endregion
#region Summary Data
# ----------------

## @internal
func get_summary():
	return _summary


## Returns the number of failing asserts in this script at the time this
## method was called.  Call in [method after_all] to get total count for script.
func get_fail_count():
	return _summary.failed


## Returns the number of passing asserts in this script at the time this method
## was called.  Call in [method after_all] to get total count for script.
func get_pass_count():
	return _summary.passed


## Returns the number of pending tests in this script at the time this method
## was called.  Call in [method after_all] to get total count for script.
func get_pending_count():
	return _summary.pending


## Returns the total number of asserts this script has made as of the time of
## this was called.  Call in [method after_all] to get total count for script.
func get_assert_count():
	return _summary.asserts


# Convert the _summary dictionary into text
## @internal
func get_summary_text():
	var to_return = get_script().get_path() + "\n"
	to_return += str('  ', _summary.passed, ' of ', _summary.asserts, ' passed.')
	if(_summary.pending > 0):
		to_return += str("\n  ", _summary.pending, ' pending')
	if(_summary.failed > 0):
		to_return += str("\n  ", _summary.failed, ' failed.')
	return to_return


# ----------------
#endregion
#region Double Methods
# ----------------


## Create a Double of [param thing].  [param thing] should be a Class, script,
## or scene.  See [wiki]Doubles[/wiki]
func double(thing, double_strat=null, not_used_anymore=null):
	if(!_are_double_parameters_valid(thing, double_strat, not_used_anymore)):
		return null

	return _smart_double(thing, double_strat, false)


## Create a Partial Double of [param thing].  [param thing] should be a Class,
## script, or scene.  See [wiki]Partial-Doubles[/wiki]
func partial_double(thing, double_strat=null, not_used_anymore=null):
	if(!_are_double_parameters_valid(thing, double_strat, not_used_anymore)):
		return null

	return _smart_double(thing, double_strat, true)


## @internal
func double_singleton(singleton_name):
	return null
	# var to_return = null
	# if(_validate_singleton_name(singleton_name)):
	# 	to_return = gut.get_doubler().double_singleton(singleton_name)
	# return to_return


## @internal
func partial_double_singleton(singleton_name):
	return null
	# var to_return = null
	# if(_validate_singleton_name(singleton_name)):
	# 	to_return = gut.get_doubler().partial_double_singleton(singleton_name)
	# return to_return


## This was implemented to allow the doubling of classes with static methods.
## There might be other valid use cases for this method, but you should always
## try stubbing before using this method.  Using
## [code]stub(my_double, 'method').to_call_super()[/code] or  creating a
## [method partial_double] works for any other known scenario.  You cannot stub
## or spy on methods passed to [code skip-lint]ignore_method_when_doubling[/code].
func ignore_method_when_doubling(thing, method_name):
	if(typeof(thing) == TYPE_STRING):
		_lgr.error('ignore_method_when_doubling no longer supports paths to scripts or scenes.  Load them and pass them instead.')
		return

	var r = thing
	if(thing is PackedScene):
		r = GutUtils.get_scene_script_object(thing)

	gut.get_doubler().add_ignored_method(r, method_name)


## Stub something.  See [wiki]Stubbing[/wiki] for detailed information about stubbing.
func stub(thing, p2=null, p3=null):
	var method_name = p2
	var subpath = null

	if(p3 != null):
		subpath = p2
		method_name = p3

	if(GutUtils.is_instance(thing)):
		var msg = _get_bad_double_or_method_message(thing, method_name, 'stub')
		if(msg != ''):
			_lgr.error(msg)
			return GutUtils.StubParams.new()

	var sp = null
	if(typeof(thing) == TYPE_CALLABLE):
		if(p2 != null or p3 != null):
			_lgr.error("Only one parameter expected when using a callable.")
		sp = GutUtils.StubParams.new(thing)
	else:
		sp = GutUtils.StubParams.new(thing, method_name, subpath)

	sp.logger = _lgr
	gut.get_stubber().add_stub(sp)
	return sp

# ----------------
#endregion
#region Memory Mgmt
# ----------------


## Marks whatever is passed in to be freed after the test finishes.  It also
## returns what is passed in so you can save a line of code.
##   var thing = autofree(Thing.new())
func autofree(thing):
	gut.get_autofree().add_free(thing)
	return thing


## Works the same as autofree except queue_free will be called on the object
## instead.  This also imparts a brief pause after the test finishes so that
## the queued object has time to free.
func autoqfree(thing):
	gut.get_autofree().add_queue_free(thing)
	return thing


## The same as autofree but it also adds the object as a child of the test.
func add_child_autofree(node, legible_unique_name = false):
	gut.get_autofree().add_free(node)
	# Explicitly calling super here b/c add_child MIGHT change and I don't want
	# a bug sneaking its way in here.
	super.add_child(node, legible_unique_name)
	return node


## The same as autoqfree but it also adds the object as a child of the test.
func add_child_autoqfree(node, legible_unique_name=false):
	gut.get_autofree().add_queue_free(node)
	# Explicitly calling super here b/c add_child MIGHT change and I don't want
	# a bug sneaking its way in here.
	super.add_child(node, legible_unique_name)
	return node



# ----------------
#endregion
#region Deprecated/Removed
# ----------------


## REMOVED
## @ignore
func compare_shallow(v1, v2, max_differences=null):
	_fail('compare_shallow has been removed.  Use compare_deep or just compare using == instead.')
	_lgr.error('compare_shallow has been removed.  Use compare_deep or just compare using == instead.')
	return null


## REMOVED
## @ignore
func assert_eq_shallow(v1, v2):
	_fail('assert_eq_shallow has been removed.  Use assert_eq/assert_same/assert_eq_deep')


## REMOVED
## @ignore
func assert_ne_shallow(v1, v2):
	_fail('assert_eq_shallow has been removed.  Use assert_eq/assert_same/assert_eq_deep')


## @deprecated: use wait_seconds
func yield_for(time, msg=''):
	_lgr.deprecated('yield_for', 'wait_seconds')
	return wait_seconds(time, msg)


## @deprecated: use wait_for_signal
func yield_to(obj, signal_name, max_wait, msg=''):
	_lgr.deprecated('yield_to', 'wait_for_signal')
	return await wait_for_signal(Signal(obj, signal_name), max_wait, msg)


## @deprecated: use wait_frames
func yield_frames(frames, msg=''):
	_lgr.deprecated("yield_frames", "wait_frames")
	return wait_frames(frames, msg)


## @deprecated: no longer supported.  Use double
func double_scene(path, strategy=null):
	_lgr.deprecated('test.double_scene has been removed.', 'double')
	return null


## @deprecated: no longer supported.  Use double
func double_script(path, strategy=null):
	_lgr.deprecated('test.double_script has been removed.', 'double')
	return null

	# var override_strat = GutUtils.nvl(strategy, gut.get_doubler().get_strategy())
	# return gut.get_doubler().double(path, override_strat)


## @deprecated: no longer supported.  Use register_inner_classes + double
func double_inner(path, subpath, strategy=null):
	_lgr.deprecated('double_inner should not be used.  Use register_inner_classes and double instead.', 'double')
	return null

	var override_strat = GutUtils.nvl(strategy, gut.get_doubler().get_strategy())
	return gut.get_doubler().double_inner(path, subpath, override_strat)


## @deprecated:  Use [method assert_called_count] instead.
func assert_call_count(inst, method_name, expected_count, parameters=null):
	gut.logger.deprecated('This has been replaced with assert_called_count which accepts a Callable with optional bound arguments.')
	var callable = Callable.create(inst, method_name)
	if(parameters != null):
		callable = callable.bindv(parameters)
	assert_called_count(callable, expected_count)


## @deprecated: no longer supported.
func assert_setget(
	instance, name_property,
	const_or_setter = null, getter="__not_set__"):
	_lgr.deprecated('assert_property')
	_fail('assert_setget has been removed.  Use assert_property, assert_set_property, assert_readonly_property instead.')


# ----------------
#endregion



# ##############################################################################
#(G)odot (U)nit (T)est class
#
# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
# View readme for usage details.
#
# Version - see gut.gd
# ##############################################################################
# Class that all test scripts must extend.`
#
# This provides all the asserts and other testing features.  Test scripts are
# run by the Gut class in gut.gd
# ##############################################################################
