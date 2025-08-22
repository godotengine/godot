# ------------------------------------------------------------------------------
# This holds all the meta information for a test script.  It contains the
# name of the inner class and an array of CollectedTests.  This does not parse
# anything, it just holds the data about parsed scripts and tests.  The
# TestCollector is responsible for populating this object.
#
# This class also facilitates all the exporting and importing of tests.
# ------------------------------------------------------------------------------
var CollectedTest = GutUtils.CollectedTest

var _lgr = null

# One entry per test found in the script.  Added externally by TestCollector
var tests = []
# One entry for before_all and after_all (maybe add before_each and after_each).
# These are added by Gut when running before_all and after_all for the script.
var setup_teardown_tests = []
var inner_class_name:StringName
var path:String


# Set externally by test_collector after it can verify that the script was
# actually loaded.  This could probably be changed to just hold the GutTest
# script that was loaded, cutting down on complexity elsewhere.
var is_loaded = false

# Set by Gut when it decides that a script should be skipped.
# Right now this is whenever the script has the variable skip_script declared.
# the value of skip_script is put into skip_reason.
var was_skipped = false
var skip_reason = ''
var was_run = false


var name = '' :
	get: return path
	set(val):pass


func _init(logger=null):
	_lgr = logger


func get_new():
	return load_script().new()


func load_script():
	var to_return = load(path)

	if(inner_class_name != null and inner_class_name != ''):
		# If we wanted to do inner classes in inner classses
		# then this would have to become some kind of loop or recursive
		# call to go all the way down the chain or this class would
		# have to change to hold onto the loaded class instead of
		# just path information.
		to_return = to_return.get(inner_class_name)

	return to_return

# script.gd.InnerClass
func get_filename_and_inner():
	var to_return = get_filename()
	if(inner_class_name != ''):
		to_return += '.' + String(inner_class_name)
	return to_return


# res://foo/bar.gd.FooBar
func get_full_name():
	var to_return = path
	if(inner_class_name != ''):
		to_return += '.' + String(inner_class_name)
	return to_return


func get_filename():
	return path.get_file()


func has_inner_class():
	return inner_class_name != ''


# Note:  although this no longer needs to export the inner_class names since
#        they are pulled from metadata now, it is easier to leave that in
#        so we don't have to cut the export down to unique script names.
func export_to(config_file, section):
	config_file.set_value(section, 'path', path)
	config_file.set_value(section, 'inner_class', inner_class_name)
	var names = []
	for i in range(tests.size()):
		names.append(tests[i].name)
	config_file.set_value(section, 'tests', names)


func _remap_path(source_path):
	var to_return = source_path
	if(!FileAccess.file_exists(source_path)):
		_lgr.debug('Checking for remap for:  ' + source_path)
		var remap_path = source_path.get_basename() + '.gd.remap'
		if(FileAccess.file_exists(remap_path)):
			var cf = ConfigFile.new()
			cf.load(remap_path)
			to_return = cf.get_value('remap', 'path')
		else:
			_lgr.warn('Could not find remap file ' + remap_path)
	return to_return


func import_from(config_file, section):
	path = config_file.get_value(section, 'path')
	path = _remap_path(path)
	# Null is an acceptable value, but you can't pass null as a default to
	# get_value since it thinks you didn't send a default...then it spits
	# out red text.  This works around that.
	var inner_name = config_file.get_value(section, 'inner_class', 'Placeholder')
	if(inner_name != 'Placeholder'):
		inner_class_name = inner_name
	else: # just being explicit
		inner_class_name = StringName("")


func get_test_named(test_name):
	return GutUtils.search_array(tests, 'name', test_name)


func get_ran_test_count():
	var count = 0
	for t in tests:
		if(t.was_run):
			count += 1
	return count


func get_assert_count():
	var count = 0
	for t in tests:
		count += t.pass_texts.size()
		count += t.fail_texts.size()
	for t in setup_teardown_tests:
		count += t.pass_texts.size()
		count += t.fail_texts.size()
	return count


func get_pass_count():
	var count = 0
	for t in tests:
		count += t.pass_texts.size()
	for t in setup_teardown_tests:
		count += t.pass_texts.size()
	return count


func get_fail_count():
	var count = 0
	for t in tests:
		count += t.fail_texts.size()
	for t in setup_teardown_tests:
		count += t.fail_texts.size()
	return count


func get_pending_count():
	var count = 0
	for t in tests:
		count += t.pending_texts.size()
	return count


func get_passing_test_count():
	var count = 0
	for t in tests:
		if(t.is_passing()):
			count += 1
	return count


func get_failing_test_count():
	var count = 0
	for t in tests:
		if(t.is_failing()):
			count += 1
	return count


func get_risky_count():
	var count = 0
	if(was_skipped):
		count = 1
	else:
		for t in tests:
			if(t.is_risky()):
				count += 1
	return count


func to_s():
	var to_return = path
	if(inner_class_name != null):
		to_return += str('.', inner_class_name)
	to_return += "\n"
	for i in range(tests.size()):
		to_return += str('  ', tests[i].to_s())
	return to_return
