extends Node

var Optparse = load('res://addons/gut/cli/optparse.gd')
var Gut = load('res://addons/gut/gut.gd')
var GutRunner = load('res://addons/gut/gui/GutRunner.tscn')

# ------------------------------------------------------------------------------
# Helper class to resolve the various different places where an option can
# be set.  Using the get_value method will enforce the order of precedence of:
# 	1.  command line value
#	2.  config file value
#	3.  default value
#
# The idea is that you set the base_opts.  That will get you a copies of the
# hash with null values for the other types of values.  Lower precedented hashes
# will punch through null values of higher precedented hashes.
# ------------------------------------------------------------------------------
class OptionResolver:
	var base_opts = {}
	var cmd_opts = {}
	var config_opts = {}


	func get_value(key):
		return _nvl(cmd_opts[key], _nvl(config_opts[key], base_opts[key]))

	func set_base_opts(opts):
		base_opts = opts
		cmd_opts = _null_copy(opts)
		config_opts = _null_copy(opts)

	# creates a copy of a hash with all values null.
	func _null_copy(h):
		var new_hash = {}
		for key in h:
			new_hash[key] = null
		return new_hash

	func _nvl(a, b):
		if(a == null):
			return b
		else:
			return a

	func _string_it(h):
		var to_return = ''
		for key in h:
			to_return += str('(',key, ':', _nvl(h[key], 'NULL'), ')')
		return to_return

	func to_s():
		return str("base:\n", _string_it(base_opts), "\n", \
				"config:\n", _string_it(config_opts), "\n", \
				"cmd:\n", _string_it(cmd_opts), "\n", \
				"resolved:\n", _string_it(get_resolved_values()))

	func get_resolved_values():
		var to_return = {}
		for key in base_opts:
			to_return[key] = get_value(key)
		return to_return

	func to_s_verbose():
		var to_return = ''
		var resolved = get_resolved_values()
		for key in base_opts:
			to_return += str(key, "\n")
			to_return += str('  default: ', _nvl(base_opts[key], 'NULL'), "\n")
			to_return += str('  config:  ', _nvl(config_opts[key], ' --'), "\n")
			to_return += str('  cmd:     ', _nvl(cmd_opts[key], ' --'), "\n")
			to_return += str('  final:   ', _nvl(resolved[key], 'NULL'), "\n")

		return to_return

# ------------------------------------------------------------------------------
# Here starts the actual script that uses the Options class to kick off Gut
# and run your tests.
# ------------------------------------------------------------------------------
var _gut_config = load('res://addons/gut/gut_config.gd').new()

# array of command line options specified
var _final_opts = []


func setup_options(options, font_names):
	var opts = Optparse.new()
	opts.banner =\
"""
The GUT CLI
-----------
The default behavior for GUT is to load options from a res://.gutconfig.json if
it exists.  Any options specified on the command line will take precedence over
options specified in the gutconfig file.  You can specify a different gutconfig
file with the -gconfig option.

To generate a .gutconfig.json file you can use -gprint_gutconfig_sample
To see the effective values of a CLI command and a gutconfig use -gpo

Values for options can be supplied using:
    option=value    # no space around "="
    option value    # a space between option and value w/o =

Options whose values are lists/arrays can be specified multiple times:
	-gdir=a,b
	-gdir c,d
	-gdir e
	# results in -gdir equaling [a, b, c, d, e]
"""
	opts.add_heading("Test Config:")
	opts.add('-gdir', options.dirs, 'List of directories to search for test scripts in.')
	opts.add('-ginclude_subdirs', false, 'Flag to include all subdirectories specified with -gdir.')
	opts.add('-gtest', [], 'List of full paths to test scripts to run.')
	opts.add('-gprefix', options.prefix, 'Prefix used to find tests when specifying -gdir.  Default "[default]".')
	opts.add('-gsuffix', options.suffix, 'Test script suffix, including .gd extension.  Default "[default]".')
	opts.add('-gconfig', 'res://.gutconfig.json', 'The config file to load options from.  The default is [default].  Use "-gconfig=" to not use a config file.')
	opts.add('-gpre_run_script', '', 'pre-run hook script path')
	opts.add('-gpost_run_script', '', 'post-run hook script path')
	opts.add('-gerrors_do_not_cause_failure', false, 'When an internal GUT error occurs tests will fail.  With this option set, that does not happen.')
	opts.add('-gdouble_strategy', 'SCRIPT_ONLY', 'Default strategy to use when doubling.  Valid values are [INCLUDE_NATIVE, SCRIPT_ONLY].  Default "[default]"')

	opts.add_heading("Run Options:")
	opts.add('-gselect', '', 'All scripts that contain the specified string in their filename will be ran')
	opts.add('-ginner_class', '', 'Only run inner classes that contain the specified string in their name.')
	opts.add('-gunit_test_name', '', 'Any test that contains the specified text will be run, all others will be skipped.')
	opts.add('-gexit', false, 'Exit after running tests.  If not specified you have to manually close the window.')
	opts.add('-gexit_on_success', false, 'Only exit if zero tests fail.')
	opts.add('-gignore_pause', false, 'Ignores any calls to pause_before_teardown.')

	opts.add_heading("Display Settings:")
	opts.add('-glog', options.log_level, 'Log level [0-3].  Default [default]')
	opts.add('-ghide_orphans', false, 'Display orphan counts for tests and scripts.  Default [default].')
	opts.add('-gmaximize', false, 'Maximizes test runner window to fit the viewport.')
	opts.add('-gcompact_mode', false, 'The runner will be in compact mode.  This overrides -gmaximize.')
	opts.add('-gopacity', options.opacity, 'Set opacity of test runner window. Use range 0 - 100. 0 = transparent, 100 = opaque.')
	opts.add('-gdisable_colors', false, 'Disable command line colors.')
	opts.add('-gfont_name', options.font_name, str('Valid values are:  ', font_names, '.  Default "[default]"'))
	opts.add('-gfont_size', options.font_size, 'Font size, default "[default]"')
	opts.add('-gbackground_color', options.background_color, 'Background color as an html color, default "[default]"')
	opts.add('-gfont_color',options.font_color, 'Font color as an html color, default "[default]"')
	opts.add('-gpaint_after', options.paint_after, 'Delay before GUT will add a 1 frame pause to paint the screen/GUI.  default [default]')

	opts.add_heading("Result Export:")
	opts.add('-gjunit_xml_file', options.junit_xml_file, 'Export results of run to this file in the Junit XML format.')
	opts.add('-gjunit_xml_timestamp', options.junit_xml_timestamp, 'Include a timestamp in the -gjunit_xml_file, default [default]')

	opts.add_heading("Help:")
	opts.add('-gh', false, 'Print this help.  You did this to see this, so you probably understand.')
	opts.add('-gpo', false, 'Print option values from all sources and the value used.')
	opts.add('-gprint_gutconfig_sample', false, 'Print out json that can be used to make a gutconfig file.')

	return opts


# Parses options, applying them to the _tester or setting values
# in the options struct.
func extract_command_line_options(from, to):
	to.config_file = from.get_value_or_null('-gconfig')
	to.dirs = from.get_value_or_null('-gdir')
	to.disable_colors =  from.get_value_or_null('-gdisable_colors')
	to.double_strategy = from.get_value_or_null('-gdouble_strategy')
	to.ignore_pause = from.get_value_or_null('-gignore_pause')
	to.include_subdirs = from.get_value_or_null('-ginclude_subdirs')
	to.inner_class = from.get_value_or_null('-ginner_class')
	to.log_level = from.get_value_or_null('-glog')
	to.opacity = from.get_value_or_null('-gopacity')
	to.post_run_script = from.get_value_or_null('-gpost_run_script')
	to.pre_run_script = from.get_value_or_null('-gpre_run_script')
	to.prefix = from.get_value_or_null('-gprefix')
	to.selected = from.get_value_or_null('-gselect')
	to.should_exit = from.get_value_or_null('-gexit')
	to.should_exit_on_success = from.get_value_or_null('-gexit_on_success')
	to.should_maximize = from.get_value_or_null('-gmaximize')
	to.compact_mode = from.get_value_or_null('-gcompact_mode')
	to.hide_orphans = from.get_value_or_null('-ghide_orphans')
	to.suffix = from.get_value_or_null('-gsuffix')
	to.errors_do_not_cause_failure = from.get_value_or_null('-gerrors_do_not_cause_failure')
	to.tests = from.get_value_or_null('-gtest')
	to.unit_test_name = from.get_value_or_null('-gunit_test_name')

	to.font_size = from.get_value_or_null('-gfont_size')
	to.font_name = from.get_value_or_null('-gfont_name')
	to.background_color = from.get_value_or_null('-gbackground_color')
	to.font_color = from.get_value_or_null('-gfont_color')
	to.paint_after = from.get_value_or_null('-gpaint_after')

	to.junit_xml_file = from.get_value_or_null('-gjunit_xml_file')
	to.junit_xml_timestamp = from.get_value_or_null('-gjunit_xml_timestamp')



func _print_gutconfigs(values):
	var header = """Here is a sample of a full .gutconfig.json file.
You do not need to specify all values in your own file.  The values supplied in
this sample are what would be used if you ran gut w/o the -gprint_gutconfig_sample
option.   Option priority is:  command-line, .gutconfig, default)."""
	print("\n", header.replace("\n", ' '), "\n")
	var resolved = values

	# remove_at some options that don't make sense to be in config
	resolved.erase("config_file")
	resolved.erase("show_help")

	print(JSON.stringify(resolved, '  '))

	for key in resolved:
		resolved[key] = null

	print("\n\nAnd here's an empty config for you fill in what you want.")
	print(JSON.stringify(resolved, ' '))


func _run_tests(opt_resolver):
	_final_opts = opt_resolver.get_resolved_values();
	_gut_config.options = _final_opts

	var runner = GutRunner.instantiate()
	runner.set_gut_config(_gut_config)
	get_tree().root.add_child(runner)

	runner.run_tests()


# parse options and run Gut
func main():
	var opt_resolver = OptionResolver.new()
	opt_resolver.set_base_opts(_gut_config.default_options)

	var cli_opts = setup_options(_gut_config.default_options, _gut_config.valid_fonts)

	cli_opts.parse()
	var all_options_valid = cli_opts.unused.size() == 0
	extract_command_line_options(cli_opts, opt_resolver.cmd_opts)

	var config_path = opt_resolver.get_value('config_file')
	var load_result = 1
	# Checking for an empty config path allows us to not use a config file via
	# the -gconfig_file option since using "-gconfig_file=" or -gconfig_file=''"
	# will result in an empty string.
	if(config_path != ''):
		load_result = _gut_config.load_options_no_defaults(config_path)

	# SHORTCIRCUIT
	if(!all_options_valid):
		print('Unknown arguments:  ', cli_opts.unused)
		get_tree().quit(1)
	elif(load_result == -1):
		print('Invalid gutconfig ', load_result)
		get_tree().quit(1)
	else:
		opt_resolver.config_opts = _gut_config.options

		if(cli_opts.get_value('-gh')):
			print(GutUtils.version_numbers.get_version_text())
			cli_opts.print_help()
			get_tree().quit(0)
		elif(cli_opts.get_value('-gpo')):
			print('All config options and where they are specified.  ' +
				'The "final" value shows which value will actually be used ' +
				'based on order of precedence (default < .gutconfig < cmd line).' + "\n")
			print(opt_resolver.to_s_verbose())
			get_tree().quit(0)
		elif(cli_opts.get_value('-gprint_gutconfig_sample')):
			_print_gutconfigs(opt_resolver.get_resolved_values())
			get_tree().quit(0)
		else:
			_run_tests(opt_resolver)



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
