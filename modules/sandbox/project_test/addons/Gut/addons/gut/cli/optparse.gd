## Parses command line arguments, as one might expect.
##
## Parses command line arguments with a bunch of options including generating
## text that displays all the arguments your script accepts.  This
## is included in the GUT ClassRef since it might be usable by others and is
## portable (everything it needs is in this one file).
## [br]
## This does alot, if you want to see it in action have a look at
##	[url=https://github.com/bitwes/Gut/blob/main/scratch/optparse_example.gd]scratch/optparse_example.gd[/url]
## [codeblock lang=text]
##
## Godot Argument Lists
## -------------------------
## There are two sets of command line arguments that Godot populates:
##	OS.get_cmdline_args
##	OS.get_cmdline_user_args.
##
## OS.get_cmdline_args contains any arguments that are not used by the engine
## itself.  This means options like --help and -d will never appear in this list
## since these are used by the engine.  The one exception is the -s option which
## is always included as the first entry and the script path as the second.
## Optparse ignores these values for argument processing but can be accessed
## with my_optparse.options.script_option.  This list does not contain any
## arguments that appear in OS.get_cmdline_user_args.
##
## OS.get_cmdline_user_args contains any arguments that appear on the command
## line AFTER " -- " or " ++ ".  This list CAN contain options that the engine
## would otherwise use, and are ignored completely by the engine.
##
## The parse method, by default, includes arguments from OS.get_cmdline_args and
## OS.get_cmdline_user_args.  You can optionally pass one of these to the parse
## method to limit which arguments are parsed.  You can also conjure up your own
## array of arguments and pass that to parse.
##
## See Godot's documentation for get_cmdline_args and get_cmdline_user_args for
## more information.
##
##
## Adding Options
## --------------
## Use the following to add options to be parsed.  These methods return the
## created Option instance.  See that class above for more info.  You can use
## the returned instance to get values, or use get_value/get_value_or_null.
##   add("--name", "default", "Description goes here")
##   add(["--name", "--aliases"], "default", "Description goes here")
##   add_required(["--name", "--aliases"], "default", "Description goes here")
##   add_positional("--name", "default", "Description goes here")
##   add_positional_required("--name", "default", "Description goes here")
##
## get_value will return the value of the option or the default if it was not
## set.  get_value_or_null will return the value of the option or null if it was
## not set.
##
## The Datatype for an option is determined from the default value supplied to
## the various add methods.  Supported types are
##   String
##   Int
##   Float
##   Array of strings
##   Boolean
##
##
## Value Parsing
## -------------
## optparse uses option_name_prefix to differentiate between option names and
## values.  Any argument that starts with this value will be treated as an
## argument name.  The default is "-".  Set this before calling parse if you want
## to change it.
##
## Values for options can be supplied on the command line with or without an "=":
##	option=value    # no space around "="
##	option value    # a space between option and value w/o =
## There is no way to escape "=" at this time.
##
## Array options can be specified multiple times and/or set from a comma delimited
## list.
##   -gdir=a,b
##   -gdir c,d
##   -gdir e
## Results in -gdir equaling [a, b, c, d, e].  There is no way to escape commas
## at this time.
##
## To specify an empty list via the command line follow the option with an equal
## sign
##   -gdir=
##
## Boolean options will have thier value set to !default when they are supplied
## on the command line.  Boolean options cannot have a value on the command line.
## They are either supplied or not.
##
## If a value is not an array and is specified multiple times on the command line
## then the last entry will be used as the value.
##
## Positional argument values are parsed after all named arguments are parsed.
## This means that other options can appear before, between, and after positional
## arguments.
##   --foo=bar positional_0_value --disabled --bar foo positional_1_value --a_flag
##
## Anything that is not used by named or positional arguments will appear in the
## unused property.  You can use this to detect unrecognized arguments or treat
## everything else provided as a list of things, or whatever you want.  You can
## use is_option on the elements of unused (or whatever you want really) to see
## if optparse would treat it as an option name.
##
## Use get_missing_required_options to get an array of Option with all required
## options that were not found when parsing.
##
## The parsed_args property holds the list of arguments that were parsed.
##
##
## Help Generation
## ---------------
## You can call get_help to generate help text, or you can just call print_help
## and this will print it for you.
##
## Set the banner property to any text you want to appear before the usage and
## options sections.
##
## Options are printed in the order they are added.  You can add a heading for
## different options sections with add_heading.
##   add("--asdf", 1, "This will have no heading")
##   add_heading("foo")
##   add("--foo", false, "This will have the foo heading")
##   add("--another_foo", 1.5, "This too.")
##   add_heading("This is after foo")
##   add("--bar", true, "You probably get it by now.")
##
## If you include "[default]" in the description of a option, then the help will
## substitue it with the default value.
## [/codeblock]


#-------------------------------------------------------------------------------
# Holds all the properties of a command line option
#
# value will return the default when it has not been set.
#-------------------------------------------------------------------------------
class Option:
	var _has_been_set = false
	var _value = null
	# REMEMBER that when this option is an array, you have to set the value
	# before you alter the contents of the array (append etc) or has_been_set
	# will return false and it might not be used right.  For example
	# get_value_or_null will return null when you've actually changed the value.
	var value = _value:
		get:
			return _value

		set(val):
			_has_been_set = true
			_value = val

	var option_name = ''
	var default = null
	var description = ''
	var required = false
	var aliases: Array[String] = []


	func _init(name,default_value,desc=''):
		option_name = name
		default = default_value
		description = desc
		_value = default


	func to_s(min_space=0):
		var line_indent = str("\n", " ".repeat(min_space + 1))
		var subbed_desc = description
		if not aliases.is_empty():
			subbed_desc += "\naliases: " + ", ".join(aliases)
		subbed_desc = subbed_desc.replace('[default]', str(default))
		subbed_desc = subbed_desc.replace("\n", line_indent)
		return str(option_name.rpad(min_space), ' ', subbed_desc)


	func has_been_set():
		return _has_been_set




#-------------------------------------------------------------------------------
# A struct for organizing options by a heading
#-------------------------------------------------------------------------------
class OptionHeading:
	var options = []
	var display = 'default'




#-------------------------------------------------------------------------------
# Organizes options by order, heading, position.  Also responsible for all
# help related text generation.
#-------------------------------------------------------------------------------
class Options:
	var options = []
	var positional = []
	var default_heading = OptionHeading.new()
	var script_option = Option.new('-s', '?', 'script option provided by Godot')

	var _options_by_name = {"--script": script_option, "-s": script_option}
	var _options_by_heading = [default_heading]
	var _cur_heading = default_heading


	func add_heading(display):
		var heading = OptionHeading.new()
		heading.display = display
		_cur_heading = heading
		_options_by_heading.append(heading)


	func add(option, aliases=null):
		options.append(option)
		_options_by_name[option.option_name] = option
		_cur_heading.options.append(option)

		if aliases != null:
			for a in aliases:
				_options_by_name[a] = option
			option.aliases.assign(aliases)


	func add_positional(option):
		positional.append(option)
		_options_by_name[option.option_name] = option


	func get_by_name(option_name):
		var found_param = null
		if(_options_by_name.has(option_name)):
			found_param = _options_by_name[option_name]

		return found_param


	func get_help_text():
		var longest = 0
		var text = ""
		for i in range(options.size()):
			if(options[i].option_name.length() > longest):
				longest = options[i].option_name.length()

		for heading in _options_by_heading:
			if(heading != default_heading):
				text += str("\n", heading.display, "\n")
			for option in heading.options:
				text += str('  ', option.to_s(longest + 2).replace("\n", "\n  "), "\n")

		return text


	func get_option_value_text():
		var text = ""
		var i = 0
		for option in positional:
			text += str(i, '.  ', option.option_name, ' = ', option.value)

			if(!option.has_been_set()):
				text += " (default)"
			text += "\n"
			i += 1

		for option in options:
			text += str(option.option_name, ' = ', option.value)

			if(!option.has_been_set()):
				text += " (default)"
			text += "\n"
		return text


	func print_option_values():
		print(get_option_value_text())


	func get_missing_required_options():
		var to_return = []
		for opt in options:
			if(opt.required and !opt.has_been_set()):
				to_return.append(opt)

		for opt in positional:
			if(opt.required and !opt.has_been_set()):
				to_return.append(opt)

		return to_return


	func get_usage_text():
		var pos_text = ""
		for opt in positional:
			pos_text += str("[", opt.description, "] ")

		if(pos_text != ""):
			pos_text += " [opts] "

		return "<path to godot> -s " + script_option.value + " [opts] " + pos_text




#-------------------------------------------------------------------------------
#
# optarse
#
#-------------------------------------------------------------------------------
## @ignore
var options := Options.new()
## Set the banner property to any text you want to appear before the usage and
## options sections when printing the options help.
var banner := ''
## optparse uses option_name_prefix to differentiate between option names and
## values.  Any argument that starts with this value will be treated as an
## argument name.  The default is "-".  Set this before calling parse if you want
## to change it.
var option_name_prefix := '-'
## @ignore
var unused = []
## @ignore
var parsed_args = []
## @ignore
var values: Dictionary = {}


func _populate_values_dictionary():
	for entry in options.options:
		var value_key = entry.option_name.lstrip('-')
		values[value_key] = entry.value

	for entry in options.positional:
		var value_key = entry.option_name.lstrip('-')
		values[value_key] = entry.value


func _convert_value_to_array(raw_value):
	var split = raw_value.split(',')
	# This is what an empty set looks like from the command line.  If we do
	# not do this then we will always get back [''] which is not what it
	# shoudl be.
	if(split.size() == 1 and split[0] == ''):
		split = []
	return split

# REMEMBER raw_value not used for bools.
func _set_option_value(option, raw_value):
	var t = typeof(option.default)
	# only set values that were specified at the command line so that
	# we can punch through default and config values correctly later.
	# Without this check, you can't tell the difference between the
	# defaults and what was specified, so you can't punch through
	# higher level options.
	if(t == TYPE_INT):
		option.value = int(raw_value)
	elif(t == TYPE_STRING):
		option.value = str(raw_value)
	elif(t == TYPE_ARRAY):
		var values = _convert_value_to_array(raw_value)
		if(!option.has_been_set()):
			option.value = []
		option.value.append_array(values)
	elif(t == TYPE_BOOL):
		option.value = !option.default
	elif(t == TYPE_FLOAT):
		option.value = float(raw_value)
	elif(t == TYPE_NIL):
		print(option.option_name + ' cannot be processed, it has a nil datatype')
	else:
		print(option.option_name + ' cannot be processed, it has unknown datatype:' + str(t))


func _parse_command_line_arguments(args):
	var parsed_opts = args.duplicate()
	var i = 0
	var positional_index = 0

	while i < parsed_opts.size():
		var opt  = ''
		var value = ''
		var entry = parsed_opts[i]

		if(is_option(entry)):
			if(entry.find('=') != -1):
				var parts = entry.split('=')
				opt = parts[0]
				value = parts[1]
				var the_option = options.get_by_name(opt)
				if(the_option != null):
					parsed_opts.remove_at(i)
					_set_option_value(the_option, value)
				else:
					i += 1
			else:
				var the_option = options.get_by_name(entry)
				if(the_option != null):
					parsed_opts.remove_at(i)
					if(typeof(the_option.default) == TYPE_BOOL):
						_set_option_value(the_option, null)
					elif(i < parsed_opts.size() and !is_option(parsed_opts[i])):
						value = parsed_opts[i]
						parsed_opts.remove_at(i)
						_set_option_value(the_option, value)
				else:
					i += 1
		else:
			if(positional_index < options.positional.size()):
				_set_option_value(options.positional[positional_index], entry)
				parsed_opts.remove_at(i)
				positional_index += 1
			else:
				i += 1

	# this is the leftovers that were not extracted.
	return parsed_opts


## Test if something is an existing argument. If [code]str(arg)[/code] begins
## with the [member option_name_prefix], it will considered true,
## otherwise it will be considered false.
func is_option(arg) -> bool:
	return str(arg).begins_with(option_name_prefix)


## Adds a command line option.
## If [param op_names] is a String, this is set as the argument's name.
## If [param op_names] is an Array of Strings, all elements of the array
## will be aliases for the same argument and will be treated as such during
## parsing.
## [param default] is the default value the option will be set to if it is not
## explicitly set during parsing.
## [param desc] is a human readable text description of the option.
## If the option is successfully added, the Option object will be returned.
## If the option is not successfully added (e.g. a name collision with another
## option occurs), an error message will be printed and [code]null[/code]
## will be returned.
func add(op_names, default, desc: String) -> Option:
	var op_name: String
	var aliases: Array[String] = []
	var new_op: Option = null

	if(typeof(op_names) == TYPE_STRING):
		op_name = op_names
	else:
		op_name = op_names[0]
		aliases.assign(op_names.slice(1))

	var bad_alias: int = aliases.map(
		func (a: String) -> bool: return options.get_by_name(a) != null
	).find(true)

	if(options.get_by_name(op_name) != null):
		push_error(str('Option [', op_name, '] already exists.'))
	elif bad_alias != -1:
		push_error(str('Option [', aliases[bad_alias], '] already exists.'))
	else:
		new_op = Option.new(op_name, default, desc)
		options.add(new_op, aliases)

	return new_op


## Adds a required command line option.
## Required options that have not been set may be collected after parsing
## by calling [method get_missing_required_options].
## If [param op_names] is a String, this is set as the argument's name.
## If [param op_names] is an Array of Strings, all elements of the array
## will be aliases for the same argument and will be treated as such during
## parsing.
## [param default] is the default value the option will be set to if it is not
## explicitly set during parsing.
## [param desc] is a human readable text description of the option.
## If the option is successfully added, the Option object will be returned.
## If the option is not successfully added (e.g. a name collision with another
## option occurs), an error message will be printed and [code]null[/code]
## will be returned.
func add_required(op_names, default, desc: String) -> Option:
	var op := add(op_names, default, desc)
	if(op != null):
		op.required = true
	return op


## Adds a positional command line option.
## Positional options are parsed by their position in the list of arguments
## are are not assigned by name by the user.
## If [param op_name] is a String, this is set as the argument's name.
## If [param op_name] is an Array of Strings, all elements of the array
## will be aliases for the same argument and will be treated as such during
## parsing.
## [param default] is the default value the option will be set to if it is not
## explicitly set during parsing.
## [param desc] is a human readable text description of the option.
## If the option is successfully added, the Option object will be returned.
## If the option is not successfully added (e.g. a name collision with another
## option occurs), an error message will be printed and [code]null[/code]
## will be returned.
func add_positional(op_name, default, desc: String) -> Option:
	var new_op = null
	if(options.get_by_name(op_name) != null):
		push_error(str('Positional option [', op_name, '] already exists.'))
	else:
		new_op = Option.new(op_name, default, desc)
		options.add_positional(new_op)
	return new_op


## Adds a required positional command line option.
## If [param op_name] is a String, this is set as the argument's name.
## Required options that have not been set may be collected after parsing
## by calling [method get_missing_required_options].
## Positional options are parsed by their position in the list of arguments
## are are not assigned by name by the user.
## If [param op_name] is an Array of Strings, all elements of the array
## will be aliases for the same argument and will be treated as such during
## parsing.
## [param default] is the default value the option will be set to if it is not
## explicitly set during parsing.
## [param desc] is a human readable text description of the option.
## If the option is successfully added, the Option object will be returned.
## If the option is not successfully added (e.g. a name collision with another
## option occurs), an error message will be printed and [code]null[/code]
## will be returned.
func add_positional_required(op_name, default, desc: String) -> Option:
	var op = add_positional(op_name, default, desc)
	if(op != null):
		op.required = true
	return op


## Headings are used to separate logical groups of command line options
## when printing out options from the help menu.
## Headings are printed out between option descriptions in the order
## that [method add_heading] was called.
func add_heading(display_text: String) -> void:
	options.add_heading(display_text)


## Gets the value assigned to an option after parsing.
## [param name] can be the name of the option or an alias of it.
## [param name] specifies the option whose value you wish to query.
## If the option exists, the value assigned to it during parsing is returned.
## Otherwise, an error message is printed and [code]null[/code] is returned.
func get_value(name: String):
	var found_param: Option = options.get_by_name(name)

	if(found_param != null):
		return found_param.value
	else:
		push_error("COULD NOT FIND OPTION " + name)
		return null


## Gets the value assigned to an option after parsing,
## returning null if the option was not assigned instead of its default value.
## [param name] specifies the option whose value you wish to query.
## This can be useful when providing an order of precedence to your values.
## For example if
## [codeblock]
##     default value < config file < command line
## [/codeblock]
## then you do not want to get the default value for a command line option or
## it will overwrite the value in a config file.
func get_value_or_null(name: String):
	var found_param: Option = options.get_by_name(name)

	if(found_param != null and found_param.has_been_set()):
		return found_param.value
	else:
		return null


## Returns the help text for all defined options.
func get_help() -> String:
	var sep := '---------------------------------------------------------'

	var text := str(sep, "\n", banner, "\n\n")
	text += "Usage\n-----------\n"
	text += "  " + options.get_usage_text() + "\n\n"
	text += "\nOptions\n-----------\n"
	text += options.get_help_text()
	text += str(sep, "\n")
	return text


## Prints out the help text for all defined options.
func print_help() -> void:
	print(get_help())


## Parses a string for all options that have been set in this optparse.
## if [param cli_args] is passed as a String, then it is parsed.
## Otherwise if [param cli_args] is null,
## aruments passed to the Godot engine at startup are parsed.
## See the explanation at the top of addons/gut/cli/optparse.gd to understand
## which arguments this will have access to.
func parse(cli_args=null) -> void:
	parsed_args = cli_args

	if(parsed_args == null):
		parsed_args = OS.get_cmdline_args()
		parsed_args.append_array(OS.get_cmdline_user_args())

	unused = _parse_command_line_arguments(parsed_args)
	_populate_values_dictionary()


## Get all options that were required and were not set during parsing.
## The return value is an Array of Options.
func get_missing_required_options() -> Array:
	return options.get_missing_required_options()


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