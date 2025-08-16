extends SceneTree

var Optparse = load('res://addons/gut/cli/optparse.gd')
var WarningsManager = load("res://addons/gut/warnings_manager.gd")
const WARN_VALUE_PRINT_POSITION = 36

var godot_default_warnings = {
  "assert_always_false": 1,             "assert_always_true": 1,  			"confusable_identifier": 1,
  "confusable_local_declaration": 1,    "confusable_local_usage": 1,  		"constant_used_as_function": 1,
  "deprecated_keyword": 1,              "empty_file": 1,  					"enable": true,
  "exclude_addons": true, 				"function_used_as_property": 1,  	"get_node_default_without_onready": 2,
  "incompatible_ternary": 1,  			"inference_on_variant": 2,  		"inferred_declaration": 0,
  "int_as_enum_without_cast": 1,  		"int_as_enum_without_match": 1,  	"integer_division": 1,
  "narrowing_conversion": 1,  			"native_method_override": 2,  		"onready_with_export": 2,
  "property_used_as_function": 1,  		"redundant_await": 1,  				"redundant_static_unload": 1,
  "renamed_in_godot_4_hint": 1,  		"return_value_discarded": 0,  		"shadowed_global_identifier": 1,
  "shadowed_variable": 1,  				"shadowed_variable_base_class": 1,  "standalone_expression": 1,
  "standalone_ternary": 1,  			"static_called_on_instance": 1,  	"unassigned_variable": 1,
  "unassigned_variable_op_assign": 1,  	"unreachable_code": 1,  			"unreachable_pattern": 1,
  "unsafe_call_argument": 0,  			"unsafe_cast": 0,  					"unsafe_method_access": 0,
  "unsafe_property_access": 0,  		"unsafe_void_return": 1,  			"untyped_declaration": 0,
  "unused_local_constant": 1,  			"unused_parameter": 1,  			"unused_private_class_variable": 1,
  "unused_signal": 1,  					"unused_variable": 1
}

var gut_default_changes = {
  "exclude_addons": false, 				"redundant_await": 0,
}

var warning_settings = {}

func _setup_warning_settings():
	warning_settings["godot_default"] = godot_default_warnings
	warning_settings["current"] = WarningsManager.create_warnings_dictionary_from_project_settings()
	warning_settings["all_warn"] = WarningsManager.create_warn_all_warnings_dictionary()

	var gut_default = godot_default_warnings.duplicate()
	gut_default.merge(gut_default_changes, true)
	warning_settings["gut_default"] = gut_default


func _warn_value_to_s(value):
	var readable = str(value).capitalize()
	if(typeof(value) == TYPE_INT):
		readable = WarningsManager.WARNING_LOOKUP.get(value, str(readable, ' ???'))
		readable = readable.capitalize()
	return readable


func _human_readable(warnings):
	var to_return = ""
	for key in warnings:
		var readable = _warn_value_to_s(warnings[key])
		to_return += str(key.capitalize().rpad(35, ' '), readable, "\n")
	return to_return


func _dump_settings(which):
	if(warning_settings.has(which)):
		GutUtils.pretty_print(warning_settings[which])
	else:
		print("UNKNOWN print option ", which)


func _print_settings(which):
	if(warning_settings.has(which)):
		print(_human_readable(warning_settings[which]))
	else:
		print("UNKNOWN print option ", which)


func _apply_settings(which):
	if(!warning_settings.has(which)):
		print("UNKNOWN set option ", which)
		return

	var pre_settings = warning_settings["current"]
	var new_settings = warning_settings[which]

	if(new_settings == pre_settings):
		print("-- Settings are the same, no changes were made --")
		return

	WarningsManager.apply_warnings_dictionary(new_settings)
	ProjectSettings.save()
	print("-- Project Warning Settings have been updated --")
	print(_diff_changes_text(pre_settings))


func _diff_text(w1, w2, diff_col_pad=10):
	var to_return = ""
	for key in w1:
		var v1_text = _warn_value_to_s(w1[key])
		var v2_text = _warn_value_to_s(w2[key])
		var diff_text = v1_text
		var prefix = "  "

		if(v1_text != v2_text):
			var diff_prefix = " "
			if(w1[key] > w2[key]):
				diff_prefix = "-"
			else:
				diff_prefix = "+"
			prefix = "* "
			diff_text = str(v1_text.rpad(diff_col_pad, ' '), diff_prefix, v2_text)

		to_return += str(str(prefix, key.capitalize()).rpad(WARN_VALUE_PRINT_POSITION, ' '), diff_text, "\n")

	return to_return.rstrip("\n")


func _diff_changes_text(pre_settings):
	var orig_diff_text = _diff_text(
		pre_settings,
		WarningsManager.create_warnings_dictionary_from_project_settings(),
		0)
	# these next two lines are fragile and brute force...enjoy
	var diff_text = orig_diff_text.replace("-", " -> ")
	diff_text = diff_text.replace("+", " -> ")

	if(orig_diff_text == diff_text):
		diff_text += "\n-- No changes were made --"
	else:
		diff_text += "\nChanges will not be visible in Godot until it is restarted.\n"
		diff_text += "Even if it asks you to reload...Maybe.  Probably."

	return diff_text



func _diff(name_1, name_2):
	if(warning_settings.has(name_1) and warning_settings.has(name_2)):
		var c2_pad = name_1.length() + 2
		var heading = str(" ".repeat(WARN_VALUE_PRINT_POSITION), name_1.rpad(c2_pad, ' '), name_2, "\n")
		heading += str(
			" ".repeat(WARN_VALUE_PRINT_POSITION),
			"-".repeat(name_1.length()).rpad(c2_pad, " "),
			"-".repeat(name_2.length()),
			"\n")

		var text = _diff_text(warning_settings[name_1], warning_settings[name_2], c2_pad)

		print(heading)
		print(text)

		var diff_count = 0
		for line in text.split("\n"):
			if(!line.begins_with("  ")):
				diff_count += 1

		if(diff_count == 0):
			print('-- [', name_1, "] and [", name_2, "] are the same --")
		else:
			print('-- There are ', diff_count, ' differences between [', name_1, "] and [", name_2, "] --")
	else:
		print("One or more unknown Warning Level Names:, [", name_1, "] [", name_2, "]")


func _set_settings(nvps):
	var pre_settings = warning_settings["current"]
	for i in range(nvps.size()/2):
		var s_name = nvps[i * 2]
		var s_value = nvps[i * 2 + 1]
		if(godot_default_warnings.has(s_name)):
			var t = typeof(godot_default_warnings[s_name])
			if(t == TYPE_INT):
				s_value = s_value.to_int()
			elif(t == TYPE_BOOL):
				s_value = s_value.to_lower() == 'true'

			WarningsManager.set_project_setting_warning(s_name, s_value)
			ProjectSettings.save()
	print(_diff_changes_text(pre_settings))



func _setup_options():
	var opts = Optparse.new()
	opts.banner = """
	This script prints info about or sets the warning settings for the project.
	Each action requires one or more Warning Level Names.

	Warning Level Names:
	    * current        The current settings for the project.
	    * godot_default  The default settings for Godot.
	    * gut_default    The warning settings that is used when developing GUT.
	    * all_warn       Everything set to warn.
	""".dedent()

	opts.add('-h', false, 'Print this help')
	opts.add('-set', [], "Sets a single setting in the project settings and saves.\n" +
						 "Use -dump to see a list of setting names and values.\n" +
						 "Example: -set enabled,true -set unsafe_cast,2 -set unreachable_code,0")
	opts.add_heading(" Actions (require Warning Level Name)")
	opts.add('-diff', [], "Shows the difference between two Warning Level Names.\n" +
						  "Example:  -diff current,all_warn")
	opts.add('-dump', 'none', "Prints a dictionary of the warning values.")
	opts.add('-print', 'none', "Print human readable warning values.")
	opts.add('-apply', 'none', "Applys one of the Warning Level Names to the project settings.  You should restart after using this")

	return opts

func _print_help(opts):
	opts.print_help()



func _init():
	# Testing might set this flag but it should never be disabled for this tool
	# or it cannot save project settings, but says it did.  Sneakily use the
	# private property to get around this property being read-only.  Don't
	# try this at home.
	WarningsManager._disabled = false

	_setup_warning_settings()

	var opts = _setup_options()
	opts.parse()

	if(opts.unused.size() != 0):
		opts.print_help()
		print("Unknown arguments ", opts.unused)
	if(opts.values.h):
		opts.print_help()
	elif(opts.values.print != 'none'):
		_print_settings(opts.values.print)
	elif(opts.values.dump != 'none'):
		_dump_settings(opts.values.dump)
	elif(opts.values.apply != 'none'):
		_apply_settings(opts.values.apply )
	elif(opts.values.diff.size() == 2):
		_diff(opts.values.diff[0], opts.values.diff[1])
	elif(opts.values.set.size() % 2 == 0):
		_set_settings(opts.values.set)
	else:
		opts.print_help()
		print("You didn't specify any options or too many or not the right size or something invalid.  I don't know what you want to do.")

	quit()