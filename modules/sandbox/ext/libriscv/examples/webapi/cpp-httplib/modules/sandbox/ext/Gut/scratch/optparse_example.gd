extends SceneTree

var OptParse = load('res://addons/gut/cli/optparse.gd')

func _init():
	var opts = OptParse.new()
	opts.banner = \
"""
This is an example banner so you can see that it is displayed when you specify
the --hlp option.
"""
	opts.add_positional_required("name", "", "Your name")
	opts.add_positional("fav_letter", "", "Favorite Letter (optional)")

	opts.add("--foo", "bar", "Ubiquitous foobar.")
	opts.add("--disabled", false, "Disables something.")
	opts.add("--list", ['a', 'b'], "This is a list of things.")
	opts.add("--volume", 10, "The volume.  Default is not 11, it is [default]")
	opts.add_required("--fav_number", 9, "We must know your favorite number.")

	opts.add_heading("The Other Stuff")
	opts.add("--float_val", 1.5, "This float will be [default] if you do not specify it.")
	var help_flag = opts.add("--hlp", false, "--help is used by Godot, so we need a different one.")

	# This will parse OS.get_cmdline_args and OS.get_cmdline_user_args.  You
	# can pass any array into this though and limit which arguments you want
	# to process.
	opts.parse()
	var missing = opts.get_missing_required_options()
	var unused = opts.unused

	if(help_flag.value):
		opts.print_help()
		quit(0)
	elif(missing.size() > 0):
		print('Required options missing')
		for m in missing:
			print('  ', m.to_s())
		quit(1)
	elif(unused.size() > 0):
		print("Unknown options:  ", unused)
		quit(1)
	elif(opts.get_value("--disabled")):
		print("You have disabled it, so we will do nothing.")
		print("Even though nothing is still soemthing and this")
		print("is clearly something...we shall nevertheless")
		print("consider this nothing")
		print("Quit this yourself")
	else:
		if(opts.get_value("name") == "Me"):
			print("--- Hello me, it's you! ---")
		opts.options.print_option_values()
		quit(0)
