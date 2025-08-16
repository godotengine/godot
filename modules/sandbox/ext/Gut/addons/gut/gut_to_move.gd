# Temporary base script for gut.gd to hold the things to be remvoed and added
# to some utility somewhere.
extends Node

# ------------------------------------------------------------------------------
# deletes all files in a given directory
# ------------------------------------------------------------------------------
func directory_delete_files(path):
	var d = DirAccess.open(path)

	# SHORTCIRCUIT
	if(d == null):
		return

	# Traversing a directory is kinda odd.  You have to start the process of listing
	# the contents of a directory with list_dir_begin then use get_next until it
	# returns an empty string.  Then I guess you should end it.
	d.list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547
	var thing = d.get_next() # could be a dir or a file or something else maybe?
	var full_path = ''
	while(thing != ''):
		full_path = path + "/" + thing
		# file_exists returns fasle for directories
		if(d.file_exists(full_path)):
			d.remove(full_path)
		thing = d.get_next()

	d.list_dir_end()

# ------------------------------------------------------------------------------
# deletes the file at the specified path
# ------------------------------------------------------------------------------
func file_delete(path):
	var d = DirAccess.open(path.get_base_dir())
	if(d != null):
		d.remove(path)

# ------------------------------------------------------------------------------
# Checks to see if the passed in file has any data in it.
# ------------------------------------------------------------------------------
func is_file_empty(path):
	var f = FileAccess.open(path, FileAccess.READ)
	var result = FileAccess.get_open_error()
	var empty = true
	if(result == OK):
		empty = f.get_length() == 0
	f = null
	return empty

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
func get_file_as_text(path):
	return GutUtils.get_file_as_text(path)

# ------------------------------------------------------------------------------
# Creates an empty file at the specified path
# ------------------------------------------------------------------------------
func file_touch(path):
	FileAccess.open(path, FileAccess.WRITE)

# ------------------------------------------------------------------------------
# Simulate a number of frames by calling '_process' and '_physics_process' (if
# the methods exist) on an object and all of its descendents. The specified frame
# time, 'delta', will be passed to each simulated call.
#
# NOTE: Objects can disable their processing methods using 'set_process(false)' and
# 'set_physics_process(false)'. This is reflected in the 'Object' methods
# 'is_processing()' and 'is_physics_processing()', respectively. To make 'simulate'
# respect this status, for example if you are testing an object which toggles
# processing, pass 'check_is_processing' as 'true'.
# ------------------------------------------------------------------------------
func simulate(obj, times, delta, check_is_processing: bool = false):
	for _i in range(times):
		if (
			obj.has_method("_process")
			and (
				not check_is_processing
				or obj.is_processing()
			)
		):
			obj._process(delta)
		if(
			obj.has_method("_physics_process")
			and (
				not check_is_processing
				or obj.is_physics_processing()
			)
		):
			obj._physics_process(delta)

		for kid in obj.get_children():
			simulate(kid, 1, delta, check_is_processing)
