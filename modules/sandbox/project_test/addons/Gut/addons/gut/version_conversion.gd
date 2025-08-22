class ConfigurationUpdater:
	var EditorGlobals = load("res://addons/gut/gui/editor_globals.gd")

	func warn(message):
		print('GUT Warning:  ', message)


	func info(message):
		print("GUT Info:  ", message)


	func moved_file(from, to):
		if(FileAccess.file_exists(from) and !FileAccess.file_exists(to)):
			info(str('Copying [', from, '] to [', to, ']'))
			var result = DirAccess.copy_absolute(from, to)
			if(result != OK):
				warn(str('Could not copy [', from, '] to [', to, ']'))

		if(FileAccess.file_exists(from) and FileAccess.file_exists(to)):
			warn(str('File [', from, '] has been moved to [', to, "].\n    You can delete ", from))


	func move_user_file(from, to):
		if(from.begins_with('user://') and to.begins_with('user://')):
			if(FileAccess.file_exists(from) and !FileAccess.file_exists(to)):
				info(str('Moving [', from, '] to [', to, ']'))
				var result = DirAccess.copy_absolute(from, to)
				if(result == OK):
					info(str('    ', 'Created ', to))
					result = DirAccess.remove_absolute(from)
					if(result != OK):
						warn(str('    ', 'Could not delete ', from))
					else:
						info(str('    ', 'Deleted ', from))
				else:
					warn(str('    ', 'Could not copy [', from, '] to [', to, ']'))
		else:
			warn(str('Attempt to move_user_file with files not in user:// ', from, '->', to))


	func remove_user_file(which):
		if(which.begins_with('user://') and FileAccess.file_exists(which)):
			info(str('Deleting obsolete file ', which))
			var result = DirAccess.remove_absolute(which)
			if(result != OK):
				warn(str('    ', 'Could not delete ', which))
			else:
				info(str('    ', 'Deleted ', which))

class v9_2_0:
	extends ConfigurationUpdater

	func validate():
		moved_file('res://.gut_editor_config.json', EditorGlobals.editor_run_gut_config_path)
		moved_file('res://.gut_editor_shortcuts.cfg', EditorGlobals.editor_shortcuts_path)
		remove_user_file('user://.gut_editor.bbcode')
		remove_user_file('user://.gut_editor.json')

# list=Array[Dictionary]([{
# "base": &"RefCounted",
# "class": &"DynamicGutTest",
# "icon": "",
# "language": &"GDScript",
# "path": "res://test/resources/tools/dynamic_gut_test.gd"
# }, {
# "base": &"RefCounted",
# "class": &"GutDoubleTestInnerClasses",
# "icon": "",
# "language": &"GDScript",
# "path": "res://test/resources/doubler_test_objects/inner_classes.gd"
# }, ... ])
static func get_missing_gut_class_names() -> Array:
	var gut_class_names = ["GutHookScript",
		"GutInputFactory",
		"GutInputSender",
		"GutMain",
		"GutStringUtils",
		"GutTest",
		"GutUtils",]

	var class_cach_path = 'res://.godot/global_script_class_cache.cfg'
	var cfg = ConfigFile.new()
	cfg.load(class_cach_path)

	var all_class_names = {}
	var missing  = []
	var class_cache_entries = cfg.get_value('', 'list', [])

	for entry in class_cache_entries:
		if(entry.path.begins_with(&"res://addons/gut/")):
			# print(entry["class"], ':  ', entry["path"])
			all_class_names[entry["class"]] = entry

	for cn in gut_class_names:
		if(!all_class_names.has(cn)):
			missing.append(cn)

	return missing


static func error_if_not_all_classes_imported() -> bool:
	var missing_class_names = get_missing_gut_class_names()
	if(missing_class_names.size() > 0):
		push_error(str("Some GUT class_names have not been imported.  Please restart the Editor or run godot --headless --import\n",
			"Missing class_names:  ",
			missing_class_names))
		return true
	else:
		return false




static func convert():
	var inst = v9_2_0.new()
	inst.validate()
