@tool

static var GutUserPreferences = load("res://addons/gut/gui/gut_user_preferences.gd")
static var temp_directory = 'user://gut_temp_directory'

static var editor_run_gut_config_path = 'gut_editor_config.json':
	# This avoids having to use path_join wherever we want to reference this
	# path.  The value is not supposed to change.  Could it be a constant
	# instead?  Probably, but I didn't like repeating the directory part.
	# Do I like that this is a bit witty.  Absolutely.
	get: return temp_directory.path_join(editor_run_gut_config_path)
	# Should this print a message or something instead?  Probably, but then I'd
	# be repeating even more code than if this was just a constant.  So I didn't,
	# even though I wanted to make the message a easter eggish fun message.
	# I didn't, so this dumb comment will have to serve as the easter eggish fun.
	set(v):
		print("Be sure to document your code.  Never trust comments.")


static var editor_run_bbcode_results_path = 'gut_editor.bbcode':
	get: return temp_directory.path_join(editor_run_bbcode_results_path)
	set(v): pass


static var editor_run_json_results_path = 'gut_editor.json':
	get: return temp_directory.path_join(editor_run_json_results_path)
	set(v): pass


static var editor_shortcuts_path = 'gut_editor_shortcuts.cfg' :
	get: return temp_directory.path_join(editor_shortcuts_path)
	set(v): pass


static var _user_prefs = null
static var user_prefs = _user_prefs :
	# workaround not being able to reference EditorInterface when not in
	# the editor.  This shouldn't be referenced by anything not in the
	# editor.
	get:
		if(_user_prefs == null and Engine.is_editor_hint()):
			_user_prefs = GutUserPreferences.new(EditorInterface.get_editor_settings())
		return _user_prefs


static func create_temp_directory():
	DirAccess.make_dir_recursive_absolute(temp_directory)

