@tool
extends EditorPlugin
var VersionConversion = load("res://addons/gut/version_conversion.gd")
var _bottom_panel = null

func _init():
	if(VersionConversion.error_if_not_all_classes_imported()):
		return


func _version_conversion():
	var EditorGlobals = load("res://addons/gut/gui/editor_globals.gd")
	EditorGlobals.create_temp_directory()

	if(VersionConversion.error_if_not_all_classes_imported()):
		return false

	VersionConversion.convert()
	return true


func _enter_tree():
	if(!_version_conversion()):
		return

	_bottom_panel = preload('res://addons/gut/gui/GutBottomPanel.tscn').instantiate()

	var button = add_control_to_bottom_panel(_bottom_panel, 'GUT')
	button.shortcut_in_tooltip = true

	# ---------
	# I removed this delay because it was causing issues with the shortcut button.
	# The shortcut button wouldn't work right until load_shortcuts is called., but
	# the delay gave you 3 seconds to click it before they were loaded.  This
	# await came with the conversion to 4 and probably isn't needed anymore.
	# I'm leaving it here becuase I don't know why it showed up to begin with
	# and if it's needed, it will be pretty hard to debug without seeing this.
	#
	# This should be deleted after the next release or two if not needed.
	# await get_tree().create_timer(3).timeout
	# ---
	_bottom_panel.set_interface(get_editor_interface())
	_bottom_panel.set_plugin(self)
	_bottom_panel.set_panel_button(button)
	_bottom_panel.load_shortcuts()


func _exit_tree():
	# Clean-up of the plugin goes here
	# Always remember to remove_at it from the engine when deactivated
	remove_control_from_bottom_panel(_bottom_panel)
	_bottom_panel.free()


# This seems like a good idea at first, but it deletes the settings for ALL
# projects.  If by chance you want to do that you can uncomment this, reload the
# project and then disable GUT.
# func _disable_plugin():
#	var GutEditorGlobals = load('res://addons/gut/gui/editor_globals.gd')
# 	GutEditorGlobals.user_prefs.erase_all()