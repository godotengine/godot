tool
extends EditorPlugin

var import_plugin

func _enter_tree():
	
	import_plugin = preload("res://addons/custom_import_plugin/import_plugin.gd").new()

	# pass the GUI base control, so the dialog has a parent node
	import_plugin.config( get_base_control() )

	add_import_plugin( import_plugin) 

func _exit_tree():

	remove_import_plugin( import_plugin ) 




	