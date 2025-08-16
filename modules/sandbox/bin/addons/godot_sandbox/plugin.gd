@tool
extends EditorPlugin

var popup_window: Window

func _enter_tree():
	# Load your popup scene
	var popup_scene = preload("res://addons/godot_sandbox/downloader.tscn")
	popup_window = popup_scene.instantiate()
	popup_window.close_requested.connect(popup_window.hide)
	
	# Add it to the editor's main screen
	get_editor_interface().get_base_control().add_child(popup_window)
	
	# Optional: Add a menu item to trigger it
	add_tool_menu_item("Godot Sandbox Dependencies...", show_popup)

func _exit_tree():
	remove_tool_menu_item("Godot Sandbox Dependencies...")
	if popup_window:
		popup_window.queue_free()

func show_popup():
	popup_window.popup_centered()
