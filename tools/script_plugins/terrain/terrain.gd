tool # Always declare as Tool, if it's meant to run in the editor.
extends EditorPlugin


func get_name(): 
	return "Terrain"


func _init():
	print("PLUGIN INIT")
	
 
func _enter_scene():
	add_custom_type("Terrain","Spatial",preload("terrain_node.gd"),preload("terrain.png"))

func _exit_scene():
	remove_custom_type("Terrain")
