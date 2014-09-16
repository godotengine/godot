extends Node

# This class is the controller of the main scene. It manages scenes and transitions, as well as the music.

# Variables ---------------------------------------------------

# currently loaded scene. Can be a cinematic or a playable level or whatever is in the MapLayer node
var current_scene=null
# MapLayer node
var main_scene=null
# root node of the main scene
var root=null

# Classes ----------------------------------------------------------
var player_class = preload("res://player.res")

# Functions ----------------------------------------------------------

# initialize pointers to nodes
func _ready():
	var _root=get_scene().get_root()
	root = _root.get_child(_root.get_child_count()-1)
	main_scene = root.get_node("MapLayer")

# load a new scene and replace the current scene. Should not be directly called.
# String scene : name of the new scene to load
func _goto_scene(scene):
	if(current_scene!=null):
		current_scene.queue_free()
	var s=ResourceLoader.load(scene)
	current_scene = s.instance()
	main_scene.add_child(current_scene)

# loads a cinematic scene. HUD will be hidden.
# String scene : name of the new scene to load
func goto_cinematic_scene(scene):
	# load scene
	_goto_scene(scene)
	# hide the HUD
	get_scene().get_nodes_in_group("HUDroot")[0].hide()
	

# loads a playable scene. HUD will be shown.
# String scene : name of the new scene to load
# String entry_point : name (must be unique) of the node containing the position where the player must appears. If null, the player starts at the default position. The node must be part of the group "entries".
func goto_playable_scene(scene,entry_point):
	# load scene
	_goto_scene(scene)
	# show the HUD
	get_scene().get_nodes_in_group("HUDroot")[0].show()
	
	# move the player to the entry point.
	# First, search both nodes
	var player_nodes=get_scene().get_nodes_in_group("player") # keep in mind that the player of the old scene is maybe still not gone by the garbage collector.
	var entry_nodes=get_scene().get_nodes_in_group("entries")
	
	var entry_node=null
	for e in entry_nodes:
		if(e.get_name()==entry_point):
			entry_node=e
			break
	
	# if the player and the entry point are found, move the player
	if(player_nodes!=null && entry_node!=null):
		var entry_pos=entry_node.get_global_pos()
		# move all found players to the position. If the player of the old scene is still there, it will also be moved. But it doesn't matter since we don't see it and it'll be garbage collected soon.
		for player_node in player_nodes:
			player_node.set_pos(entry_pos)
	
	# starts transition, since the screen should be actually all white
	fade_in()

# starts a fade out transition
func fade_out():
	root.get_node("anim").play("fadeout")

# starts a fade in transition
func fade_in():
	root.get_node("anim").play("fadein")

# plays the current default theme
func play_map_track():
	get_node("/root/soundMgr").play_track(get_node("/root/gamedata").get_map_track())