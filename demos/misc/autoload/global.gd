extends Node


var current_scene = null


func goto_scene(scene):
	#load new scene
	var s = ResourceLoader.load(scene)
	#queue erasing old (don't use free because that scene is calling this method)
	current_scene.queue_free()
	#instance the new scene
	current_scene = s.instance()
	#add it to the active scene, as child of root
	get_tree().get_root().add_child(current_scene)


func _ready():
	# get the current scene
	# it is always the last child of root,
	# after the autoloaded nodes
	var root = get_tree().get_root()
	current_scene = root.get_child( root.get_child_count() -1 )
