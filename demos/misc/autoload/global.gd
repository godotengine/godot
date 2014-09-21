extends Node


var current_scene = null


func goto_scene(scene):
	#load new scene
	var s = ResourceLoader.load(scene)
	#queue erasing old (don't use free because that scene is calling this method)
	current_scene.queue_free()
	# Remove the scene before loading the previous one.
	# The node is removed when deleted anyway, but this will fix issues that 
	# might arise if both have a root node with the same name,
        # as adding both together will cause the second to be renamed. (not usually a problem, but you might be wanting to look for the node later and not find it)
	get_scene().get_root().remove(current_scene)
	#instance the new scene
	current_scene = s.instance()
	#add it to the active scene, as child of root
	get_scene().get_root().add_child(current_scene)


func _ready():
	# get the current scene
	# it is always the last child of root,
	# after the autoloaded nodes
	var root = get_scene().get_root()
	current_scene = root.get_child( root.get_child_count() -1 )
