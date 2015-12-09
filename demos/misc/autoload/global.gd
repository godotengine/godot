extends Node

# Member variables
var current_scene = null


func goto_scene(path):
	# This function will usually be called from a signal callback,
	# or some other function from the running scene.
	# Deleting the current scene at this point might be
	# a bad idea, because it may be inside of a callback or function of it.
	# The worst case will be a crash or unexpected behavior.
	
	# The way around this is deferring the load to a later time, when
	# it is ensured that no code from the current scene is running:
	
	call_deferred("_deferred_goto_scene",path)


func _deferred_goto_scene(path):
	# Immediately free the current scene,
	# there is no risk here.
	current_scene.free()
	
	# Load new scene
	var s = ResourceLoader.load(path)
	
	# Instance the new scene
	current_scene = s.instance()
	
	# Add it to the active scene, as child of root
	get_tree().get_root().add_child(current_scene)


func _ready():
	# Get the current scene at the time of initialization
	current_scene = get_tree().get_current_scene()
