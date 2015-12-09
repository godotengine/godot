extends Node


# Changing scenes is most easily done using the functions `change_scene`
# and `change_scene_to` of the SceneTree. This script demonstrates how to
# change scenes without those helpers.


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
	# Immediately free the current scene, there is no risk here.
	get_tree().get_current_scene().free()
	
	# Load new scene
	var packed_scene = ResourceLoader.load(path)
	
	# Instance the new scene
	var instanced_scene = packed_scene.instance()
	
	# Add it to the scene tree, as direct child of root
	get_tree().get_root().add_child(instanced_scene)
	
	# Set it as the current scene, only after it has been added to the tree
	get_tree().set_current_scene(instanced_scene)
