
extends Node2D

# member variables
var thread = Thread.new()


# this function runs in a thread!
# threads always take one userdata argument
func _bg_load(path):
	print("THREAD FUNC!")
	# load the resource
	var tex = ResourceLoader.load(path)
	# call _bg_load_done on main thread	
	call_deferred("_bg_load_done")
	return tex # return it


func _bg_load_done():
	# wait for the thread to complete, get the returned value
	var tex = thread.wait_to_finish()
	# set to the sprite
	get_node("sprite").set_texture(tex)


func _on_load_pressed():
	if (thread.is_active()):
		# already working
		return
	print("START THREAD!")
	thread.start(self, "_bg_load", "res://mona.png")
