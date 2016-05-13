
extends Node2D

# Member variables
var thread = Thread.new()


# This function runs in a thread!
# Threads always take one userdata argument
func _bg_load(path):
	print("THREAD FUNC!")
	# Load the resource
	var tex = ResourceLoader.load(path)
	# Call _bg_load_done on main thread
	call_deferred("_bg_load_done")
	return tex # return it


func _bg_load_done():
	# Wait for the thread to complete, get the returned value
	var tex = thread.wait_to_finish()
	# Set to the sprite
	get_node("sprite").set_texture(tex)


func _on_load_pressed():
	if (thread.is_active()):
		# Already working
		return
	print("START THREAD!")
	thread.start(self, "_bg_load", "res://mona.png")
