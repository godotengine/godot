
extends Control

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():
	# Initialization here
	pass




func _on_button_pressed():
	get_viewport().queue_screen_capture()
	#let two frames pass to make sure the screen was aptured
	yield(get_tree(),"idle_frame")
	yield(get_tree(),"idle_frame")
	#retrieve the captured image
	var img = get_viewport().get_screen_capture()
	#create a texture for it
	var tex = ImageTexture.new()
	tex.create_from_image(img)
	#set it to the capture node
	get_node("capture").set_texture(tex)
	pass # replace with function body
