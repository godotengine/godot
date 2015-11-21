
extends Control


func _on_button_pressed():
	get_viewport().queue_screen_capture()
	# Let two frames pass to make sure the screen was captured
	yield(get_tree(), "idle_frame")
	yield(get_tree(), "idle_frame")
	# Retrieve the captured image
	var img = get_viewport().get_screen_capture()
	# Create a texture for it
	var tex = ImageTexture.new()
	tex.create_from_image(img)
	# Set it to the capture node
	get_node("capture").set_texture(tex)
