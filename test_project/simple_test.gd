extends Control

func _ready():
	print("Testing Wayland Layer Shell implementation...")
	
	# Test setting wayland layer property with numeric values first
	print("Testing numeric values...")
	get_window().wayland_layer = 1  # WAYLAND_LAYER_BACKGROUND
	print("Set layer to 1")
	
	get_window().wayland_layer = 3  # WAYLAND_LAYER_TOP
	print("Set layer to 3")
	
	# Test getting wayland layer property
	var current_layer = get_window().wayland_layer
	print("Current wayland_layer: ", current_layer)
	
	print("Test completed!")
	get_tree().quit()
