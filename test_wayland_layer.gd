extends Node

func _ready():
	print("Testing Wayland Layer Shell implementation...")
	
	# Test setting wayland layer property
	print("Setting wayland_layer to BACKGROUND")
	get_window().wayland_layer = Window.WaylandLayer.BACKGROUND
	
	print("Setting wayland_layer to TOP")  
	get_window().wayland_layer = Window.WaylandLayer.TOP
	
	print("Setting wayland_layer to OVERLAY")
	get_window().wayland_layer = Window.WaylandLayer.OVERLAY
	
	# Test getting wayland layer property
	var current_layer = get_window().wayland_layer
	print("Current wayland_layer: ", current_layer)
	
	print("Test completed!")
