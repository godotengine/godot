extends RigidBody2D

var mouse_inside
var mouse_pressed

func _ready():
	set_pickable(true)
	set_process_input(true)
	set_process(true)
	self.connect("mouse_enter", self, "_on_mouse_enter")
	self.connect("mouse_exit", self, "_on_mouse_exit")

func _on_mouse_enter():
	mouse_inside = true

func _on_mouse_exit():
	mouse_inside = false

func _input(event):
	if event.is_action_pressed("click") and mouse_inside:
		mouse_pressed = true
	elif event.is_action_released("click") and mouse_inside:
		mouse_pressed = false

func _process(delta):
	if mouse_pressed:
		var mouse_position = get_viewport().get_mouse_pos()
		set_pos(mouse_position)