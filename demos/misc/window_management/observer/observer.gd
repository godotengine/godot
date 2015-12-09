
extends Spatial

# member variables
var r_pos = Vector2()
var state

const STATE_MENU = 0
const STATE_GRAB = 1


func direction(vector):
	var v = get_node("Camera").get_global_transform().basis*vector
	v = v.normalized()
	return v


func impulse(event, action):
	if(event.is_action(action) && event.is_pressed() && !event.is_echo()):
		return true
	else:
		return false


func _fixed_process(delta):
	if(state != STATE_GRAB):
		return
	
	if(Input.get_mouse_mode() != Input.MOUSE_MODE_CAPTURED):
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	var dir = Vector3() 
	var cam = get_global_transform()
	var org = get_translation()
	
	if (Input.is_action_pressed("move_forward")):
		dir += direction(Vector3(0, 0, -1))
	if (Input.is_action_pressed("move_backwards")):
		dir += direction(Vector3(0, 0, 1))
	if (Input.is_action_pressed("move_left")):
		dir += direction(Vector3(-1, 0, 0))
	if (Input.is_action_pressed("move_right")):
		dir += direction(Vector3(1, 0, 0))
	
	dir = dir.normalized()
	
	move(dir*10*delta)
	var d = delta*0.1
	
	var yaw = get_transform().rotated(Vector3(0, 1, 0), d*r_pos.x)
	set_transform(yaw)
	
	var cam = get_node("Camera")
	var pitch = cam.get_transform().rotated(Vector3(1, 0, 0), d*r_pos.y)
	cam.set_transform(pitch)
	
	r_pos.x = 0.0
	r_pos.y = 0.0


func _input(event):
	if(event.type == InputEvent.MOUSE_MOTION):
		r_pos = event.relative_pos
	
	if(impulse(event, "ui_cancel")):
		if(state == STATE_GRAB):
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
			state = STATE_MENU
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
			state = STATE_GRAB


func _ready():
	set_fixed_process(true)
	set_process_input(true)
	
	state = STATE_MENU
