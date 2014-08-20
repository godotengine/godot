
extends Control

# member variables here, example:
# var a=2
# var b="textvar"

var trans = ["linear", "sine", "quint", "quart", "quad", "expo", "elastic", "cubic", "circ", "bounce", "back"]
var eases = ["in", "out", "in_out", "out_in"]
var modes = ["move", "color", "scale", "rotate", "repeat", "pause"]

var state = {
	trans = Tween.TRANS_LINEAR,
	eases = Tween.EASE_IN,
}

func _ready():
	for index in range(trans.size()):
		var name = trans[index]
		get_node("trans/" + name).connect("pressed", self, "on_trans_changed", [name, index])
	
	for index in range(eases.size()):
		var name = eases[index]
		get_node("eases/" + name).connect("pressed", self, "on_eases_changed", [name, index])
	
	for index in range(modes.size()):
		var name = modes[index]
		get_node("modes/" + name).connect("pressed", self, "on_modes_changed", [name])
	
	get_node("color/color_from").set_color(Color(1, 0, 0, 1))
	get_node("color/color_from").connect("color_changed", self, "on_color_changed")
	
	get_node("color/color_to").set_color(Color(0, 1, 1, 1))
	get_node("color/color_to").connect("color_changed", self, "on_color_changed")
	
	get_node("trans/linear").set_pressed(true)
	get_node("eases/in").set_pressed(true)
	get_node("modes/move").set_pressed(true)
	get_node("modes/repeat").set_pressed(true)
	
	reset_tween()
	
	# Initalization here
	pass

func on_trans_changed(name, index):
	for index in range(trans.size()):
		var pressed = trans[index] == name
		var btn = get_node("trans/" + trans[index])
		
		btn.set_pressed(pressed)
		btn.set_ignore_mouse(pressed)
	
	state.trans = index
	reset_tween()
	
func on_eases_changed(name, index):
	for index in range(eases.size()):
		var pressed = eases[index] == name
		var btn = get_node("eases/" + eases[index])
		
		btn.set_pressed(pressed)
		btn.set_ignore_mouse(pressed)
	
	state.eases = index
	reset_tween()
	
func on_modes_changed(name):
	var tween = get_node("tween")
	if name == "pause":
		if get_node("modes/pause").is_pressed():
			tween.stop_all()
			get_node("timeline").show()
		else:
			tween.resume_all()
			get_node("timeline").hide()
	else:
		reset_tween()
	
func on_color_changed(color):
	reset_tween()
	
func reset_tween():
	var tween = get_node("tween")
	tween.reset_all()
	tween.remove_all()
	
	var sprite = get_node("tween/area/sprite")
	
	if get_node("modes/move").is_pressed():
		tween.interpolate_method(sprite, "set_pos", Vector2(0,0), Vector2(736, 184), 2, state.trans, state.eases)
	
	if get_node("modes/color").is_pressed():
		tween.interpolate_method(sprite, "set_modulate", get_node("color/color_from").get_color(), get_node("color/color_to").get_color(), 2, state.trans, state.eases)
	else:
		sprite.set_modulate(Color(1, 1, 1, 1))
	
	if get_node("modes/scale").is_pressed():
		tween.interpolate_method(sprite, "set_scale", Vector2(0.5,0.5), Vector2(1.5, 1.5), 2, state.trans, state.eases)
	
	if get_node("modes/rotate").is_pressed():
		tween.interpolate_method(sprite, "set_rot", 0.0, 6.28, 2, state.trans, state.eases)
	
	tween.set_repeat(get_node("modes/repeat").is_pressed())
	tween.start()
	
	if get_node("modes/pause").is_pressed():
		tween.stop_all()
		get_node("timeline").show()
		get_node("timeline").set_value(0)
	else:
		tween.resume_all()
		get_node("timeline").hide()
	
func _on_tween_step( object, key, elapsed, value ):
	var timeline = get_node("timeline")
	var ratio = 100 * (elapsed / 2)
	timeline.set_value(ratio)
	

func _on_timeline_value_changed( value ):
	if !get_node("modes/pause").is_pressed():
		return
	
	var tween = get_node("tween")
	tween.seek(2.0 * value / 100)
	
