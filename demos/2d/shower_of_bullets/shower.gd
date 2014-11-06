
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"

var touching=0

func _input(ev):

	if (ev.type==InputEvent.MOUSE_MOTION):
		get_node("player").set_pos(ev.pos-Vector2(0,16))


func _on_player_body_enter_shape( body_id, body, body_shape, area_shape ):

	touching+=1
	if (touching==1):
		get_node("player/sprite").set_frame(1)


func _on_player_body_exit_shape( body_id, body, body_shape, area_shape ):

	touching-=1
	if (touching==0):
		get_node("player/sprite").set_frame(0)


func _ready():
	set_process_input(true)
	pass
