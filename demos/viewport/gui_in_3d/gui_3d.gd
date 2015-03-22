
extends Spatial

# member variables here, example:
# var a=2
# var b="textvar"

var prev_pos=null


func _input( ev ):
	#all other (non-mouse) events
	if (not ev.type in [InputEvent.MOUSE_BUTTON,InputEvent.MOUSE_MOTION,InputEvent.SCREEN_DRAG,InputEvent.SCREEN_TOUCH]):	
		get_node("viewport").input(ev)
		
	
#mouse events for area
func _on_area_input_event( camera, ev, click_pos, click_normal, shape_idx ):
	
	#use click pos (click in 3d space, convert to area space
	var pos = get_node("area").get_global_transform().affine_inverse() * click_pos
	#convert to 2D
	pos = Vector2(pos.x,pos.y)
	#convert to viewport coordinate system		
	pos.x=(pos.x+1.5)*100
	pos.y=(-pos.y+0.75)*100
	#set to event
	ev.pos=pos
	ev.global_pos=pos
	if (prev_pos==null):
		prev_pos=pos
	if (ev.type==InputEvent.MOUSE_MOTION):
		ev.relative_pos=pos-prev_pos
	prev_pos=pos
		
	get_node("viewport").input(ev)
	
	

func _ready():
	# Initalization here
	get_node("area/quad").get_material_override().set_texture(FixedMaterial.PARAM_DIFFUSE, get_node("viewport").get_render_target_texture() )
	set_process_input(true)
	pass

