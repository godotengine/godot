
extends Spatial

# member variables here, example:
# var a=2
# var b="textvar"

var prev_pos=null

func _input(ev):
	if (ev.type in [InputEvent.MOUSE_BUTTON,InputEvent.MOUSE_MOTION]):
		var pos = ev.pos
		var rfrom = get_node("camera").project_ray_origin(pos)
		var rnorm = get_node("camera").project_ray_normal(pos)
		
		#simple collision test against aligned plane
		#for game UIs of this kind consider more complex collision against plane
		var p = Plane(Vector3(0,0,1),0).intersects_ray(rfrom,rnorm)
		if (p==null):
			return
			
		pos.x=(p.x+1.5)*100
		pos.y=(-p.y+0.75)*100
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
	get_node("quad").get_material_override().set_texture(FixedMaterial.PARAM_DIFFUSE, get_node("viewport").get_render_target_texture() )
	set_process_input(true)
	
	pass


