
extends Navigation

# member variables here, example:
# var a=2
# var b="textvar"

const SPEED=4.0

var camrot=0.0

var begin=Vector3()
var end=Vector3()
var m = FixedMaterial.new()

var path=[]

func _process(delta):


	if (path.size()>1):
	
		var to_walk = delta*SPEED
		var to_watch = Vector3(0,1,0)
		while(to_walk>0 and path.size()>=2):
			var pfrom = path[path.size()-1]
			var pto = path[path.size()-2]
			to_watch = (pto - pfrom).normalized()
			var d = pfrom.distance_to(pto)
			if (d<=to_walk):
				path.remove(path.size()-1)
				to_walk-=d
			else:
				path[path.size()-1] = pfrom.linear_interpolate(pto,to_walk/d)
				to_walk=0
				
		var atpos = path[path.size()-1]
		var atdir = to_watch
		atdir.y=0
		
		var t = Transform()
		t.origin=atpos
		t=t.looking_at(atpos+atdir,Vector3(0,1,0))
		get_node("robot_base").set_transform(t)
		
		if (path.size()<2):
			path=[]
			set_process(false)
				
	else:
		set_process(false)

var draw_path=false

func _update_path():

	var p = get_simple_path(begin,end,true)
	path=Array(p) # Vector3array to complex to use, convert to regular array
	path.invert()
	set_process(true)

	if (draw_path):
		var im = get_node("draw")
		im.set_material_override(m)
		im.clear()
		im.begin(Mesh.PRIMITIVE_POINTS,null)
		im.add_vertex(begin)
		im.add_vertex(end)
		im.end()
		im.begin(Mesh.PRIMITIVE_LINE_STRIP,null)
		for x in p:
			im.add_vertex(x)
		im.end()

func _input(ev):

	if (ev.type==InputEvent.MOUSE_BUTTON and ev.button_index==BUTTON_LEFT and ev.pressed):
                
		var from = get_node("cambase/Camera").project_ray_origin(ev.pos)
		var to = from+get_node("cambase/Camera").project_ray_normal(ev.pos)*100
		var p = get_closest_point_to_segment(from,to)
	
		begin=get_closest_point(get_node("robot_base").get_translation())
		end=p

		_update_path()
		
	if (ev.type==InputEvent.MOUSE_MOTION):
		if (ev.button_mask&BUTTON_MASK_MIDDLE):
			
			camrot+=ev.relative_x*0.005
			get_node("cambase").set_rotation(Vector3(0,camrot,0))
			print("camrot ", camrot)

		

func _ready():
	# Initalization here
	set_process_input(true)
	m.set_line_width(3)
	m.set_point_size(3)
	m.set_fixed_flag(FixedMaterial.FLAG_USE_POINT_SIZE,true)
	m.set_flag(Material.FLAG_UNSHADED,true)
	#begin = get_closest_point(get_node("start").get_translation())
	#end = get_closest_point(get_node("end").get_translation())
	#call_deferred("_update_path")

	pass


