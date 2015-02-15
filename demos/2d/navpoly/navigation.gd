
extends Navigation2D

# member variables here, example:
# var a=2
# var b="textvar"
var begin=Vector2()
var end=Vector2()
var path=[]

const SPEED=200.0

func _process(delta):


	if (path.size()>1):
	
		var to_walk = delta*SPEED
		while(to_walk>0 and path.size()>=2):
			var pfrom = path[path.size()-1]
			var pto = path[path.size()-2]
			var d = pfrom.distance_to(pto)
			if (d<=to_walk):
				path.remove(path.size()-1)
				to_walk-=d
			else:
				path[path.size()-1] = pfrom.linear_interpolate(pto,to_walk/d)
				to_walk=0
				
		var atpos = path[path.size()-1]	
		get_node("agent").set_pos(atpos)
		
		if (path.size()<2):
			path=[]
			set_process(false)
				
	else:
		set_process(false)



func _update_path():

	var p = get_simple_path(begin,end,true)
	path=Array(p) # Vector2array to complex to use, convert to regular array
	path.invert()
	
	set_process(true)


func _input(ev):
	if (ev.type==InputEvent.MOUSE_BUTTON and ev.pressed and ev.button_index==1):
		begin=get_node("agent").get_pos()
		#mouse to local navigatio cooards
		end=ev.pos - get_pos()
		_update_path()

func _ready():
	# Initialization here
	set_process_input(true)
	pass


