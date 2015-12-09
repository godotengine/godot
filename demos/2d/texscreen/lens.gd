
extends BackBufferCopy

# Member variables
const MOTION_SPEED = 150

var vsize
var dir


func _process(delta):
	var pos = get_pos() + dir*delta*MOTION_SPEED
	
	if (pos.x < 0):
		dir.x = abs(dir.x)
	elif (pos.x > vsize.x):
		dir.x = -abs(dir.x)
	
	if (pos.y < 0):
		dir.y = abs(dir.y)
	elif (pos.y > vsize.y):
		dir.y = -abs(dir.y)
	
	set_pos(pos)


func _ready():
	vsize = get_viewport_rect().size
	var pos = vsize*Vector2(randf(), randf())
	set_pos(pos)
	dir = Vector2(randf()*2.0 - 1, randf()*2.0 - 1).normalized()
	set_process(true)
