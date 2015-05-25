
extends Sprite


export var use_idle=true

# member variables here, example:
# var a=2
# var b="textvar"
const BEGIN = -113
const END = 907
const TIME = 5.0 # seconds
const SPEED = (END-BEGIN)/TIME

func _process(delta):
	var ofs = get_pos()
	ofs.x+=delta*SPEED
	if (ofs.x>END):
		ofs.x=BEGIN
	set_pos(ofs)
	
func _fixed_process(delta):
	var ofs = get_pos()
	ofs.x+=delta*SPEED
	if (ofs.x>END):
		ofs.x=BEGIN
	set_pos(ofs)
	

func _ready():
	# Initialization here
	if (use_idle):
		set_process(true)
	else:
		set_fixed_process(true)
	pass


