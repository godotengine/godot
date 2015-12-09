
extends Sprite

# Member variables
const BEGIN = -113
const END = 907
const TIME = 5.0 # Seconds
const SPEED = (END - BEGIN)/TIME

export var use_idle = true


func _process(delta):
	var ofs = get_pos()
	ofs.x += delta*SPEED
	if (ofs.x > END):
		ofs.x = BEGIN
	set_pos(ofs)


func _fixed_process(delta):
	var ofs = get_pos()
	ofs.x += delta*SPEED
	if (ofs.x > END):
		ofs.x = BEGIN
	set_pos(ofs)


func _ready():
	if (use_idle):
		set_process(true)
	else:
		set_fixed_process(true)
