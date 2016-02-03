
extends LightOccluder2D

# member variables here, example:
# var a=2
# var b="textvar"
var diameter
var timer
var firstTime = true
func _ready():
	if(firstTime):
		diameter = 40.0
		timer = 0
		set_fixed_process(true)
		firstTime = false

func _fixed_process(delta):
	timer += delta*1.5
	var pos = self.get_pos()
	pos.x = diameter * sin(timer)
	pos.y = diameter * cos(timer)
	self.set_pos(pos)
