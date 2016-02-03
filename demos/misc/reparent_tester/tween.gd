
extends Tween

# member variables here, example:
# var a=2
# var b="textvar"
var firstTime = true
func _ready():
	if(firstTime):
		var sprite = get_node("Sprite")
		self.interpolate_method(sprite, "set_pos", Vector2(0,0), Vector2(100.0, 100.0), 2, TRANS_BOUNCE,EASE_IN)
		#self.interpolate_property(sprite, "transform/pos", Vector2(0,0), Vector2(100.0, 100.0), 2, TRANS_BOUNCE,EASE_IN)
		self.start()
		firstTime = false
	else:
		self.resume_all()