
extends StreamPlayer

# member variables here, example:
# var a=2
# var b="textvar"
var firstTime = true
func _ready():
	if(firstTime):
		self.play()
		firstTime = false



