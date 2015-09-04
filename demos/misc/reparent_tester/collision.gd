
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"
var sprite
var firstTime = true
func _ready():
	if(firstTime):
		# Initialization here
		sprite = self.get_node("Sprite")
		sprite.set_modulate(Color(randf(),randf(),randf()))
		connectCollisions()
		firstTime = false

func connectCollisions():
	self.connect("body_enter_shape", self, "_on_body_enter_shape")
	self.connect("body_exit_shape", self, "_on_body_exit_shape")
	self.connect("area_enter_shape", self, "_on_area_enter_shape")
	self.connect("area_exit_shape", self, "_on_area_exit_shape")

func disconnectCollisions():
	if(self.is_connected("body_enter_shape", self, "_on_body_enter_shape")):
		self.disconnect("body_enter_shape", self, "_on_body_enter_shape")
	if(self.is_connected("body_exit_shape", self, "_on_body_exit_shape")):
		self.disconnect("body_exit_shape", self, "_on_body_exit_shape")
	if(self.is_connected("area_enter_shape", self, "_on_area_enter_shape")):
		self.disconnect("area_enter_shape", self, "_on_area_enter_shape")
	if(self.is_connected("area_exit_shape", self, "_on_area_exit_shape")):
		self.disconnect("area_exit_shape", self, "_on_area_exit_shape")

func _on_body_enter_shape( body_id, body, body_shape, area_shape ):
	checkCollision(body, body_shape, area_shape)
	
func _on_body_exit_shape( body_id, body, body_shape, area_shape ):
	pass

func _on_area_enter_shape( area_id, area, area_shape, area_shape ):
	checkCollision(area, area_shape, area_shape)
	
func _on_area_exit_shape( area_id, area, area_shape, area_shape ):
	pass

func checkCollision(body, body_shape, area_shape):
	destroy()

func destroy():
	var main = get_tree().get_root().get_node("main")
	main.collisioner = null
	#self.set_enable_monitoring(false)
	#self.set_monitorable(false)
	disconnectCollisions()
	self.queue_free()