# Our template test class
extends Node

var errors: Array[LuaError]

# id will determine the load order
var id: int = 0
var done: bool = false
var status: bool = true
var time: float
var frames: int

var testName = "Test"
var testDescription = "Base test for all other test's to inhirt from for poly"

# Called when the node enters the scene tree for the first time.
func _ready():
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	time += delta
	frames += 1

func _finalize():
	pass
