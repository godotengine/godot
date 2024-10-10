extends Resource

@export var number: int
@export var bar: Resource


func _init(p_number = 0, p_bar = null):
	number = p_number
	bar = p_bar
