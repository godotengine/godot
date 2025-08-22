extends Node


@export var max_hp: int = 0
@export var current_hp: int = 0 :
	get:
		return current_hp
	set(value):
		current_hp = clamp(value, 0, max_hp)

func set_max_hp(value: int) -> void:
	if value < 0:
		value = 0
	max_hp = value

func get_max_hp() -> int:
	return max_hp
