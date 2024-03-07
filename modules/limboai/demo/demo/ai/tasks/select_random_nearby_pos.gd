@tool
extends BTAction
## SelectRandomNearbyPos: Select a position nearby within specified range.
## Returns SUCCESS.

## Minimum distance to the desired position.
@export var range_min: float = 300.0

## Maximum distance to the desired position.
@export var range_max: float = 500.0

## Blackboard variable that will be used to store the desired position.
@export var position_var: StringName = &"pos"


# Display a customized name (requires @tool).
func _generate_name() -> String:
	return "SelectRandomNearbyPos  range: [%s, %s]  ➜%s" % [
		range_min, range_max,
		LimboUtility.decorate_var(position_var)]


# Called each time this task is ticked (aka executed).
func _tick(_delta: float) -> Status:
	var pos: Vector2
	var is_good_position: bool = false
	while not is_good_position:
		# Randomize until we find a good position (good position == not outside the arena).
		var angle: float = randf() * TAU
		var rand_distance: float = randf_range(range_min, range_max)
		pos = agent.global_position + Vector2(sin(angle), cos(angle)) * rand_distance
		is_good_position = agent.is_good_position(pos)
	blackboard.set_var(position_var, pos)
	return SUCCESS
