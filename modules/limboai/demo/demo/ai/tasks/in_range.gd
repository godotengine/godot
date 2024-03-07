#*
#* in_range.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
@tool
extends BTCondition

## InRange condition checks if the agent is within a range of target,
## defined by distance_min and distance_max.
## Returns SUCCESS if the agent is within the defined range;
## otherwise, returns FAILURE.

## Minimum distance to target.
@export var distance_min: float

## Maximum distance to target.
@export var distance_max: float

## Blackboard variable that holds the target (expecting Node2D).
@export var target_var: StringName = &"target"

var _min_distance_squared: float
var _max_distance_squared: float


# Called to generate a display name for the task.
func _generate_name() -> String:
	return "InRange (%d, %d) of %s" % [distance_min, distance_max,
		LimboUtility.decorate_var(target_var)]


# Called to initialize the task.
func _setup() -> void:
	## Small performace optimization
	_min_distance_squared = distance_min * distance_min
	_max_distance_squared = distance_max * distance_max


# Called when the task is executed.
func _tick(_delta: float) -> Status:
	var target: Node2D = blackboard.get_var(target_var, null)
	if not is_instance_valid(target):
		return FAILURE

	var dist_sq: float = agent.global_position.distance_squared_to(target.global_position)
	if dist_sq >= _min_distance_squared and dist_sq <= _max_distance_squared:
		return SUCCESS
	else:
		return FAILURE
