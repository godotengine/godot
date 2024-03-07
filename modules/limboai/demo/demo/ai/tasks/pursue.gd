#*
#* pursue.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
@tool
extends BTAction
## Pursue: Move towards target until agent is flanking it.
##
## Returns RUNNING, while moving towards target but not yet at the desired position.
## Returns SUCCESS, when at the desired position from target (flanking it).
## Returns FAILURE, if target is not a valid Node2D instance.

## How close should the agent be to the desired position to return SUCCESS.
const TOLERANCE := 30.0

## Blackboard variable that stores our target (expecting Node2D).
@export var target_var: StringName = &"target"

## Blackboard variable that stores desired speed.
@export var speed_var: StringName = &"speed"

## Desired distance from target.
@export var approach_distance: float = 100.0

var _waypoint: Vector2


# Display a customized name (requires @tool).
func _generate_name() -> String:
	return "Pursue %s" % [LimboUtility.decorate_var(target_var)]


# Called each time this task is entered.
func _enter() -> void:
	var target: Node2D = blackboard.get_var(target_var, null)
	if is_instance_valid(target):
		# Movement is performed in smaller steps.
		# For each step, we select a new waypoint.
		_select_new_waypoint(_get_desired_position(target))


# Called each time this task is ticked (aka executed).
func _tick(_delta: float) -> Status:
	var target: Node2D = blackboard.get_var(target_var, null)
	if not is_instance_valid(target):
		return FAILURE

	var desired_pos: Vector2 = _get_desired_position(target)
	if agent.global_position.distance_to(desired_pos) < TOLERANCE:
		return SUCCESS

	if agent.global_position.distance_to(_waypoint) < TOLERANCE:
		_select_new_waypoint(desired_pos)

	var speed: float = blackboard.get_var(speed_var, 200.0)
	var desired_velocity: Vector2 = agent.global_position.direction_to(_waypoint) * speed
	agent.move(desired_velocity)
	agent.update_facing()
	return RUNNING


## Get the closest flanking position to target.
func _get_desired_position(target: Node2D) -> Vector2:
	var side: float = signf(agent.global_position.x - target.global_position.x)
	var desired_pos: Vector2 = target.global_position
	desired_pos.x += approach_distance * side
	return desired_pos


## Select an intermidiate waypoint towards the desired position.
func _select_new_waypoint(desired_position: Vector2) -> void:
	var distance_vector: Vector2 = desired_position - agent.global_position
	var angle_variation: float = randf_range(-0.2, 0.2)
	_waypoint = agent.global_position + distance_vector.limit_length(150.0).rotated(angle_variation)
