#*
#* select_flanking_pos.gd
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
## SelectFlankingPos on the side of a target, and return SUCCESS.
## Returns FAILURE, if the target is not valid.

enum AgentSide {
	CLOSEST,
	FARTHEST,
	BACK,
	FRONT,
}

## Blackboard variable that holds current target (should be a Node2D instance).
@export var target_var: StringName = &"target"

## Which agent's side should we flank?
@export var flank_side: AgentSide = AgentSide.CLOSEST

## Minimum range relative to the target.
@export var range_min: int = 300

## Maximum range relative to the target.
@export var range_max: int = 400

## Blackboard variable that will be used to store selected position.
@export var position_var: StringName = &"pos"


# Display a customized name (requires @tool).
func _generate_name() -> String:
	return "SelectFlankingPos  target: %s  range: [%s, %s]  side: %s  ➜%s" % [
		LimboUtility.decorate_var(target_var),
		range_min,
		range_max,
		AgentSide.keys()[flank_side],
		LimboUtility.decorate_var(position_var)]


# Called each time this task is ticked (aka executed).
func _tick(_delta: float) -> Status:
	var target := blackboard.get_var(target_var) as Node2D
	if not is_instance_valid(target):
		return FAILURE

	var dir: float  # 1.0 is right, -1.0 is left (relative to target)
	match flank_side:
		AgentSide.FARTHEST:
			dir = signf(target.global_position.x - agent.global_position.x)
		AgentSide.CLOSEST :
			dir = -signf(target.global_position.x - agent.global_position.x)
		AgentSide.BACK:
			dir = -target.get_facing()
		AgentSide.FRONT:
			dir = target.get_facing()

	var flank_pos: Vector2
	var offset := Vector2(dir * randf_range(range_min, range_max), 0.0)
	flank_pos = target.global_position + offset
	if not agent.is_good_position(flank_pos):
		# Choose the opposite side if the preferred side is not good (i.e., inside a collision shape).
		flank_pos = target.global_position - offset
	blackboard.set_var(position_var, flank_pos)
	return SUCCESS

