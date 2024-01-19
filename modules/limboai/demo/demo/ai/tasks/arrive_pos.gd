#*
#* arrive_pos.gd
#* =============================================================================
#* Copyright 2021-2023 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*

@tool
@icon("res://icon.png")
extends BTAction

@export var target_position_var := "target_position"
@export var speed_var := "speed"
@export var tolerance := 50.0


@warning_ignore("native_method_override") # needed for GDExtension version.
func _generate_name() -> String:
	return "Arrive  pos: %s  speed: %s" % [
		LimboUtility.decorate_var(target_position_var),
		LimboUtility.decorate_var(speed_var),
	]

@warning_ignore("native_method_override")
func _tick(p_delta: float) -> Status:
	var target_pos: Vector2 = blackboard.get_var(target_position_var, Vector2.ZERO)
	if target_pos.distance_to(agent.global_position) < tolerance:
		return SUCCESS

	var speed: float = blackboard.get_var(speed_var, 10.0)
	var dir: Vector2 = agent.global_position.direction_to(target_pos)
	agent.global_position += dir * speed * p_delta
	return RUNNING
