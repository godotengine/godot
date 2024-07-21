#*
#* move_forward.gd
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
## Applies velocity in the direction the agent is facing on each tick
## until the [member duration] is exceeded. [br]
## Returns [code]SUCCESS[/code] if the elapsed time exceeds [member duration]. [br]
## Returns [code]RUNNING[/code] if the elapsed time does not exceed [member duration]. [br]

## Blackboard variable that stores desired speed.
@export var speed_var: StringName = &"speed"

## How long to perform this task (in seconds).
@export var duration: float = 0.1

# Display a customized name (requires @tool).
func _generate_name() -> String:
	return "MoveForward  speed: %s  duration: %ss" % [
		LimboUtility.decorate_var(speed_var),
		duration]

# Called each time this task is ticked (aka executed).
func _tick(_delta: float) -> Status:
	var facing: float = agent.get_facing()
	var speed: float = blackboard.get_var(speed_var, 100.0)
	var desired_velocity: Vector2 = Vector2.RIGHT * facing * speed
	agent.move(desired_velocity)
	agent.update_facing()
	if elapsed_time > duration:
		return SUCCESS
	return RUNNING
