#*
#* move_state.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
extends LimboState
## Move state.


const VERTICAL_FACTOR := 0.8

@export var animation_player: AnimationPlayer
@export var animation: StringName
@export var speed: float = 500.0


func _enter() -> void:
	animation_player.play(animation, 0.1)


func _update(_delta: float) -> void:
	var horizontal_move: float = Input.get_axis(&"move_left", &"move_right")
	var vertical_move: float = Input.get_axis(&"move_up", &"move_down")

	if not is_zero_approx(horizontal_move):
		agent.face_dir(horizontal_move)

	var desired_velocity := Vector2(horizontal_move, vertical_move * VERTICAL_FACTOR) * speed
	agent.move(desired_velocity)

	if horizontal_move == 0.0 and vertical_move == 0.0:
		get_root().dispatch(EVENT_FINISHED)
