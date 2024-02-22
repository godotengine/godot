#*
#* idle_state.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
extends LimboState
## Idle state.


@export var animation_player: AnimationPlayer
@export var idle_animation: StringName


func _enter() -> void:
	animation_player.play(idle_animation, 0.1)


func _update(_delta: float) -> void:
	var horizontal_move: float = Input.get_axis(&"move_left", &"move_right")
	var vertical_move: float = Input.get_axis(&"move_up", &"move_down")
	if horizontal_move != 0.0 or vertical_move != 0.0:
		get_root().dispatch(EVENT_FINISHED)
