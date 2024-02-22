#*
#* dodge_state.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
extends LimboState
## Dodge state.


@export var animation_player: AnimationPlayer
@export var animation: StringName
@export var duration: float = 0.4
@export var dodge_speed: float = 1000.0
@export var hurtbox_collision: CollisionShape2D

var move_dir: Vector2
var elapsed_time: float


func _enter() -> void:
	elapsed_time = 0.0
	hurtbox_collision.disabled = true

	var horizontal_move: float = Input.get_axis(&"move_left", &"move_right")
	if is_zero_approx(horizontal_move):
		move_dir = Vector2.RIGHT * agent.get_facing()
	else:
		move_dir = Vector2.RIGHT * signf(horizontal_move)
	agent.face_dir(move_dir.x)

	animation_player.play(animation, 0.1)


func _exit() -> void:
	hurtbox_collision.set_deferred(&"disabled", false)


func _update(p_delta: float) -> void:
	elapsed_time += p_delta
	var desired_velocity: Vector2 = move_dir * dodge_speed
	agent.move(desired_velocity)
	if elapsed_time > duration:
		get_root().dispatch(EVENT_FINISHED)
