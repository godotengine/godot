#*
#* attack_state.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
extends LimboState

## Attack state: Perform 3-part combo attack for as long as player hits attack button.

@export var animation_player: AnimationPlayer
@export var animations: Array[StringName]
@export var hitbox: Hitbox

## Cooldown duration after third attack in the combo is complete.
@export var combo_cooldown: float = 0.1

var anim_index: int = 0
var last_attack_msec: int = -10000
var _can_enter: bool = true


## This func is used to prevent entering this state using LimboState.set_guard().
## Entry is denied for a short duration after the third attack in the combo is complete.
func can_enter() -> bool:
	return _can_enter


func _enter() -> void:
	if (Time.get_ticks_msec() - last_attack_msec) < 200:
		# Perform next attack animation in the 3-part combo, if an attack was recently performed.
		anim_index = (anim_index + 1) % 3
	else:
		anim_index = 0

	var horizontal_move: float = Input.get_axis(&"move_left", &"move_right")
	if not is_zero_approx(horizontal_move):
		agent.face_dir(horizontal_move)

	hitbox.damage = 2 if anim_index == 2 else 1  # deal 2 damage on a third attack in the combo
	animation_player.play(animations[anim_index])

	await animation_player.animation_finished
	if is_active():
		get_root().dispatch(EVENT_FINISHED)


func _exit() -> void:
	hitbox.damage = 1
	last_attack_msec = Time.get_ticks_msec()
	if anim_index == 2 and _can_enter:
		# Prevent entering this state for a short duration after the third attack
		# in the combo sequence is complete.
		_can_enter = false
		await get_tree().create_timer(combo_cooldown).timeout
		_can_enter = true
