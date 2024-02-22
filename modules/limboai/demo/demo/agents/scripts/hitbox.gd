#*
#* hitbox.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
class_name Hitbox
extends Area2D
## Area that deals damage.

## Damage value to apply.
@export var damage: float = 1.0

## Push back the victim.
@export var knockback_enabled: bool = false

## Desired pushback speed.
@export var knockback_strength: float = 500.0


func _ready() -> void:
	area_entered.connect(_area_entered)


func _area_entered(hurtbox: Hurtbox) -> void:
	if hurtbox.owner == owner:
		return
	hurtbox.take_damage(damage, get_knockback(), self)


func get_knockback() -> Vector2:
	var knockback: Vector2
	if knockback_enabled:
		knockback = Vector2.RIGHT.rotated(global_rotation) * knockback_strength
	return knockback
