#*
#* health.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
class_name Health
extends Node
## Tracks health and emits signal when damaged or dead.

## Emitted when health is reduced to 0.
signal death

## Emitted when health is damaged.
signal damaged(amount: float, knockback: Vector2)

## Initial health value.
@export var max_health: float = 10.0

var _current: float


func _ready() -> void:
	_current = max_health


func take_damage(amount: float, knockback: Vector2) -> void:
	if _current <= 0.0:
		return

	_current -= amount
	_current = max(_current, 0.0)

	if _current <= 0.0:
		death.emit()
	else:
		damaged.emit(amount, knockback)


## Returns current health.
func get_current() -> float:
	return _current
