#*
#* ninja_star.gd
#* =============================================================================
#* Copyright 2021-2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*
extends Node2D

const SPEED := 800.0
const DEAD_SPEED := 400.0

@export var dir: float = 1.0

var _is_dead: bool = false

@onready var ninja_star: Sprite2D = $Root/NinjaStar
@onready var death: GPUParticles2D = $Death
@onready var collision_shape_2d: CollisionShape2D = $Hitbox/CollisionShape2D
@onready var root: Node2D = $Root


func _ready() -> void:
	var tween := create_tween().set_loops()
	tween.tween_property(ninja_star, ^"rotation", TAU * signf(dir), 1.0).as_relative()

	var tween2 := create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	tween2.tween_property(ninja_star, "position:y", -10.0, 0.5).as_relative().set_ease(Tween.EASE_OUT)
	tween2.tween_property(ninja_star, "position:y", 0.0, 1.0)
	tween2.tween_callback(_die)


func _physics_process(delta: float) -> void:
	var speed: float = SPEED if not _is_dead else DEAD_SPEED
	position += Vector2.RIGHT * speed * dir * delta


func _die() -> void:
	_is_dead = true
	root.hide()
	collision_shape_2d.set_deferred(&"disabled", true)
	death.emitting = true
	await death.finished
	queue_free()


func _on_hitbox_area_entered(_area: Area2D) -> void:
	_die()
