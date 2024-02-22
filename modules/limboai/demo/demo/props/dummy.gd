extends CharacterBody2D

@onready var animation_player: AnimationPlayer = $AnimationPlayer
@onready var hurtbox: Hurtbox = $Hurtbox
@onready var root: Node2D = $Root


func _on_health_damaged(_amount: float, _knockback: Vector2) -> void:
	root.scale.x = -signf(hurtbox.last_attack_vector.x)
	animation_player.clear_queue()
	animation_player.play(&"hurt", 0.1)


func get_facing() -> float:
	return signf(root.scale.x)
