extends StaticBody2D

signal gong_struck

var enabled: bool = true

@onready var animation_player: AnimationPlayer = $AnimationPlayer


func _on_health_damaged(_amount: float, _knockback: Vector2) -> void:
	if not enabled:
		return
	animation_player.play(&"struck")
	gong_struck.emit()
	enabled = false
