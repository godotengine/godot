extends Area3D


func _ready() -> void:
	body_entered.connect(_on_body_entered)
	body_exited.connect(_on_body_exited)


func _on_body_entered(body: Node3D) -> void:
	if body.name == "Player":
		var env: WorldEnvironment = get_node_or_null("../../World/Environment")
		if env:
			var tween: Tween = get_tree().create_tween()
			tween.tween_property(env.environment, "ambient_light_energy", .1, .33)
	

func _on_body_exited(body: Node3D) -> void:
	if body.name == "Player":
		var env: WorldEnvironment = get_node_or_null("../../World/Environment")
		if env:
			var tween: Tween = get_tree().create_tween()
			tween.tween_property(env.environment, "ambient_light_energy", 1., .33)
