extends Node3D

# Spin all GaussianSplatNode3D children — isolates transform-update flickering
func _process(delta: float) -> void:
	for child in get_children():
		if child is GaussianSplatNode3D:
			child.rotate_y(delta * 0.5)
