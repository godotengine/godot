extends Node3D

## Animates lights orbiting around the tunnel Z axis.

@export var orbit_radius: float = 4.0
@export var orbit_speed: float = 1.2

var _elapsed: float = 0.0

func _process(delta: float) -> void:
	_elapsed += delta
	var idx := 0
	for child in get_children():
		if child is OmniLight3D:
			var angle: float = _elapsed * orbit_speed + idx * PI * 0.5  # Offset each light by 90°.
			var base_z: float = child.position.z  # Keep Z from scene, only orbit XY.
			child.position.x = cos(angle) * orbit_radius
			child.position.y = sin(angle) * orbit_radius
			idx += 1
