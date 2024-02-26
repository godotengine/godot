@tool
extends MeshInstance3D


func _on_visibility_changed():
	if visible:
		$anim.play("rotate")
	else:
		$anim.stop()
