# meta-description: Basic import script template
@tool
extends EditorScenePostImport


# Called by the editor when a scene has this script set as the import script in the import tab.
func _post_import(scene: Node) -> Object:
	# Modify the contents of the scene upon import. For example, setting up LODs:
#	(scene.get_node(^"HighPolyMesh") as MeshInstance3D).draw_distance_end = 5.0
#	(scene.get_node(^"LowPolyMesh") as MeshInstance3D).draw_distance_begin = 5.0
	return scene # Return the modified root node when you're done.
