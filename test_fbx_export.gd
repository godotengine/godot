@tool
extends EditorScript

## Test script to verify FBX export doesn't crash
## Run this from the Editor -> Run Script menu

func _run():
	print("=== FBX Export Test ===")
	
	# Create a simple scene with a cube mesh
	var root = Node3D.new()
	root.name = "TestRoot"
	
	var mesh_instance = MeshInstance3D.new()
	mesh_instance.name = "Cube"
	
	# Create a box mesh (cube)
	var box_mesh = BoxMesh.new()
	box_mesh.size = Vector3(1, 1, 1)
	mesh_instance.mesh = box_mesh
	
	root.add_child(mesh_instance)
	mesh_instance.owner = root
	
	# Try to export to FBX
	var export_path = OS.get_user_data_dir() + "/test_export.fbx"
	print("Exporting to: ", export_path)
	
	var fbx_doc = FBXDocument.new()
	var gltf_state = GLTFState.new()
	
	# Set binary format (default)
	fbx_doc.export_format = 0
	
	print("Appending scene to FBX state...")
	var err_append = fbx_doc.append_from_scene(root, gltf_state)
	if err_append != OK:
		print("ERROR: append_from_scene failed with error: ", err_append)
		root.queue_free()
		return
	
	print("Writing to filesystem...")
	var err_export = fbx_doc.write_to_filesystem(gltf_state, export_path)
	
	if err_export == OK:
		print("SUCCESS: FBX exported successfully!")
		print("File exists: ", FileAccess.file_exists(export_path))
		var file = FileAccess.open(export_path, FileAccess.READ)
		if file:
			print("File size: ", file.get_length(), " bytes")
			file.close()
	elif err_export == ERR_FILE_CANT_WRITE:
		print("EXPECTED FAILURE: FBX export failed gracefully (no crash!)")
		print("This is expected with current ASAN parameter corruption issue.")
	else:
		print("ERROR: FBX export failed with error: ", err_export)
	
	# Cleanup
	root.queue_free()
	print("=== Test Complete ===")

