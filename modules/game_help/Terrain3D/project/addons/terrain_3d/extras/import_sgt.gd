## Import From SimpleGrassTextured
# 
# This script demonstrates how to import transforms from SimpleGrassTextured. To use it:
#
# 1. Assign this script to your Terrain3D node and reselect the node to update the inspector.
# 2. In the inspector select your SGT node.
# 3. Setup the desired mesh asset in the asset dock.
# 4. Assign that mesh id in the inspector.
# 5. Click Import.
#
# When finished, clear the script from your Terrain3DNode.
#
# Use clear_instances to erase all instances that match the assign_mesh_id.
#
# The add_transforms function (called by add_multimesh) applies the height_offset specified in the 
# Terrain3DMeshAsset. 
#
# The SimpleGrassTextured default mesh is a cross of two texture cards. The default Terrain3D 
# texture card is a single quadmesh, so assign your own mesh if you wish an exact match. Once the 
# transforms are imported, you can reassign any mesh you like into this mesh slot.

@tool
extends Terrain3D

@export var simple_grass_textured: MultiMeshInstance3D
@export var assign_mesh_id: int
@export var import: bool = false : set = import_sgt
@export var clear_instances: bool = false : set = clear_multimeshes


func clear_multimeshes(value: bool) -> void:
	get_instancer().clear_by_mesh(assign_mesh_id)


func import_sgt(value: bool) -> void:
	var sgt_mm: MultiMesh = simple_grass_textured.multimesh
	var global_xform: Transform3D = simple_grass_textured.global_transform	
	print("Starting to import %d instances from SimpleGrassTextured using mesh id %d" % [ sgt_mm.instance_count, assign_mesh_id])
	var time: int = Time.get_ticks_msec()
	get_instancer().add_multimesh(assign_mesh_id, sgt_mm, simple_grass_textured.global_transform)	
	print("Import complete in %.2f seconds" % [ float(Time.get_ticks_msec() - time)/1000. ])
	
