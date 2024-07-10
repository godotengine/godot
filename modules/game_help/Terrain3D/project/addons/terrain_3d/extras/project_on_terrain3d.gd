# This script is an addon for HungryProton's Scatter https://github.com/HungryProton/scatter
# It provides a `Project on Terrain3D` modifier, which allows Scatter 
# to detect the terrain height from Terrain3D without using collision.
# Copy this file into /addons/proton_scatter/src/modifiers
# Then uncomment everything below
# In the editor, add this modifier to Scatter, then set your Terrain3D node

# This script is an addon for HungryProton's Scatter https://github.com/HungryProton/scatter
# It allows Scatter to detect the terrain height from Terrain3D
# Copy this file into /addons/proton_scatter/src/modifiers
# Then uncomment everything below (select, press CTRL+K)
# In the editor, add this modifier, then set your Terrain3D node

#@tool
#extends "base_modifier.gd"
#
#
#signal projection_completed
#
#
#@export var terrain_node : NodePath
#@export var align_with_collision_normal := false
#
#var _terrain: Terrain3D
#
#
#func _init() -> void:
	#display_name = "Project On Terrain3D"
	#category = "Edit"
	#can_restrict_height = false
	#global_reference_frame_available = true
	#local_reference_frame_available = true
	#individual_instances_reference_frame_available = true
	#use_global_space_by_default()
#
	#documentation.add_paragraph(
		#"This is a duplicate of `Project on Colliders` that queries the terrain system
		#for height and sets the transform height appropriately.
#
		#This modifier must have terrain_node set to a Terrain3D node.")
#
	#var p := documentation.add_parameter("Terrain Node")
	#p.set_type("NodePath")
	#p.set_description("Set your Terrain3D node.")
		#
	#p = documentation.add_parameter("Align with collision normal")
	#p.set_type("bool")
	#p.set_description(
		#"Rotate the transform to align it with the collision normal in case
		#the ray cast hit a collider.")
#
#
#func _process_transforms(transforms, domain, _seed) -> void:
	#if transforms.is_empty():
		#return
#
	#if terrain_node:
		#_terrain = domain.get_root().get_node_or_null(terrain_node)
#
	#if not _terrain:
		#warning += """No Terrain3D node found"""
		#return
#
	#if not _terrain.storage:
		#warning += """Terrain3D storage is not initialized"""
		#return
#
	## Get global transform
	#var gt: Transform3D = domain.get_global_transform()
	#var gt_inverse := gt.affine_inverse()
	#for i in transforms.list.size():
		#var location: Vector3 = (gt * transforms.list[i]).origin
		#var height: float = _terrain.storage.get_height(location)
		#var normal: Vector3 = _terrain.storage.get_normal(location)
		#
		#if align_with_collision_normal and not is_nan(normal.x):
			#transforms.list[i].basis.y = normal
			#transforms.list[i].basis.x = -transforms.list[i].basis.z.cross(normal)
			#transforms.list[i].basis = transforms.list[i].basis.orthonormalized()
#
		#transforms.list[i].origin.y = gt.origin.y if is_nan(height) else height - gt.origin.y
#
	#if transforms.is_empty():
		#warning += """Every point has been removed. Possible reasons include: \n
		#+ No collider is close enough to the shapes.
		#+ Ray length is too short.
		#+ Ray direction is incorrect.
		#+ Collision mask is not set properly.
		#+ Max slope is too low.
		#"""
