extends Node

const BakeLodDialog: PackedScene = preload("res://addons/terrain_3d/src/bake_lod_dialog.tscn")
const BAKE_MESH_DESCRIPTION: String = "This will create a child MeshInstance3D. LOD4+ is recommended. LOD0 is slow and dense with vertices every 1 unit. It is not an optimal mesh."
const BAKE_OCCLUDER_DESCRIPTION: String = "This will create a child OccluderInstance3D. LOD4+ is recommended and will take 5+ seconds per region to generate. LOD0 is unnecessarily dense and slow."
const SET_UP_NAVIGATION_DESCRIPTION: String = "This operation will:

- Create a NavigationRegion3D node,
- Assign it a blank NavigationMesh resource,
- Move the Terrain3D node to be a child of the new node,
- And bake the nav mesh.

Once setup is complete, you can modify the settings on your nav mesh, and rebake
without having to run through the setup again.

If preferred, this setup can be canceled and the steps performed manually. For
the best results, adjust the settings on the NavigationMesh resource to match
the settings of your navigation agents and collisions."

var plugin: EditorPlugin
var bake_method: Callable
var bake_lod_dialog: ConfirmationDialog
var confirm_dialog: ConfirmationDialog


func _enter_tree() -> void:
	bake_lod_dialog = BakeLodDialog.instantiate()
	bake_lod_dialog.hide()
	bake_lod_dialog.confirmed.connect(func(): bake_method.call())
	bake_lod_dialog.set_unparent_when_invisible(true)
	
	confirm_dialog = ConfirmationDialog.new()
	confirm_dialog.hide()
	confirm_dialog.confirmed.connect(func(): bake_method.call())
	confirm_dialog.set_unparent_when_invisible(true)


func _exit_tree() -> void:
	bake_lod_dialog.queue_free()
	confirm_dialog.queue_free()


func bake_mesh_popup() -> void:
	if plugin.terrain:
		bake_method = _bake_mesh
		bake_lod_dialog.description = BAKE_MESH_DESCRIPTION
		plugin.get_editor_interface().popup_dialog_centered(bake_lod_dialog)


func _bake_mesh() -> void:
	var mesh: Mesh = plugin.terrain.bake_mesh(bake_lod_dialog.lod, Terrain3DStorage.HEIGHT_FILTER_NEAREST)
	if !mesh:
		push_error("Failed to bake mesh from Terrain3D")
		return

	var undo: EditorUndoRedoManager = plugin.get_undo_redo()
	undo.create_action("Terrain3D Bake ArrayMesh")
	
	var mesh_instance := plugin.terrain.get_node_or_null(^"MeshInstance3D") as MeshInstance3D
	if !mesh_instance:
		mesh_instance = MeshInstance3D.new()
		mesh_instance.name = &"MeshInstance3D"
		mesh_instance.set_skeleton_path(NodePath())
		mesh_instance.mesh = mesh
		
		undo.add_do_method(plugin.terrain, &"add_child", mesh_instance, true)
		undo.add_undo_method(plugin.terrain, &"remove_child", mesh_instance)
		undo.add_do_property(mesh_instance, &"owner", plugin.terrain.owner)
		undo.add_do_reference(mesh_instance)
	
	else:
		undo.add_do_property(mesh_instance, &"mesh", mesh)
		undo.add_undo_property(mesh_instance, &"mesh", mesh_instance.mesh)
		
		if mesh_instance.mesh.resource_path:
			var path := mesh_instance.mesh.resource_path
			undo.add_do_method(mesh, &"take_over_path", path)
			undo.add_undo_method(mesh_instance.mesh, &"take_over_path", path)
			undo.add_do_method(ResourceSaver, &"save", mesh)
			undo.add_undo_method(ResourceSaver, &"save", mesh_instance.mesh)
	
	undo.commit_action()


func bake_occluder_popup() -> void:
	if plugin.terrain:
		bake_method = _bake_occluder
		bake_lod_dialog.description = BAKE_OCCLUDER_DESCRIPTION
		plugin.get_editor_interface().popup_dialog_centered(bake_lod_dialog)


func _bake_occluder() -> void:
	var mesh: Mesh = plugin.terrain.bake_mesh(bake_lod_dialog.lod, Terrain3DStorage.HEIGHT_FILTER_MINIMUM)
	if !mesh:
		push_error("Failed to bake mesh from Terrain3D")
		return
	assert(mesh.get_surface_count() == 1)

	var undo: EditorUndoRedoManager = plugin.get_undo_redo()
	undo.create_action("Terrain3D Bake Occluder3D")

	var occluder := ArrayOccluder3D.new()
	var arrays: Array = mesh.surface_get_arrays(0)
	assert(arrays.size() > Mesh.ARRAY_INDEX)
	assert(arrays[Mesh.ARRAY_INDEX] != null)
	occluder.set_arrays(arrays[Mesh.ARRAY_VERTEX], arrays[Mesh.ARRAY_INDEX])

	var occluder_instance := plugin.terrain.get_node_or_null(^"OccluderInstance3D") as OccluderInstance3D
	if !occluder_instance:
		occluder_instance = OccluderInstance3D.new()
		occluder_instance.name = &"OccluderInstance3D"
		occluder_instance.occluder = occluder

		undo.add_do_method(plugin.terrain, &"add_child", occluder_instance, true)
		undo.add_undo_method(plugin.terrain, &"remove_child", occluder_instance)
		undo.add_do_property(occluder_instance, &"owner", plugin.terrain.owner)
		undo.add_do_reference(occluder_instance)
	
	else:
		undo.add_do_property(occluder_instance, &"occluder", occluder)
		undo.add_undo_property(occluder_instance, &"occluder", occluder_instance.occluder)
		
		if occluder_instance.occluder.resource_path:
			var path := occluder_instance.occluder.resource_path
			undo.add_do_method(occluder, &"take_over_path", path)
			undo.add_undo_method(occluder_instance.occluder, &"take_over_path", path)
			undo.add_do_method(ResourceSaver, &"save", occluder)
			undo.add_undo_method(ResourceSaver, &"save", occluder_instance.occluder)
	
	undo.commit_action()


func find_nav_region_terrains(p_nav_region: NavigationRegion3D) -> Array[Terrain3D]:
	var result: Array[Terrain3D] = []
	if not p_nav_region.navigation_mesh:
		return result
	
	var source_mode: NavigationMesh.SourceGeometryMode
	source_mode = p_nav_region.navigation_mesh.geometry_source_geometry_mode
	if source_mode == NavigationMesh.SOURCE_GEOMETRY_ROOT_NODE_CHILDREN:
		result.append_array(p_nav_region.find_children("", "Terrain3D", true, true))
		return result
	
	var group_nodes: Array = p_nav_region.get_tree().get_nodes_in_group(p_nav_region.navigation_mesh.geometry_source_group_name)
	for node in group_nodes:
		if node is Terrain3D:
			result.push_back(node)
		if source_mode == NavigationMesh.SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN:
			result.append_array(node.find_children("", "Terrain3D", true, true))
	
	return result


func find_terrain_nav_regions(p_terrain: Terrain3D) -> Array[NavigationRegion3D]:
	var result: Array[NavigationRegion3D] = []
	var root: Node = plugin.get_editor_interface().get_edited_scene_root()
	if not root:
		return result
	for nav_region in root.find_children("", "NavigationRegion3D", true, true):
		if find_nav_region_terrains(nav_region).has(p_terrain):
			result.push_back(nav_region)
	return result


func bake_nav_mesh() -> void:
	if plugin.nav_region:
		# A NavigationRegion3D is selected. We only need to bake that one navmesh.
		_bake_nav_region_nav_mesh(plugin.nav_region)
		print("Terrain3DNavigation: Finished baking 1 NavigationMesh.")
	
	elif plugin.terrain:
		# A Terrain3D is selected. There are potentially multiple navmeshes to bake and we need to
		# find them all. (The multiple navmesh use-case is likely on very large scenes with lots of
		# geometry. Each navmesh in this case would define its own, non-overlapping, baking AABB, to
		# cut down on the amount of geometry to bake. In a large open-world RPG, for instance, there
		# could be a navmesh for each town.)
		var nav_regions: Array[NavigationRegion3D] = find_terrain_nav_regions(plugin.terrain)
		for nav_region in nav_regions:
			_bake_nav_region_nav_mesh(nav_region)
		print("Terrain3DNavigation: Finished baking %d NavigationMesh(es)." % nav_regions.size())


func _bake_nav_region_nav_mesh(p_nav_region: NavigationRegion3D) -> void:
	var nav_mesh: NavigationMesh = p_nav_region.navigation_mesh
	assert(nav_mesh != null)
	
	var source_geometry_data := NavigationMeshSourceGeometryData3D.new()
	NavigationMeshGenerator.parse_source_geometry_data(nav_mesh, source_geometry_data, p_nav_region)
	
	for terrain in find_nav_region_terrains(p_nav_region):
		var aabb: AABB = nav_mesh.filter_baking_aabb
		aabb.position += nav_mesh.filter_baking_aabb_offset
		aabb = p_nav_region.global_transform * aabb
		var faces: PackedVector3Array = terrain.generate_nav_mesh_source_geometry(aabb)
		if not faces.is_empty():
			source_geometry_data.add_faces(faces, Transform3D.IDENTITY)
	
	NavigationMeshGenerator.bake_from_source_geometry_data(nav_mesh, source_geometry_data)
	
	_postprocess_nav_mesh(nav_mesh)
	
	# Assign null first to force the debug display to actually update:
	p_nav_region.set_navigation_mesh(null)
	p_nav_region.set_navigation_mesh(nav_mesh)
	
	# Trigger save to disk if it is saved as an external file
	if not nav_mesh.get_path().is_empty():
		ResourceSaver.save(nav_mesh, nav_mesh.get_path(), ResourceSaver.FLAG_COMPRESS)
	
	# Let other editor plugins and tool scripts know the nav mesh was just baked:
	p_nav_region.bake_finished.emit()


func _postprocess_nav_mesh(p_nav_mesh: NavigationMesh) -> void:
	# Post-process the nav mesh to work around Godot issue #85548
	
	# Round all the vertices in the nav_mesh to the nearest cell_size/cell_height so that it doesn't
	# contain any edges shorter than cell_size/cell_height (one cause of #85548).
	var vertices: PackedVector3Array = _postprocess_nav_mesh_round_vertices(p_nav_mesh)
	
	# Rounding vertices can collapse some edges to 0 length. We remove these edges, and any polygons
	# that have been reduced to 0 area.
	var polygons: Array[PackedInt32Array] = _postprocess_nav_mesh_remove_empty_polygons(p_nav_mesh, vertices)
	
	# Another cause of #85548 is baking producing overlapping polygons. We remove these.
	_postprocess_nav_mesh_remove_overlapping_polygons(p_nav_mesh, vertices, polygons)
	
	p_nav_mesh.clear_polygons()
	p_nav_mesh.set_vertices(vertices)
	for polygon in polygons:
		p_nav_mesh.add_polygon(polygon)


func _postprocess_nav_mesh_round_vertices(p_nav_mesh: NavigationMesh) -> PackedVector3Array:
	assert(p_nav_mesh != null)
	assert(p_nav_mesh.cell_size > 0.0)
	assert(p_nav_mesh.cell_height > 0.0)
	
	var cell_size: Vector3 = Vector3(p_nav_mesh.cell_size, p_nav_mesh.cell_height, p_nav_mesh.cell_size)
	
	# Round a little harder to avoid rounding errors with non-power-of-two cell_size/cell_height
	# causing the navigation map to put two non-matching edges in the same cell:
	var round_factor := cell_size * 1.001
	
	var vertices: PackedVector3Array = p_nav_mesh.get_vertices()
	for i in range(vertices.size()):
		vertices[i] = (vertices[i] / round_factor).floor() * round_factor
	return vertices


func _postprocess_nav_mesh_remove_empty_polygons(p_nav_mesh: NavigationMesh, p_vertices: PackedVector3Array) -> Array[PackedInt32Array]:
	var polygons: Array[PackedInt32Array] = []
	
	for i in range(p_nav_mesh.get_polygon_count()):
		var old_polygon: PackedInt32Array = p_nav_mesh.get_polygon(i)
		var new_polygon: PackedInt32Array = []
		
		# Remove duplicate vertices (introduced by rounding) from the polygon:
		var polygon_vertices: PackedVector3Array = []
		for index in old_polygon:
			var vertex: Vector3 = p_vertices[index]
			if polygon_vertices.has(vertex):
				continue
			polygon_vertices.push_back(vertex)
			new_polygon.push_back(index)
		
		# If we removed some vertices, we might be able to remove the polygon too:
		if new_polygon.size() <= 2:
			continue
		polygons.push_back(new_polygon)
		
	return polygons


func _postprocess_nav_mesh_remove_overlapping_polygons(p_nav_mesh: NavigationMesh, p_vertices: PackedVector3Array, p_polygons: Array[PackedInt32Array]) -> void:
	# Occasionally, a baked nav mesh comes out with overlapping polygons:
	# https://github.com/godotengine/godot/issues/85548#issuecomment-1839341071
	# Until the bug is fixed in the engine, this function attempts to detect and remove overlapping
	# polygons.
	
	# This function has to make a choice of which polygon to remove when an overlap is detected,
	# because in this case the nav mesh is ambiguous. To do this it uses a heuristic:
	# (1) an 'overlap' is defined as an edge that is shared by 3 or more polygons.
	# (2) a 'bad polygon' is defined as a polygon that contains 2 or more 'overlaps'.
	# The function removes the 'bad polygons', which in practice seems to be enough to remove all
	# overlaps without creating holes in the nav mesh.
	
	var cell_size: Vector3 = Vector3(p_nav_mesh.cell_size, p_nav_mesh.cell_height, p_nav_mesh.cell_size)
	
	# `edges` is going to map edges (vertex pairs) to arrays of polygons that contain that edge.
	var edges: Dictionary = {}
	
	for polygon_index in range(p_polygons.size()):
		var polygon: PackedInt32Array = p_polygons[polygon_index]
		for j in range(polygon.size()):
			var vertex: Vector3 = p_vertices[polygon[j]]
			var next_vertex: Vector3 = p_vertices[polygon[(j + 1) % polygon.size()]]
			
			# edge_key is a key we can use in the edges dictionary that uniquely identifies the
			# edge. We use cell coordinates here (Vector3i) because with a non-power-of-two
			# cell_size, rounding errors can cause Vector3 vertices to not be equal.
			# Array.sort IS defined for vector types - see the Godot docs. It's necessary here
			# because polygons that share an edge can have their vertices in a different order.
			var edge_key: Array = [Vector3i(vertex / cell_size), Vector3i(next_vertex / cell_size)]
			edge_key.sort()
			
			if !edges.has(edge_key):
				edges[edge_key] = []
			edges[edge_key].push_back(polygon_index)
	
	var overlap_count: Dictionary = {}
	for connections in edges.values():
		if connections.size() <= 2:
			continue
		for polygon_index in connections:
			overlap_count[polygon_index] = overlap_count.get(polygon_index, 0) + 1
	
	var bad_polygons: Array = []
	for polygon_index in overlap_count.keys():
		if overlap_count[polygon_index] >= 2:
			bad_polygons.push_back(polygon_index)
	
	bad_polygons.sort()
	for i in range(bad_polygons.size() - 1, -1, -1):
		p_polygons.remove_at(bad_polygons[i])


func set_up_navigation_popup() -> void:
	if plugin.terrain:
		bake_method = _set_up_navigation
		confirm_dialog.dialog_text = SET_UP_NAVIGATION_DESCRIPTION
		plugin.get_editor_interface().popup_dialog_centered(confirm_dialog)


func _set_up_navigation() -> void:
	assert(plugin.terrain)
	var terrain: Terrain3D = plugin.terrain
	
	var nav_region := NavigationRegion3D.new()
	nav_region.name = &"NavigationRegion3D"
	nav_region.navigation_mesh = NavigationMesh.new()
	
	var undo_redo: EditorUndoRedoManager = plugin.get_undo_redo()
	
	undo_redo.create_action("Terrain3D Set up Navigation")
	undo_redo.add_do_method(self, &"_do_set_up_navigation", nav_region, terrain)
	undo_redo.add_undo_method(self, &"_undo_set_up_navigation", nav_region, terrain)
	undo_redo.add_do_reference(nav_region)
	undo_redo.commit_action()

	plugin.get_editor_interface().inspect_object(nav_region)
	assert(plugin.nav_region == nav_region)
	
	bake_nav_mesh()


func _do_set_up_navigation(p_nav_region: NavigationRegion3D, p_terrain: Terrain3D) -> void:
	var parent: Node = p_terrain.get_parent()
	var index: int = p_terrain.get_index()
	var t_owner: Node = p_terrain.owner
	
	parent.remove_child(p_terrain)
	p_nav_region.add_child(p_terrain)
	
	parent.add_child(p_nav_region, true)
	parent.move_child(p_nav_region, index)
	
	p_nav_region.owner = t_owner
	p_terrain.owner = t_owner


func _undo_set_up_navigation(p_nav_region: NavigationRegion3D, p_terrain: Terrain3D) -> void:
	assert(p_terrain.get_parent() == p_nav_region)
	
	var parent: Node = p_nav_region.get_parent()
	var index: int = p_nav_region.get_index()
	var t_owner: Node = p_nav_region.get_owner()
	
	parent.remove_child(p_nav_region)
	p_nav_region.remove_child(p_terrain)
	
	parent.add_child(p_terrain, true)
	parent.move_child(p_terrain, index)
	
	p_terrain.owner = t_owner
