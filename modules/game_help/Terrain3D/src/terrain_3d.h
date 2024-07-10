// Copyright © 2024 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_CLASS_H
#define TERRAIN3D_CLASS_H

#ifndef __FLT_MAX__
#define __FLT_MAX__ FLT_MAX
#endif

//#include <godot_cpp/classes/camera3d.hpp>
//#include <godot_cpp/classes/editor_plugin.hpp>
//#include <godot_cpp/classes/geometry_instance3d.hpp>
//#include <godot_cpp/classes/mesh.hpp>
//#include <godot_cpp/classes/static_body3d.hpp>

#include "scene/3d/camera_3d.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/resources/mesh.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/3d/physics/physics_body_3d.h"
#include "scene/3d/physics/static_body_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/viewport.h"


#include "terrain_3d_material.h"
#include "terrain_3d_storage.h"
#include "terrain_3d_assets.h"
#include "terrain_3d_instancer.h"

using namespace godot;

class Terrain3D : public Node3D {
	GDCLASS(Terrain3D, Node3D);
	CLASS_NAME();

	// Terrain state
	String _version = "0.9.3-dev";
	bool _is_inside_world = false;
	bool _initialized = false;

	// Terrain settings
	int _mesh_size = 48;
	int _mesh_lods = 7;
	real_t _mesh_vertex_spacing = 1.0f;

	Ref<Terrain3DMaterial> _material;
	Ref<Terrain3DStorage> _storage;
	Ref<Terrain3DAssets> _assets;
	Terrain3DInstancer *_instancer = nullptr;

	// Editor components
	EditorPlugin *_plugin = nullptr;
	// Current editor or gameplay camera we are centering the terrain on.
	Camera3D *_camera = nullptr;
	uint64_t _camera_instance_id = 0;

	// X,Z Position of the camera during the previous snapping. Set to max real_t value to force a snap update.
	Vector2 _camera_last_position = Vector2(__FLT_MAX__, __FLT_MAX__);

	// Meshes and Mesh instances
	Vector<RID> _meshes;
	struct Instances {
		RID cross;
		Vector<RID> tiles;
		Vector<RID> fillers;
		Vector<RID> trims;
		Vector<RID> seams;
	} _data;

	// Renderer settings
	uint32_t _render_layers = 1 | (1 << 31); // Bit 1 and 32 for the cursor
	GeometryInstance3D::ShadowCastingSetting _cast_shadows = GeometryInstance3D::SHADOW_CASTING_SETTING_ON;
	real_t _cull_margin = 0.0f;

	// Mouse cursor
	SubViewport *_mouse_vp = nullptr;
	Camera3D *_mouse_cam = nullptr;
	MeshInstance3D *_mouse_quad = nullptr;
	uint32_t _mouse_layer = 32;

	// Physics body and settings
	RID _static_body;
	StaticBody3D *_debug_static_body = nullptr;
	bool _collision_enabled = true;
	bool _show_debug_collision = false;
	uint32_t _collision_layer = 1;
	uint32_t _collision_mask = 1;
	real_t _collision_priority = 1.0f;

	void _initialize();
	void __ready();
	void __process(const double p_delta);

	void _setup_mouse_picking();
	void _destroy_mouse_picking();
	void _grab_camera();

	void _build_meshes(const int p_mesh_lods, const int p_mesh_size);
	void _update_mesh_instances();
	void _clear_meshes();

	void _build_collision();
	void _update_collision();
	void _destroy_collision();

	void _destroy_instancer();

	void _generate_triangles(PackedVector3Array &p_vertices, PackedVector2Array *p_uvs, const int32_t p_lod,
			const Terrain3DStorage::HeightFilter p_filter, const bool require_nav, const AABB &p_global_aabb) const;
	void _generate_triangle_pair(PackedVector3Array &p_vertices, PackedVector2Array *p_uvs, const int32_t p_lod,
			const Terrain3DStorage::HeightFilter p_filter, const bool require_nav, const int32_t x, const int32_t z) const;

public:
	static int debug_level;

	Terrain3D();
	~Terrain3D();

	// Terrain settings
	String get_version() const { return _version; }
	void set_debug_level(const int p_level);
	int get_debug_level() const { return debug_level; }
	void set_mesh_lods(const int p_count);
	int get_mesh_lods() const { return _mesh_lods; }
	void set_mesh_size(const int p_size);
	int get_mesh_size() const { return _mesh_size; }
	void set_mesh_vertex_spacing(const real_t p_spacing);
	real_t get_mesh_vertex_spacing() const { return _mesh_vertex_spacing; }

	void set_material(const Ref<Terrain3DMaterial> &p_material);
	Ref<Terrain3DMaterial> get_material() const { return _material; }
	void set_storage(const Ref<Terrain3DStorage> &p_storage);
	Ref<Terrain3DStorage> get_storage() const { return _storage; }
	void set_assets(const Ref<Terrain3DAssets> &p_assets);
	Ref<Terrain3DAssets> get_assets() const { return _assets; }

	// Instancer
	Terrain3DInstancer *get_instancer() const { return _instancer; }

	// Editor components
	void set_plugin(EditorPlugin *p_plugin);
	EditorPlugin *get_plugin() const { return _plugin; }
	void set_camera(Camera3D *p_plugin);
	Camera3D *get_camera() const { return _camera; }

	// Renderer settings
	void set_render_layers(const uint32_t p_layers);
	uint32_t get_render_layers() const { return _render_layers; };
	void set_mouse_layer(const uint32_t p_layer);
	uint32_t get_mouse_layer() const { return _mouse_layer; };
	void set_cast_shadows(const GeometryInstance3D::ShadowCastingSetting p_cast_shadows);
	GeometryInstance3D::ShadowCastingSetting get_cast_shadows() const { return _cast_shadows; };
	void set_cull_margin(const real_t p_margin);
	real_t get_cull_margin() const { return _cull_margin; };

	// Physics body settings
	void set_collision_enabled(const bool p_enabled);
	bool get_collision_enabled() const { return _collision_enabled; }
	void set_show_debug_collision(const bool p_enabled);
	bool get_show_debug_collision() const { return _show_debug_collision; }
	void set_collision_layer(const uint32_t p_layers);
	uint32_t get_collision_layer() const { return _collision_layer; };
	void set_collision_mask(const uint32_t p_mask);
	uint32_t get_collision_mask() const { return _collision_mask; };
	void set_collision_priority(const real_t p_priority);
	real_t get_collision_priority() const { return _collision_priority; }

	// Terrain methods
	void snap(const Vector3 &p_cam_pos);
	void update_aabbs();
	Vector3 get_intersection(const Vector3 &p_src_pos, const Vector3 &p_direction);

	// Baking methods
	Ref<Mesh> bake_mesh(const int p_lod, const Terrain3DStorage::HeightFilter p_filter = Terrain3DStorage::HEIGHT_FILTER_NEAREST) const;
	PackedVector3Array generate_nav_mesh_source_geometry(const AABB &p_global_aabb, const bool p_require_nav = true) const;

	// Misc
	PackedStringArray get_configuration_warnings() const override;

	// DEPRECATED 0.9.2 - Remove 0.9.3+
	void set_texture_list(const Ref<Terrain3DTextureList> &p_texture_list);
	Ref<Terrain3DTextureList> get_texture_list() const { return Ref<Terrain3DTextureList>(); }

protected:
	void _notification(const int p_what);
	static void _bind_methods();
};

#endif // TERRAIN3D_CLASS_H
