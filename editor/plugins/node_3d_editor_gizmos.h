/*************************************************************************/
/*  node_3d_editor_gizmos.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef NODE_3D_EDITOR_GIZMOS_H
#define NODE_3D_EDITOR_GIZMOS_H

#include "core/templates/local_vector.h"
#include "core/templates/ordered_hash_map.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"

class Timer;
class EditorNode3DGizmoPlugin;

class EditorNode3DGizmo : public Node3DGizmo {
	GDCLASS(EditorNode3DGizmo, Node3DGizmo);

	struct Instance {
		RID instance;
		Ref<Mesh> mesh;
		Ref<Material> material;
		Ref<SkinReference> skin_reference;
		bool extra_margin = false;
		Transform3D xform;

		void create_instance(Node3D *p_base, bool p_hidden = false);
	};

	bool selected;

	Vector<Vector3> collision_segments;
	Ref<TriangleMesh> collision_mesh;

	Vector<Vector3> handles;
	Vector<int> handle_ids;
	Vector<Vector3> secondary_handles;
	Vector<int> secondary_handle_ids;

	real_t selectable_icon_size;
	bool billboard_handle;

	bool valid;
	bool hidden;
	Vector<Instance> instances;
	Node3D *spatial_node;

	void _set_spatial_node(Node *p_node) { set_spatial_node(Object::cast_to<Node3D>(p_node)); }

protected:
	static void _bind_methods();

	EditorNode3DGizmoPlugin *gizmo_plugin;

	GDVIRTUAL0(_redraw)
	GDVIRTUAL2RC(String, _get_handle_name, int, bool)
	GDVIRTUAL2RC(bool, _is_handle_highlighted, int, bool)
	GDVIRTUAL2RC(Variant, _get_handle_value, int, bool)
	GDVIRTUAL4(_set_handle, int, bool, const Camera3D *, Vector2)
	GDVIRTUAL4(_commit_handle, int, bool, Variant, bool)

	GDVIRTUAL2RC(int, _subgizmos_intersect_ray, const Camera3D *, Vector2)
	GDVIRTUAL2RC(Vector<int>, _subgizmos_intersect_frustum, const Camera3D *, TypedArray<Plane>)
	GDVIRTUAL1RC(Transform3D, _get_subgizmo_transform, int)
	GDVIRTUAL2(_set_subgizmo_transform, int, Transform3D)
	GDVIRTUAL3(_commit_subgizmos, Vector<int>, TypedArray<Transform3D>, bool)
public:
	void add_lines(const Vector<Vector3> &p_lines, const Ref<Material> &p_material, bool p_billboard = false, const Color &p_modulate = Color(1, 1, 1));
	void add_vertices(const Vector<Vector3> &p_vertices, const Ref<Material> &p_material, Mesh::PrimitiveType p_primitive_type, bool p_billboard = false, const Color &p_modulate = Color(1, 1, 1));
	void add_mesh(const Ref<Mesh> &p_mesh, const Ref<Material> &p_material = Ref<Material>(), const Transform3D &p_xform = Transform3D(), const Ref<SkinReference> &p_skin_reference = Ref<SkinReference>());
	void add_collision_segments(const Vector<Vector3> &p_lines);
	void add_collision_triangles(const Ref<TriangleMesh> &p_tmesh);
	void add_unscaled_billboard(const Ref<Material> &p_material, real_t p_scale = 1, const Color &p_modulate = Color(1, 1, 1));
	void add_handles(const Vector<Vector3> &p_handles, const Ref<Material> &p_material, const Vector<int> &p_ids = Vector<int>(), bool p_billboard = false, bool p_secondary = false);
	void add_solid_box(Ref<Material> &p_material, Vector3 p_size, Vector3 p_position = Vector3(), const Transform3D &p_xform = Transform3D());

	virtual bool is_handle_highlighted(int p_id, bool p_secondary) const;
	virtual String get_handle_name(int p_id, bool p_secondary) const;
	virtual Variant get_handle_value(int p_id, bool p_secondary) const;
	virtual void set_handle(int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false);

	virtual int subgizmos_intersect_ray(Camera3D *p_camera, const Vector2 &p_point) const;
	virtual Vector<int> subgizmos_intersect_frustum(const Camera3D *p_camera, const Vector<Plane> &p_frustum) const;
	virtual Transform3D get_subgizmo_transform(int p_id) const;
	virtual void set_subgizmo_transform(int p_id, Transform3D p_transform);
	virtual void commit_subgizmos(const Vector<int> &p_ids, const Vector<Transform3D> &p_restore, bool p_cancel = false);

	void set_selected(bool p_selected) { selected = p_selected; }
	bool is_selected() const { return selected; }

	void set_spatial_node(Node3D *p_node);
	Node3D *get_spatial_node() const { return spatial_node; }
	Ref<EditorNode3DGizmoPlugin> get_plugin() const { return gizmo_plugin; }
	bool intersect_frustum(const Camera3D *p_camera, const Vector<Plane> &p_frustum);
	void handles_intersect_ray(Camera3D *p_camera, const Vector2 &p_point, bool p_shift_pressed, int &r_id, bool &r_secondary);
	bool intersect_ray(Camera3D *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal);
	bool is_subgizmo_selected(int p_id) const;
	Vector<int> get_subgizmo_selection() const;

	virtual void clear() override;
	virtual void create() override;
	virtual void transform() override;
	virtual void redraw() override;
	virtual void free() override;

	virtual bool is_editable() const;

	void set_hidden(bool p_hidden);
	void set_plugin(EditorNode3DGizmoPlugin *p_plugin);

	EditorNode3DGizmo();
	~EditorNode3DGizmo();
};

class EditorNode3DGizmoPlugin : public Resource {
	GDCLASS(EditorNode3DGizmoPlugin, Resource);

public:
	static const int VISIBLE = 0;
	static const int HIDDEN = 1;
	static const int ON_TOP = 2;

protected:
	int current_state;
	List<EditorNode3DGizmo *> current_gizmos;
	HashMap<String, Vector<Ref<StandardMaterial3D>>> materials;

	static void _bind_methods();
	virtual bool has_gizmo(Node3D *p_spatial);
	virtual Ref<EditorNode3DGizmo> create_gizmo(Node3D *p_spatial);

	GDVIRTUAL1RC(bool, _has_gizmo, Node3D *)
	GDVIRTUAL1RC(Ref<EditorNode3DGizmo>, _create_gizmo, Node3D *)

	GDVIRTUAL0RC(String, _get_gizmo_name)
	GDVIRTUAL0RC(int, _get_priority)
	GDVIRTUAL0RC(bool, _can_be_hidden)
	GDVIRTUAL0RC(bool, _is_selectable_when_hidden)

	GDVIRTUAL1(_redraw, Ref<EditorNode3DGizmo>)
	GDVIRTUAL3RC(String, _get_handle_name, Ref<EditorNode3DGizmo>, int, bool)
	GDVIRTUAL3RC(bool, _is_handle_highlighted, Ref<EditorNode3DGizmo>, int, bool)
	GDVIRTUAL3RC(Variant, _get_handle_value, Ref<EditorNode3DGizmo>, int, bool)

	GDVIRTUAL5(_set_handle, Ref<EditorNode3DGizmo>, int, bool, const Camera3D *, Vector2)
	GDVIRTUAL5(_commit_handle, Ref<EditorNode3DGizmo>, int, bool, Variant, bool)

	GDVIRTUAL3RC(int, _subgizmos_intersect_ray, Ref<EditorNode3DGizmo>, const Camera3D *, Vector2)
	GDVIRTUAL3RC(Vector<int>, _subgizmos_intersect_frustum, Ref<EditorNode3DGizmo>, const Camera3D *, TypedArray<Plane>)
	GDVIRTUAL2RC(Transform3D, _get_subgizmo_transform, Ref<EditorNode3DGizmo>, int)
	GDVIRTUAL3(_set_subgizmo_transform, Ref<EditorNode3DGizmo>, int, Transform3D)
	GDVIRTUAL4(_commit_subgizmos, Ref<EditorNode3DGizmo>, Vector<int>, TypedArray<Transform3D>, bool)

public:
	void create_material(const String &p_name, const Color &p_color, bool p_billboard = false, bool p_on_top = false, bool p_use_vertex_color = false);
	void create_icon_material(const String &p_name, const Ref<Texture2D> &p_texture, bool p_on_top = false, const Color &p_albedo = Color(1, 1, 1, 1));
	void create_handle_material(const String &p_name, bool p_billboard = false, const Ref<Texture2D> &p_texture = nullptr);
	void add_material(const String &p_name, Ref<StandardMaterial3D> p_material);

	Ref<StandardMaterial3D> get_material(const String &p_name, const Ref<EditorNode3DGizmo> &p_gizmo = Ref<EditorNode3DGizmo>());

	virtual String get_gizmo_name() const;
	virtual int get_priority() const;
	virtual bool can_be_hidden() const;
	virtual bool is_selectable_when_hidden() const;

	virtual void redraw(EditorNode3DGizmo *p_gizmo);
	virtual bool is_handle_highlighted(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point);
	virtual void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false);

	virtual int subgizmos_intersect_ray(const EditorNode3DGizmo *p_gizmo, Camera3D *p_camera, const Vector2 &p_point) const;
	virtual Vector<int> subgizmos_intersect_frustum(const EditorNode3DGizmo *p_gizmo, const Camera3D *p_camera, const Vector<Plane> &p_frustum) const;
	virtual Transform3D get_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id) const;
	virtual void set_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id, Transform3D p_transform);
	virtual void commit_subgizmos(const EditorNode3DGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform3D> &p_restore, bool p_cancel = false);

	Ref<EditorNode3DGizmo> get_gizmo(Node3D *p_spatial);
	void set_state(int p_state);
	int get_state() const;
	void unregister_gizmo(EditorNode3DGizmo *p_gizmo);

	EditorNode3DGizmoPlugin();
	virtual ~EditorNode3DGizmoPlugin();
};

class Light3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Light3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Light3DGizmoPlugin();
};

class AudioStreamPlayer3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(AudioStreamPlayer3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	AudioStreamPlayer3DGizmoPlugin();
};

class AudioListener3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(AudioListener3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;

	void redraw(EditorNode3DGizmo *p_gizmo) override;

	AudioListener3DGizmoPlugin();
};

class Camera3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Camera3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Camera3DGizmoPlugin();
};

class MeshInstance3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(MeshInstance3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	bool can_be_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	MeshInstance3DGizmoPlugin();
};

class OccluderInstance3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(OccluderInstance3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	OccluderInstance3DGizmoPlugin();
};

class Sprite3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Sprite3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	bool can_be_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Sprite3DGizmoPlugin();
};

class Position3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Position3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Ref<ArrayMesh> pos3d_mesh;
	Vector<Vector3> cursor_points;

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Position3DGizmoPlugin();
};

class PhysicalBone3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(PhysicalBone3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	PhysicalBone3DGizmoPlugin();
};

class RayCast3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(RayCast3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	RayCast3DGizmoPlugin();
};

class SpringArm3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(SpringArm3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	SpringArm3DGizmoPlugin();
};

class VehicleWheel3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VehicleWheel3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	VehicleWheel3DGizmoPlugin();
};

class SoftDynamicBody3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(SoftDynamicBody3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;
	bool is_handle_highlighted(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;

	SoftDynamicBody3DGizmoPlugin();
};

class VisibleOnScreenNotifier3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VisibleOnScreenNotifier3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	VisibleOnScreenNotifier3DGizmoPlugin();
};

class CPUParticles3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CPUParticles3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	CPUParticles3DGizmoPlugin();
};

class GPUParticles3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(GPUParticles3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	GPUParticles3DGizmoPlugin();
};

class GPUParticlesCollision3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(GPUParticlesCollision3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	GPUParticlesCollision3DGizmoPlugin();
};

class ReflectionProbeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(ReflectionProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	ReflectionProbeGizmoPlugin();
};

class DecalGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(DecalGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	DecalGizmoPlugin();
};

class VoxelGIGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VoxelGIGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	VoxelGIGizmoPlugin();
};

class LightmapGIGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(LightmapGIGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	LightmapGIGizmoPlugin();
};

class LightmapProbeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(LightmapProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	LightmapProbeGizmoPlugin();
};

class CollisionObject3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionObject3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	CollisionObject3DGizmoPlugin();
};

class CollisionShape3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionShape3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	CollisionShape3DGizmoPlugin();
};

class CollisionPolygon3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionPolygon3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	CollisionPolygon3DGizmoPlugin();
};

class NavigationRegion3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(NavigationRegion3DGizmoPlugin, EditorNode3DGizmoPlugin);

	struct _EdgeKey {
		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	NavigationRegion3DGizmoPlugin();
};

class JointGizmosDrawer {
public:
	static Basis look_body(const Transform3D &p_joint_transform, const Transform3D &p_body_transform);
	static Basis look_body_toward(Vector3::Axis p_axis, const Transform3D &joint_transform, const Transform3D &body_transform);
	static Basis look_body_toward_x(const Transform3D &p_joint_transform, const Transform3D &p_body_transform);
	static Basis look_body_toward_y(const Transform3D &p_joint_transform, const Transform3D &p_body_transform);
	/// Special function just used for physics joints, it returns a basis constrained toward Joint Z axis
	/// with axis X and Y that are looking toward the body and oriented toward up
	static Basis look_body_toward_z(const Transform3D &p_joint_transform, const Transform3D &p_body_transform);

	// Draw circle around p_axis
	static void draw_circle(Vector3::Axis p_axis, real_t p_radius, const Transform3D &p_offset, const Basis &p_base, real_t p_limit_lower, real_t p_limit_upper, Vector<Vector3> &r_points, bool p_inverse = false);
	static void draw_cone(const Transform3D &p_offset, const Basis &p_base, real_t p_swing, real_t p_twist, Vector<Vector3> &r_points);
};

class Joint3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Joint3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Timer *update_timer;
	uint64_t update_idx = 0;

	void incremental_update_gizmos();

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	static void CreatePinJointGizmo(const Transform3D &p_offset, Vector<Vector3> &r_cursor_points);
	static void CreateHingeJointGizmo(const Transform3D &p_offset, const Transform3D &p_trs_joint, const Transform3D &p_trs_body_a, const Transform3D &p_trs_body_b, real_t p_limit_lower, real_t p_limit_upper, bool p_use_limit, Vector<Vector3> &r_common_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateSliderJointGizmo(const Transform3D &p_offset, const Transform3D &p_trs_joint, const Transform3D &p_trs_body_a, const Transform3D &p_trs_body_b, real_t p_angular_limit_lower, real_t p_angular_limit_upper, real_t p_linear_limit_lower, real_t p_linear_limit_upper, Vector<Vector3> &r_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateConeTwistJointGizmo(const Transform3D &p_offset, const Transform3D &p_trs_joint, const Transform3D &p_trs_body_a, const Transform3D &p_trs_body_b, real_t p_swing, real_t p_twist, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateGeneric6DOFJointGizmo(
			const Transform3D &p_offset,
			const Transform3D &p_trs_joint,
			const Transform3D &p_trs_body_a,
			const Transform3D &p_trs_body_b,
			real_t p_angular_limit_lower_x,
			real_t p_angular_limit_upper_x,
			real_t p_linear_limit_lower_x,
			real_t p_linear_limit_upper_x,
			bool p_enable_angular_limit_x,
			bool p_enable_linear_limit_x,
			real_t p_angular_limit_lower_y,
			real_t p_angular_limit_upper_y,
			real_t p_linear_limit_lower_y,
			real_t p_linear_limit_upper_y,
			bool p_enable_angular_limit_y,
			bool p_enable_linear_limit_y,
			real_t p_angular_limit_lower_z,
			real_t p_angular_limit_upper_z,
			real_t p_linear_limit_lower_z,
			real_t p_linear_limit_upper_z,
			bool p_enable_angular_limit_z,
			bool p_enable_linear_limit_z,
			Vector<Vector3> &r_points,
			Vector<Vector3> *r_body_a_points,
			Vector<Vector3> *r_body_b_points);

	Joint3DGizmoPlugin();
};

class FogVolumeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(FogVolumeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	FogVolumeGizmoPlugin();
};

#endif // NODE_3D_EDITOR_GIZMOS_H
