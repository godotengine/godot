/**************************************************************************/
/*  spatial.h                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef SPATIAL_H
#define SPATIAL_H

#include "scene/main/node.h"
#include "scene/main/scene_tree.h"

class SpatialGizmo : public Reference {
	GDCLASS(SpatialGizmo, Reference);

public:
	virtual void create() = 0;
	virtual void transform() = 0;
	virtual void clear() = 0;
	virtual void redraw() = 0;
	virtual void free() = 0;

	SpatialGizmo();
	virtual ~SpatialGizmo() {}
};

class Spatial : public Node {
	GDCLASS(Spatial, Node);
	OBJ_CATEGORY("3D");

	friend class SceneTreeFTI;
	friend class SceneTreeFTITests;

public:
	static constexpr AncestralClass static_ancestral_class = AncestralClass::SPATIAL;

	enum MergingMode : unsigned int {
		MERGING_MODE_INHERIT,
		MERGING_MODE_OFF,
		MERGING_MODE_ON
	};

private:
	// optionally stored if we need to do interpolation
	// client side (i.e. not in VisualServer) so interpolated transforms
	// can be read back with get_global_transform_interpolated()
	struct ClientPhysicsInterpolationData {
		Transform global_xform_curr;
		Transform global_xform_prev;
		uint64_t current_physics_tick = 0;
		uint64_t timeout_physics_tick = 0;
	};

	enum TransformDirty {
		DIRTY_NONE = 0,
		DIRTY_VECTORS = 1,
		DIRTY_LOCAL = 2,
		DIRTY_GLOBAL = 4,
		DIRTY_GLOBAL_INTERPOLATED = 8,
	};

	mutable SelfList<Node> xform_change;
	SelfList<Spatial> _client_physics_interpolation_spatials_list;

	struct Data {
		// Interpolated global transform - correct on the frame only.
		// Only used with FTI.
		Transform global_transform_interpolated;

		// Current xforms are either
		// * Used for everything (when not using FTI)
		// * Correct on the physics tick (when using FTI)
		mutable Transform global_transform;
		mutable Transform local_transform;

		// Only used with FTI.
		Transform local_transform_prev;

		mutable Vector3 rotation;
		mutable Vector3 scale;

		mutable int dirty;

		Viewport *viewport;

		MergingMode merging_mode : 2;

		bool toplevel_active : 1;
		bool toplevel : 1;
		bool inside_world : 1;

		bool ignore_notification : 1;
		bool notify_local_transform : 1;
		bool notify_transform : 1;

		bool visible : 1;
		bool visible_in_tree : 1;
		bool disable_scale : 1;

		// Scene tree interpolation
		bool fti_on_frame_xform_list : 1;
		bool fti_on_frame_property_list : 1;
		bool fti_on_tick_xform_list : 1;
		bool fti_on_tick_property_list : 1;
		bool fti_global_xform_interp_set : 1;
		bool fti_frame_xform_force_update : 1;
		bool fti_is_identity_xform : 1;
		bool fti_processed : 1;

		bool merging_allowed : 1;

		Spatial *parent;

		// An unordered vector of `Spatial` children only.
		// This is a subset of the `Node::children`, purely
		// an optimization for faster traversal.
		LocalVector<Spatial *> spatial_children;
		uint32_t index_in_parent = UINT32_MAX;

		float lod_range = 10.0f;
		ClientPhysicsInterpolationData *client_physics_interpolation_data;

#ifdef TOOLS_ENABLED
		Ref<SpatialGizmo> gizmo;
		bool gizmo_disabled : 1;
		bool gizmo_dirty : 1;
#endif

	} data;

	void _update_gizmo();
	void _notify_dirty();
	void _propagate_transform_changed(Spatial *p_origin);

	void _update_visible_in_tree();
	bool _is_visible_in_tree_reference() const;
	void _propagate_visible_in_tree(bool p_visible_in_tree);
	void _propagate_visibility_changed();
	void _propagate_merging_allowed(bool p_merging_allowed);

protected:
	_FORCE_INLINE_ void set_ignore_transform_notification(bool p_ignore) { data.ignore_notification = p_ignore; }
	_FORCE_INLINE_ void _update_local_transform() const;

	Transform _get_global_transform_interpolated(real_t p_interpolation_fraction);
	const Transform &_get_cached_global_transform_interpolated() const { return data.global_transform_interpolated; }
	void _disable_client_physics_interpolation();

	// Calling this announces to the FTI system that a node has been moved,
	// or requires an update in terms of interpolation
	// (e.g. changing Camera zoom even if position hasn't changed).
	void fti_notify_node_changed(bool p_transform_changed = true);

	// Opportunity after FTI to update the servers
	// with global_transform_interpolated,
	// and any custom interpolated data in derived classes.
	// Make sure to call the parent class fti_update_servers(),
	// so all data is updated to the servers.
	virtual void fti_update_servers_xform() {}
	virtual void fti_update_servers_property() {}

	// Pump the FTI data, also gives a chance for inherited classes
	// to pump custom data, but they *must* call the base class here too.
	// This is the opportunity for classes to move current values for
	// transforms and properties to stored previous values,
	// and this should take place both on ticks, and during resets.
	virtual void fti_pump_xform();
	virtual void fti_pump_property() {}

	void _notification(int p_what);
	static void _bind_methods();

public:
	enum {

		NOTIFICATION_TRANSFORM_CHANGED = SceneTree::NOTIFICATION_TRANSFORM_CHANGED,
		NOTIFICATION_ENTER_WORLD = 41,
		NOTIFICATION_EXIT_WORLD = 42,
		NOTIFICATION_VISIBILITY_CHANGED = 43,
		NOTIFICATION_LOCAL_TRANSFORM_CHANGED = 44,
		NOTIFICATION_ENTER_GAMEPLAY = 45,
		NOTIFICATION_EXIT_GAMEPLAY = 46,
	};

	virtual void notification_callback(int p_message_type);
	Spatial *get_parent_spatial() const;

	Ref<World> get_world() const;

	void set_translation(const Vector3 &p_translation);
	void set_rotation(const Vector3 &p_euler_rad);
	void set_rotation_degrees(const Vector3 &p_euler_deg);
	void set_scale(const Vector3 &p_scale);

	void set_global_translation(const Vector3 &p_translation);
	void set_global_rotation(const Vector3 &p_euler_rad);

	Vector3 get_translation() const;
	Vector3 get_rotation() const;
	Vector3 get_rotation_degrees() const;
	Vector3 get_scale() const;

	Vector3 get_global_translation() const;
	Vector3 get_global_rotation() const;

	void set_transform(const Transform &p_transform);
	void set_global_transform(const Transform &p_transform);

	Transform get_transform() const;
	Transform get_global_transform() const;
	Transform get_global_transform_interpolated();
	bool update_client_physics_interpolation_data();

#ifdef TOOLS_ENABLED
	virtual Transform get_global_gizmo_transform() const;
	virtual Transform get_local_gizmo_transform() const;
	virtual AABB get_fallback_gizmo_aabb() const;
#endif

	void set_as_toplevel(bool p_enabled);
	bool is_set_as_toplevel() const;

	void set_disable_scale(bool p_enabled);
	bool is_scale_disabled() const;

	void set_disable_gizmo(bool p_enabled);
	void update_gizmo();
	void set_gizmo(const Ref<SpatialGizmo> &p_gizmo);
	Ref<SpatialGizmo> get_gizmo() const;

	_FORCE_INLINE_ bool is_inside_world() const { return data.inside_world; }

	Transform get_relative_transform(const Node *p_parent) const;

	void rotate(const Vector3 &p_axis, float p_angle);
	void rotate_x(float p_angle);
	void rotate_y(float p_angle);
	void rotate_z(float p_angle);
	void translate(const Vector3 &p_offset);
	void scale(const Vector3 &p_ratio);

	void rotate_object_local(const Vector3 &p_axis, float p_angle);
	void scale_object_local(const Vector3 &p_scale);
	void translate_object_local(const Vector3 &p_offset);

	void global_rotate(const Vector3 &p_axis, float p_angle);
	void global_scale(const Vector3 &p_scale);
	void global_translate(const Vector3 &p_offset);

	void look_at(const Vector3 &p_target, const Vector3 &p_up);
	void look_at_from_position(const Vector3 &p_pos, const Vector3 &p_target, const Vector3 &p_up);

	Vector3 to_local(Vector3 p_global) const;
	Vector3 to_global(Vector3 p_local) const;

	void set_notify_transform(bool p_enable);
	bool is_transform_notification_enabled() const;

	void set_notify_local_transform(bool p_enable);
	bool is_local_transform_notification_enabled() const;

	void set_merging_mode(MergingMode p_mode);
	MergingMode get_merging_mode() const { return data.merging_mode; }
	_FORCE_INLINE_ bool is_merging_allowed() const { return data.merging_allowed; }

	void orthonormalize();
	void set_identity();

	void set_visible(bool p_visible);
	bool is_visible() const;
	void show();
	void hide();
	bool is_visible_in_tree() const {
#if DEV_ENABLED
		// As this is newly introduced, regression test the old method against the new in DEV builds.
		// If no regressions, this can be removed after a beta.
		bool visible = _is_visible_in_tree_reference();
		ERR_FAIL_COND_V_MSG(data.visible_in_tree != visible, visible, "is_visible_in_tree regression detected, recovering.");
#endif
		return data.visible_in_tree;
	}

	void force_update_transform();

	void set_lod_range(float p_range);
	float get_lod_range() const { return data.lod_range; }

	Spatial();
	~Spatial();
};

#endif // SPATIAL_H
