#pragma once
#include "../common.h"
#include "containers/hash_map.hpp"
#include "containers/hash_set.hpp"
#include "containers/local_vector.hpp"
#include "containers/inline_vector.hpp"
#ifdef TOOLS_ENABLED
class JoltJointGizmoPlugin3D final : public EditorNode3DGizmoPlugin {
	GDCLASS(JoltJointGizmoPlugin3D, EditorNode3DGizmoPlugin)

private:
	static void _bind_methods() { }

public:
	JoltJointGizmoPlugin3D() = default;

	explicit JoltJointGizmoPlugin3D(EditorInterface* p_editor_interface);

	bool has_gizmo(Node3D* p_node) override;

	Ref<EditorNode3DGizmo> create_gizmo(Node3D* p_node) override;

	String get_gizmo_name() const override;

	void redraw(EditorNode3DGizmo* p_gizmo) override;

	void redraw_gizmos();

private:
	void _create_materials();

	void _create_redraw_timer(const Ref<EditorNode3DGizmo>& p_gizmo);

	void _redraw_gizmos();

	mutable JHashSet<Ref<EditorNode3DGizmo>> gizmos;

	EditorInterface* editor_interface = nullptr;

	bool initialized = false;
};

#endif // TOOLS_ENABLED
