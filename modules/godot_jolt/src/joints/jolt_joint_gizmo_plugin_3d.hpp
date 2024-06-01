#pragma once

#ifdef GDJ_CONFIG_EDITOR

class JoltJointGizmoPlugin3D final : public EditorNode3DGizmoPlugin {
	GDCLASS(JoltJointGizmoPlugin3D, EditorNode3DGizmoPlugin)

private:
	static void _bind_methods();

public:
	JoltJointGizmoPlugin3D() = default;

	explicit JoltJointGizmoPlugin3D(EditorInterface* p_editor_interface);

	bool _has_gizmo(Node3D* p_node) const override;

	Ref<EditorNode3DGizmo> _create_gizmo(Node3D* p_node) const override;

	String _get_gizmo_name() const override;

	void _redraw(const Ref<EditorNode3DGizmo>& p_gizmo) override;

	void redraw_gizmos();

private:
	void _create_materials();

	void _create_redraw_timer(const Ref<EditorNode3DGizmo>& p_gizmo);

	mutable HashSet<Ref<EditorNode3DGizmo>> gizmos;

	EditorInterface* editor_interface = nullptr;

	bool initialized = false;
};

#endif // GDJ_CONFIG_EDITOR
