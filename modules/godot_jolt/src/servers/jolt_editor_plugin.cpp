#include "jolt_editor_plugin.hpp"

#ifdef TOOLS_ENABLED

#include "servers/jolt_physics_server_3d.hpp"

void JoltEditorPlugin::_enter_tree() {
	EditorInterface* editor_interface = get_editor_interface();

	Ref<Theme> editor_theme = editor_interface->get_editor_theme();

	Ref<Texture2D> icon_pin = editor_theme->get_icon("PinJoint3D", "EditorIcons");
	Ref<Texture2D> icon_hinge = editor_theme->get_icon("HingeJoint3D", "EditorIcons");
	Ref<Texture2D> icon_slider = editor_theme->get_icon("SliderJoint3D", "EditorIcons");
	Ref<Texture2D> icon_cone_twist = editor_theme->get_icon("ConeTwistJoint3D", "EditorIcons");
	Ref<Texture2D> icon_6dof = editor_theme->get_icon("Generic6DOFJoint3D", "EditorIcons");

	Ref<Theme> temp_theme = Ref(memnew(Theme));

	temp_theme->set_icon("JoltPinJoint3D", "EditorIcons", icon_pin);
	temp_theme->set_icon("JoltHingeJoint3D", "EditorIcons", icon_hinge);
	temp_theme->set_icon("JoltSliderJoint3D", "EditorIcons", icon_slider);
	temp_theme->set_icon("JoltConeTwistJoint3D", "EditorIcons", icon_cone_twist);
	temp_theme->set_icon("JoltGeneric6DOFJoint3D", "EditorIcons", icon_6dof);

	editor_theme->merge_with(temp_theme);

	joint_gizmo_plugin = Ref(memnew(JoltJointGizmoPlugin3D(editor_interface)));
	add_node_3d_gizmo_plugin(joint_gizmo_plugin);

	PopupMenu* tool_menu = memnew(PopupMenu);
	tool_menu->connect("id_pressed", Callable(this, NAMEOF(_tool_menu_pressed)));
	tool_menu->add_item("Dump Debug Snapshots", MENU_OPTION_DUMP_DEBUG_SNAPSHOTS);

	add_tool_submenu_item("Jolt Physics", tool_menu);
}

void JoltEditorPlugin::_exit_tree() {
	remove_node_3d_gizmo_plugin(joint_gizmo_plugin);
	joint_gizmo_plugin.unref();

	if (debug_snapshots_dialog != nullptr) {
		debug_snapshots_dialog->queue_free();
		debug_snapshots_dialog = nullptr;
	}
}

void JoltEditorPlugin::_tool_menu_pressed(int32_t p_index) {
	// NOLINTNEXTLINE(hicpp-multiway-paths-covered)
	switch (p_index) {
		case MENU_OPTION_DUMP_DEBUG_SNAPSHOTS: {
			_dump_debug_snapshots();
		} break;
	}
}

void JoltEditorPlugin::_snapshots_dir_selected(const String& p_dir) {
	JoltPhysicsServer3D::get_singleton()->dump_debug_snapshots(p_dir);
}

void JoltEditorPlugin::_dump_debug_snapshots() {
	if (debug_snapshots_dialog == nullptr) {
		debug_snapshots_dialog = memnew(EditorFileDialog);
		debug_snapshots_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
		debug_snapshots_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		debug_snapshots_dialog->set_current_dir("res://");
		debug_snapshots_dialog->connect(
			"dir_selected",
			Callable(this, NAMEOF(_snapshots_dir_selected))
		);

		get_editor_interface()->get_base_control()->add_child(debug_snapshots_dialog);
	}

	debug_snapshots_dialog->popup_centered_ratio(0.5);
}

#endif // TOOLS_ENABLED
