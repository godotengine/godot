#ifndef MESH_INSTANCE_EDITOR_PLUGIN_H
#define MESH_INSTANCE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/mesh_instance.h"
#include "scene/gui/spin_box.h"

class MeshInstanceEditor : public Node {

	OBJ_TYPE(MeshInstanceEditor, Node);

	enum Menu {

		MENU_OPTION_CREATE_STATIC_TRIMESH_BODY,
		MENU_OPTION_CREATE_STATIC_CONVEX_BODY,
		MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE,
		MENU_OPTION_CREATE_CONVEX_COLLISION_SHAPE,
		MENU_OPTION_CREATE_NAVMESH,
		MENU_OPTION_CREATE_OUTLINE_MESH,
	};

	MeshInstance *node;

	MenuButton *options;

	ConfirmationDialog *outline_dialog;
	SpinBox *outline_size;

	AcceptDialog *err_dialog;

	void _menu_option(int p_option);
	void _create_outline_mesh();

	friend class MeshInstanceEditorPlugin;

protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(MeshInstance *p_mesh);
	MeshInstanceEditor();
};

class MeshInstanceEditorPlugin : public EditorPlugin {

	OBJ_TYPE(MeshInstanceEditorPlugin, EditorPlugin);

	MeshInstanceEditor *mesh_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "MeshInstance"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	MeshInstanceEditorPlugin(EditorNode *p_node);
	~MeshInstanceEditorPlugin();
};

#endif // MESH_EDITOR_PLUGIN_H
