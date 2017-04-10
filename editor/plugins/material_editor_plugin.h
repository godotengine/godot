#ifndef MATERIAL_EDITOR_PLUGIN_H
#define MATERIAL_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/resources/material.h"

class MaterialEditor : public Control {

	OBJ_TYPE(MaterialEditor, Control);

	Viewport *viewport;
	MeshInstance *sphere_instance;
	MeshInstance *box_instance;
	DirectionalLight *light1;
	DirectionalLight *light2;
	Camera *camera;

	Ref<Mesh> sphere_mesh;
	Ref<Mesh> box_mesh;

	TextureButton *sphere_switch;
	TextureButton *box_switch;

	TextureButton *light_1_switch;
	TextureButton *light_2_switch;

	Ref<Material> material;

	void _button_pressed(Node *p_button);
	bool first_enter;

protected:
	void _notification(int p_what);
	void _input_event(InputEvent p_event);
	static void _bind_methods();

public:
	void edit(Ref<Material> p_material);
	MaterialEditor();
};

class MaterialEditorPlugin : public EditorPlugin {

	OBJ_TYPE(MaterialEditorPlugin, EditorPlugin);

	MaterialEditor *material_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "Material"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	MaterialEditorPlugin(EditorNode *p_node);
	~MaterialEditorPlugin();
};

#endif // MATERIAL_EDITOR_PLUGIN_H
