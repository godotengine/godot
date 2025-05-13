/**************************************************************************/
/*  material_editor_plugin.h                                              */
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

#pragma once

#include "editor/editor_inspector.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/plugins/editor_resource_conversion_plugin.h"
#include "scene/resources/material.h"

#ifndef _3D_DISABLED
#include "scene/resources/3d/primitive_meshes.h"

class Camera3D;
class DirectionalLight3D;
class MeshInstance3D;
#endif // _3D_DISABLED

class ColorRect;
class HBoxContainer;
class SubViewport;
class SubViewportContainer;
class Button;
class Label;

class MaterialEditor : public Control {
	GDCLASS(MaterialEditor, Control);

	SubViewportContainer *vc_2d = nullptr;
	SubViewport *viewport_2d = nullptr;
	HBoxContainer *layout_2d = nullptr;
	ColorRect *rect_instance = nullptr;

	SubViewportContainer *vc = nullptr;
	SubViewport *viewport = nullptr;

	Ref<Material> material;

#ifndef _3D_DISABLED
	Vector2 rot;

	Node3D *rotation = nullptr;
	MeshInstance3D *sphere_instance = nullptr;
	MeshInstance3D *box_instance = nullptr;
	MeshInstance3D *quad_instance = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;

	Ref<SphereMesh> sphere_mesh;
	Ref<BoxMesh> box_mesh;
	Ref<QuadMesh> quad_mesh;

	VBoxContainer *layout_error = nullptr;
	Label *error_label = nullptr;
	bool is_unsupported_shader_mode = false;

	HBoxContainer *layout_3d = nullptr;

	Button *sphere_switch = nullptr;
	Button *box_switch = nullptr;
	Button *quad_switch = nullptr;
	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;
#endif // _3D_DISABLED

	struct ThemeCache {
#ifndef _3D_DISABLED
		Ref<Texture2D> light_1_icon;
		Ref<Texture2D> light_2_icon;
		Ref<Texture2D> sphere_icon;
		Ref<Texture2D> box_icon;
		Ref<Texture2D> quad_icon;
#endif // _3D_DISABLED
		Ref<Texture2D> checkerboard;
	} theme_cache;

#ifndef _3D_DISABLED
	void _on_light_1_switch_pressed();
	void _on_light_2_switch_pressed();
	void _on_sphere_switch_pressed();
	void _on_box_switch_pressed();
	void _on_quad_switch_pressed();
#endif // _3D_DISABLED

protected:
	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
#ifndef _3D_DISABLED
	void gui_input(const Ref<InputEvent> &p_event) override;
	void _set_rotation(real_t p_x_degrees, real_t p_y_degrees);
	void _store_rotation_metadata();
	void _update_rotation();
#endif // _3D_DISABLED

public:
#ifndef _3D_DISABLED
	void edit(Ref<Material> p_material, const Ref<Environment> &p_env);
#else
	void edit(Ref<Material> p_material);
#endif // _3D_DISABLED
	MaterialEditor();
};

class EditorInspectorPluginMaterial : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginMaterial, EditorInspectorPlugin);
#ifndef _3D_DISABLED
	Ref<Environment> env;
#endif // _3D_DISABLED

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;

	void _undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, const String &p_property, const Variant &p_new_value);

	EditorInspectorPluginMaterial();
};

class MaterialEditorPlugin : public EditorPlugin {
	GDCLASS(MaterialEditorPlugin, EditorPlugin);

public:
	virtual String get_plugin_name() const override { return "Material"; }

	MaterialEditorPlugin();
};

#ifndef _3D_DISABLED
class StandardMaterial3DConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(StandardMaterial3DConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

class ORMMaterial3DConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(ORMMaterial3DConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};
#endif // _3D_DISABLED

class ParticleProcessMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(ParticleProcessMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

class CanvasItemMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(CanvasItemMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

#ifndef _3D_DISABLED
class ProceduralSkyMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(ProceduralSkyMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

class PanoramaSkyMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(PanoramaSkyMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

class PhysicalSkyMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(PhysicalSkyMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

class FogMaterialConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(FogMaterialConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};
#endif // _3D_DISABLED
