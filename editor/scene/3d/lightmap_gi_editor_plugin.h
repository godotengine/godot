/**************************************************************************/
/*  lightmap_gi_editor_plugin.h                                           */
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

#include "editor/plugins/editor_plugin.h"

struct EditorProgress;
class EditorFileDialog;
class LightmapGI;
class MeshInstance3D;
class PopupPanel;
class Shader;
class ShaderMaterial;
class SpinBox;

class LightmapGIEditorPlugin : public EditorPlugin {
	GDCLASS(LightmapGIEditorPlugin, EditorPlugin);

	LightmapGI *lightmap = nullptr;

	Button *bake = nullptr;
	Button *preview = nullptr;
	Button *close_preview = nullptr;
	Button *preview_options = nullptr;
	PopupPanel *preview_options_popup = nullptr;
	SpinBox *target_density_spinbox = nullptr;
	Ref<Shader> preview_shader;
	ObjectID preview_lightmap_id;
	bool plugin_visible = false;
	float target_density = 1.0f;
	bool median_calculation_pending = false;

	struct DensitySample {
		float density = 0.0f;
		float world_area = 0.0f;

		bool operator<(const DensitySample &p_other) const {
			return density < p_other.density;
		}
	};

	struct PreviewInstance {
		RID instance;
		Ref<ShaderMaterial> material;
		ObjectID source_id;
		Size2 base_lightmap_size;
		Transform3D last_render_transform;
		Size2i last_render_lightmap_size;
		float last_target_density = 0.0f;
		float last_saturation_multiplier = 0.0f;
		bool last_visible = true;
		bool render_state_initialized = false;
	};

	Vector<PreviewInstance> preview_instances;

	EditorFileDialog *file_dialog = nullptr;
	static EditorProgress *tmp_progress;
	static bool bake_func_step(float p_progress, const String &p_description, void *, bool p_refresh);
	static void bake_func_end(uint64_t p_time_started);

	void _bake_select_file(const String &p_file);
	void _bake();
	void _preview_pressed();
	void _close_preview_pressed();
	void _update_preview_button(bool p_preview_active);
	void _preview_options_pressed();
	void _target_density_changed(double p_value);
	void _load_target_density();
	void _create_preview();
	void _clear_preview();
	void _find_preview_meshes(Node *p_node, Vector<MeshInstance3D *> &r_meshes) const;
	void _calculate_scene_median(float p_global_scale);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return "LightmapGI"; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void edited_scene_changed() override;

	LightmapGIEditorPlugin();
	~LightmapGIEditorPlugin();
};
