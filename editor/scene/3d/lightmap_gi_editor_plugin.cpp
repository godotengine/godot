/**************************************************************************/
/*  lightmap_gi_editor_plugin.cpp                                         */
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

#include "lightmap_gi_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_viewport.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/lightmap_gi.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/popup.h"
#include "scene/gui/spin_box.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

#include "modules/modules_enabled.gen.h" // For lightmapper_rd.

void LightmapGIEditorPlugin::_bake_select_file(const String &p_file) {
	if (lightmap) {
		LightmapGI::BakeError err = LightmapGI::BAKE_ERROR_OK;
		const uint64_t time_started = OS::get_singleton()->get_ticks_msec();
		if (get_tree()->get_edited_scene_root()) {
			Ref<LightmapGIData> lightmapGIData = lightmap->get_light_data();

			if (lightmapGIData.is_valid()) {
				String path = lightmapGIData->get_path();
				if (!path.is_resource_file()) {
					int srpos = path.find("::");
					if (srpos != -1) {
						String base = path.substr(0, srpos);
						if (ResourceLoader::get_resource_type(base) == "PackedScene") {
							if (!get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->get_scene_file_path() != base) {
								err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
							}
						} else {
							if (FileAccess::exists(base + ".import")) {
								err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
							}
						}
					}
				} else {
					if (FileAccess::exists(path + ".import")) {
						err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
					}
				}
			}

			if (err == LightmapGI::BAKE_ERROR_OK) {
				if (get_tree()->get_edited_scene_root() == lightmap) {
					err = lightmap->bake(lightmap, p_file, bake_func_step);
				} else {
					err = lightmap->bake(lightmap->get_parent(), p_file, bake_func_step);
				}
			}
		} else {
			err = LightmapGI::BAKE_ERROR_NO_SCENE_ROOT;
		}

		bake_func_end(time_started);

		switch (err) {
			case LightmapGI::BAKE_ERROR_NO_SAVE_PATH: {
				String scene_path = lightmap->get_scene_file_path();
				if (scene_path.is_empty() && lightmap->get_owner()) {
					scene_path = lightmap->get_owner()->get_scene_file_path();
				}
				if (scene_path.is_empty()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for lightmap images.\nSave your scene and try again."));
					break;
				}
				scene_path = scene_path.get_basename() + ".lmbake";

				file_dialog->set_current_path(scene_path);
				file_dialog->popup_file_dialog();
			} break;
			case LightmapGI::BAKE_ERROR_NO_MESHES: {
				EditorNode::get_singleton()->show_warning(
						TTR("No meshes with lightmapping support to bake. Make sure they contain UV2 data and their Global Illumination property is set to Static.") +
						String::utf8("\n\n•  ") + TTR("To import a scene with lightmapping support, set Meshes > Light Baking to Static Lightmaps in the Import dock.") +
						String::utf8("\n•  ") + TTR("To enable lightmapping support on a primitive mesh, edit the PrimitiveMesh resource in the inspector and check Add UV2.") +
						String::utf8("\n•  ") + TTR("To enable lightmapping support on a CSG mesh, select the root CSG node and choose CSG > Bake Mesh Instance at the top of the 3D editor viewport.\nSelect the generated MeshInstance3D node and choose Mesh > Unwrap UV2 for Lightmap/AO at the top of the 3D editor viewport."));
			} break;
			case LightmapGI::BAKE_ERROR_CANT_CREATE_IMAGE: {
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images. Make sure the lightmap destination path is writable."));
			} break;
			case LightmapGI::BAKE_ERROR_NO_SCENE_ROOT: {
				EditorNode::get_singleton()->show_warning(TTR("No editor scene root found."));
			} break;
			case LightmapGI::BAKE_ERROR_FOREIGN_DATA: {
				EditorNode::get_singleton()->show_warning(TTR("Lightmap data is not local to the scene."));
			} break;
			case LightmapGI::BAKE_ERROR_TEXTURE_SIZE_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Maximum texture size is too small for the lightmap images.\nWhile this can be fixed by increasing the maximum texture size, it is recommended you split the scene into more objects instead."));
			} break;
			case LightmapGI::BAKE_ERROR_LIGHTMAP_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images. Make sure all meshes to bake have the Lightmap Size Hint property set high enough, and the LightmapGI's Texel Scale value is not too low."));
			} break;
			case LightmapGI::BAKE_ERROR_ATLAS_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Failed fitting a lightmap image into an atlas. This should never happen and should be reported."));
			} break;
			default: {
			} break;
		}
	}
}

void LightmapGIEditorPlugin::_bake() {
	_bake_select_file("");
}

void LightmapGIEditorPlugin::_preview_pressed() {
	callable_mp(this, &LightmapGIEditorPlugin::_create_preview).call_deferred();
}

void LightmapGIEditorPlugin::_close_preview_pressed() {
	callable_mp(this, &LightmapGIEditorPlugin::_clear_preview).call_deferred();
}

void LightmapGIEditorPlugin::_update_preview_button(bool p_preview_active) {
	preview->set_visible(plugin_visible && !p_preview_active);
	close_preview->set_visible(p_preview_active);
	preview_options->set_visible(plugin_visible || p_preview_active);
}

void LightmapGIEditorPlugin::_preview_options_pressed() {
	const Vector2 popup_position = preview_options->get_screen_position() + preview_options->get_size();
	preview_options_popup->set_position(popup_position - Vector2(preview_options_popup->get_contents_minimum_size().x, 0));
	preview_options_popup->reset_size();
	preview_options_popup->popup();
}

void LightmapGIEditorPlugin::_target_density_changed(double p_value) {
	target_density = p_value;
	if (lightmap) {
		lightmap->set_meta(SNAME("_editor_texel_density"), target_density);
		EditorInterface::get_singleton()->mark_scene_as_unsaved();
	}
}

void LightmapGIEditorPlugin::_load_target_density() {
	target_density = 1.0f;
	const Variant stored_density = lightmap->get_meta(SNAME("_editor_texel_density"), Variant());
	if (stored_density.get_type() == Variant::FLOAT || stored_density.get_type() == Variant::INT) {
		target_density = MAX(double(stored_density), 0.01);
	}
	target_density_spinbox->set_value_no_signal(target_density);
}

void LightmapGIEditorPlugin::_clear_preview() {
	set_process(false);
	median_calculation_pending = false;
	for (const PreviewInstance &preview_instance : preview_instances) {
		if (preview_instance.instance.is_valid()) {
			RS::get_singleton()->free_rid(preview_instance.instance);
		}
	}
	preview_instances.clear();
	preview_lightmap_id = ObjectID();
	_update_preview_button(false);
}

void LightmapGIEditorPlugin::_find_preview_meshes(Node *p_node, Vector<MeshInstance3D *> &r_meshes) const {
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh_instance && mesh_instance->get_gi_mode() == GeometryInstance3D::GI_MODE_STATIC) {
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		if (mesh.is_valid()) {
			bool has_surface = false;
			bool valid_for_lightmap = true;
			for (int surface = 0; surface < mesh->get_surface_count(); surface++) {
				if (mesh->surface_get_primitive_type(surface) != Mesh::PRIMITIVE_TRIANGLES) {
					continue;
				}
				has_surface = true;
				const BitField<Mesh::ArrayFormat> format = mesh->surface_get_format(surface);
				if (!(format & Mesh::ARRAY_FORMAT_TEX_UV2) || !(format & Mesh::ARRAY_FORMAT_NORMAL)) {
					valid_for_lightmap = false;
					break;
				}
			}
			if (has_surface && valid_for_lightmap) {
				r_meshes.push_back(mesh_instance);
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		if (child->get_owner()) {
			_find_preview_meshes(child, r_meshes);
		}
	}
}

void LightmapGIEditorPlugin::_create_preview() {
	_clear_preview();
	if (!lightmap || !lightmap->is_inside_tree() || !lightmap->get_world_3d().is_valid()) {
		return;
	}
	for (uint32_t i = 0; i < Node3DEditor::VIEWPORTS_COUNT; i++) {
		Node3DEditorViewport *editor_viewport = Node3DEditor::get_singleton()->get_editor_viewport(i);
		if (editor_viewport) {
			editor_viewport->set_display_mode_normal();
		}
	}

	Node *root = get_tree()->get_edited_scene_root() == lightmap ? static_cast<Node *>(lightmap) : lightmap->get_parent();
	if (!root) {
		return;
	}
	Vector<MeshInstance3D *> meshes;
	_find_preview_meshes(root, meshes);
	const RID scenario = lightmap->get_world_3d()->get_scenario();
	const float global_scale = lightmap->get_texel_scale();
	preview_lightmap_id = lightmap->get_instance_id();

	for (MeshInstance3D *mesh_instance : meshes) {
		const Ref<Mesh> mesh = mesh_instance->get_mesh();
		Size2 base_size = mesh->get_lightmap_size_hint();
		if (base_size == Size2()) {
			base_size = Size2(64, 64);
		}
		const Size2i estimated_size = Size2i(base_size * global_scale * mesh_instance->get_lightmap_texel_scale());
		if (estimated_size.x < 1.0 || estimated_size.y < 1.0) {
			continue;
		}

		PreviewInstance preview_instance;
		preview_instance.material.instantiate();
		preview_instance.material->set_shader(preview_shader);
		preview_instance.material->set_shader_parameter(SNAME("estimated_lightmap_size"), estimated_size);
		preview_instance.material->set_shader_parameter(SNAME("target_texel_density"), target_density);
		preview_instance.source_id = mesh_instance->get_instance_id();
		preview_instance.base_lightmap_size = base_size;
		preview_instance.instance = RS::get_singleton()->instance_create2(mesh->get_rid(), scenario);
		RS::get_singleton()->instance_set_transform(preview_instance.instance, mesh_instance->get_global_transform());
		RS::get_singleton()->instance_set_layer_mask(preview_instance.instance, mesh_instance->get_layer_mask());
		RS::get_singleton()->instance_set_visible(preview_instance.instance, mesh_instance->is_visible_in_tree());
		RS::get_singleton()->instance_geometry_set_material_override(preview_instance.instance, preview_instance.material->get_rid());
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(preview_instance.instance, RSE::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_geometry_set_flag(preview_instance.instance, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(preview_instance.instance, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
		preview_instances.push_back(preview_instance);
	}

	median_calculation_pending = !lightmap->has_meta(SNAME("_editor_texel_density"));
	_update_preview_button(true);
	set_process(true);
}

void LightmapGIEditorPlugin::_calculate_scene_median(float p_global_scale) {
	Vector<DensitySample> density_samples;
	float total_world_area = 0.0f;
	for (const PreviewInstance &preview_instance : preview_instances) {
		MeshInstance3D *source = ObjectDB::get_instance<MeshInstance3D>(preview_instance.source_id);
		if (!source || !source->is_inside_tree()) {
			continue;
		}
		const Ref<Mesh> mesh = source->get_mesh();
		if (mesh.is_null()) {
			continue;
		}
		const Basis world_basis = source->get_global_transform().basis;
		float world_area = 0.0f;
		for (const Face3 &face : mesh->get_faces()) {
			const Vector3 edge_1 = world_basis.xform(face.vertex[1] - face.vertex[0]);
			const Vector3 edge_2 = world_basis.xform(face.vertex[2] - face.vertex[0]);
			world_area += edge_1.cross(edge_2).length() * 0.5f;
		}
		if (world_area <= CMP_EPSILON2) {
			continue;
		}
		const float mesh_scale = p_global_scale * source->get_lightmap_texel_scale();
		const Size2i lightmap_size = Size2i(preview_instance.base_lightmap_size * mesh_scale);
		DensitySample sample;
		sample.density = Math::sqrt(float(lightmap_size.x) * float(lightmap_size.y) / world_area);
		sample.world_area = world_area;
		density_samples.push_back(sample);
		total_world_area += world_area;
	}

	density_samples.sort();
	target_density = 1.0f;
	float accumulated_area = 0.0f;
	for (const DensitySample &sample : density_samples) {
		accumulated_area += sample.world_area;
		if (accumulated_area >= total_world_area * 0.5f) {
			target_density = sample.density;
			break;
		}
	}
	target_density = MAX(target_density, 0.01f);
	target_density_spinbox->set_value_no_signal(target_density);
}

void LightmapGIEditorPlugin::_notification(int p_what) {
	if (p_what != NOTIFICATION_PROCESS || preview_lightmap_id.is_null()) {
		return;
	}
	LightmapGI *preview_lightmap = ObjectDB::get_instance<LightmapGI>(preview_lightmap_id);
	if (!preview_lightmap || !preview_lightmap->is_inside_tree()) {
		_clear_preview();
		return;
	}

	bool preview_visible = true;
	Node3DEditorViewport *editor_viewport = Node3DEditor::get_singleton()->get_last_used_viewport();
	if (editor_viewport && editor_viewport->get_viewport_node()->get_debug_draw() != Viewport::DEBUG_DRAW_DISABLED) {
		preview_visible = false;
	}
	const float global_scale = preview_lightmap->get_texel_scale();
	if (median_calculation_pending) {
		median_calculation_pending = false;
		if (!preview_lightmap->has_meta(SNAME("_editor_texel_density"))) {
			_calculate_scene_median(global_scale);
		}
	}

	const HashMap<ObjectID, Object *> &selection = EditorNode::get_singleton()->get_editor_selection()->get_selection();
	bool has_selected_preview_mesh = false;
	for (const PreviewInstance &preview_instance : preview_instances) {
		if (selection.has(preview_instance.source_id)) {
			has_selected_preview_mesh = true;
			break;
		}
	}

	const float clamped_target_density = MAX(target_density, 0.0001f);
	for (PreviewInstance &preview_instance : preview_instances) {
		MeshInstance3D *source = ObjectDB::get_instance<MeshInstance3D>(preview_instance.source_id);
		if (!source || !source->is_inside_tree()) {
			_clear_preview();
			return;
		}
		const Size2i lightmap_size = Size2i(preview_instance.base_lightmap_size * global_scale * source->get_lightmap_texel_scale());
		const Transform3D transform = source->get_global_transform();
		if (!preview_instance.render_state_initialized || preview_instance.last_render_lightmap_size != lightmap_size) {
			preview_instance.material->set_shader_parameter(SNAME("estimated_lightmap_size"), lightmap_size);
			preview_instance.last_render_lightmap_size = lightmap_size;
		}
		if (!preview_instance.render_state_initialized || !Math::is_equal_approx(preview_instance.last_target_density, clamped_target_density)) {
			preview_instance.material->set_shader_parameter(SNAME("target_texel_density"), clamped_target_density);
			preview_instance.last_target_density = clamped_target_density;
		}
		const float saturation_multiplier = !has_selected_preview_mesh ? 1.0f : (selection.has(preview_instance.source_id) ? 1.25f : 0.75f);
		if (!preview_instance.render_state_initialized || !Math::is_equal_approx(preview_instance.last_saturation_multiplier, saturation_multiplier)) {
			preview_instance.material->set_shader_parameter(SNAME("saturation_multiplier"), saturation_multiplier);
			preview_instance.last_saturation_multiplier = saturation_multiplier;
		}
		if (!preview_instance.render_state_initialized || preview_instance.last_render_transform != transform) {
			RS::get_singleton()->instance_set_transform(preview_instance.instance, transform);
			preview_instance.last_render_transform = transform;
		}
		const bool instance_visible = preview_visible && source->is_visible_in_tree();
		if (!preview_instance.render_state_initialized || preview_instance.last_visible != instance_visible) {
			RS::get_singleton()->instance_set_visible(preview_instance.instance, instance_visible);
			preview_instance.last_visible = instance_visible;
		}
		preview_instance.render_state_initialized = true;
	}
}

void LightmapGIEditorPlugin::edit(Object *p_object) {
	LightmapGI *s = Object::cast_to<LightmapGI>(p_object);
	if (!s) {
		return;
	}

	if (lightmap != s && preview_lightmap_id.is_valid()) {
		_clear_preview();
	}
	lightmap = s;
	_load_target_density();
}

bool LightmapGIEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("LightmapGI");
}

void LightmapGIEditorPlugin::make_visible(bool p_visible) {
	plugin_visible = p_visible;
	if (p_visible) {
		bake->show();
	} else {
		bake->hide();
	}
	_update_preview_button(preview_lightmap_id.is_valid());
}

void LightmapGIEditorPlugin::edited_scene_changed() {
	_clear_preview();
	lightmap = nullptr;
}

EditorProgress *LightmapGIEditorPlugin::tmp_progress = nullptr;

bool LightmapGIEditorPlugin::bake_func_step(float p_progress, const String &p_description, void *, bool p_refresh) {
	if (!tmp_progress) {
		tmp_progress = memnew(EditorProgress("bake_lightmaps", TTR("Bake Lightmaps"), 1000, true));
		ERR_FAIL_NULL_V(tmp_progress, false);
	}
	return tmp_progress->step(p_description, p_progress * 1000, p_refresh);
}

void LightmapGIEditorPlugin::bake_func_end(uint64_t p_time_started) {
	if (tmp_progress != nullptr) {
		memdelete(tmp_progress);
		tmp_progress = nullptr;
	}

	const int time_taken = OS::get_singleton()->get_ticks_msec() - p_time_started;
	print_line(vformat("Done baking lightmaps in %02d:%02d:%02d.%02d.", time_taken / 3'600'000, (time_taken % 3'600'000) / 60'000, (time_taken % 60'000) / 1000, (time_taken % 1000) / 10));
	// Request attention in case the user was doing something else.
	// Baking lightmaps is likely the editor task that can take the most time,
	// so only request the attention for baking lightmaps.
	DisplayServer::get_singleton()->window_request_attention();
}

void LightmapGIEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_bake", &LightmapGIEditorPlugin::_bake);
}

LightmapGIEditorPlugin::LightmapGIEditorPlugin() {
	bake = memnew(Button);
	bake->set_theme_type_variation(SceneStringName(FlatButton));
	// TODO: Rework this as a dedicated toolbar control so we can hook into theme changes and update it
	// when the editor theme updates.
	bake->set_button_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Bake"), EditorStringName(EditorIcons)));
	bake->set_text(TTR("Bake Lightmaps"));

#ifdef MODULE_LIGHTMAPPER_RD_ENABLED
	// Disable lightmap baking if not supported on the current GPU.
	if (!DisplayServer::get_singleton()->can_create_rendering_device()) {
		bake->set_disabled(true);
		bake->set_tooltip_text(vformat(TTR("Lightmap baking is not supported on this GPU (%s)."), RenderingServer::get_singleton()->get_video_adapter_name()));
	}
#else
	// Disable lightmap baking if the module is disabled at compile-time.
	bake->set_disabled(true);
#if defined(ANDROID_ENABLED) || defined(APPLE_EMBEDDED_ENABLED)
	bake->set_tooltip_text(vformat(TTR("Lightmaps cannot be baked on %s."), OS::get_singleton()->get_name()));
#else
	bake->set_tooltip_text(TTR("Lightmaps cannot be baked, as the `lightmapper_rd` module was disabled at compile-time."));
#endif
#endif // MODULE_LIGHTMAPPER_RD_ENABLED

	bake->hide();
	bake->connect(SceneStringName(pressed), Callable(this, "_bake"));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake);

	preview_shader.instantiate();
	preview_shader->set_code(R"(
shader_type spatial;
render_mode unshaded, cull_disabled, fog_disabled;

uniform vec2 estimated_lightmap_size;
uniform float target_texel_density = 1.0;
uniform float saturation_multiplier = 1.0;

varying vec3 world_position;

void vertex() {
	world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	POSITION = PROJECTION_MATRIX * MODELVIEW_MATRIX * vec4(VERTEX, 1.0);
	POSITION.z += 0.0001 * POSITION.w;
}

void fragment() {
	vec2 checker_uv = UV2 * estimated_lightmap_size * 0.5;
	vec2 checker_width = max(fwidth(checker_uv), vec2(0.0001));
	vec2 checker_square = smoothstep(vec2(0.5) - checker_width, vec2(0.5) + checker_width, fract(checker_uv));
	float checker = abs(checker_square.x - checker_square.y);

	vec2 texel_uv = UV2 * estimated_lightmap_size;
	vec2 texel_dx = dFdx(texel_uv);
	vec2 texel_dy = dFdy(texel_uv);
	vec3 world_dx = dFdx(world_position);
	vec3 world_dy = dFdy(world_position);
	float texel_area = abs(texel_dx.x * texel_dy.y - texel_dx.y * texel_dy.x);
	float world_area = length(cross(world_dx, world_dy));
	float texel_density = sqrt(texel_area / max(world_area, 0.000001));
	float relative_texel_density = texel_density / max(target_texel_density, 0.0001);
	float relative_log = log2(max(relative_texel_density, 0.001));
	float deviation = abs(relative_log) / (abs(relative_log) + 1.0);
	vec3 below_color = vec3(0.10, 0.32, 0.85);
	vec3 target_color = vec3(0.12, 0.72, 0.30);
	vec3 above_color = mix(vec3(1.0, 0.68, 0.08), vec3(0.90, 0.12, 0.06), smoothstep(1.0, 4.0, relative_log));
	vec3 density_color = relative_log < 0.0 ? mix(target_color, below_color, deviation) : mix(target_color, above_color, deviation);
	float luminance = dot(density_color, vec3(0.299, 0.587, 0.114));
	vec3 display_color = max(mix(vec3(luminance), density_color, saturation_multiplier), vec3(0.0));
	ALBEDO = display_color * mix(0.72, 1.0, checker);
}
)");

	preview = memnew(Button);
	preview->set_theme_type_variation(SceneStringName(FlatButton));
	preview->set_text(TTR("Preview Texel Density"));
	preview->set_tooltip_text(TTR("Displays an estimate of the UV2 texel density used by the next lightmap bake."));
	preview->hide();
	preview->connect(SceneStringName(pressed), callable_mp(this, &LightmapGIEditorPlugin::_preview_pressed));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, preview);

	close_preview = memnew(Button);
	close_preview->set_theme_type_variation(SceneStringName(FlatButton));
	close_preview->set_text(TTR("Close Preview"));
	close_preview->set_tooltip_text(TTR("Closes the texel density preview."));
	close_preview->hide();
	close_preview->connect(SceneStringName(pressed), callable_mp(this, &LightmapGIEditorPlugin::_close_preview_pressed));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, close_preview);

	preview_options = memnew(Button);
	preview_options->set_theme_type_variation(SceneStringName(FlatButton));
	preview_options->set_button_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GuiTabMenuHl"), EditorStringName(EditorIcons)));
	preview_options->set_tooltip_text(TTR("Configure the target texel density."));
	preview_options->hide();
	preview_options->connect(SceneStringName(pressed), callable_mp(this, &LightmapGIEditorPlugin::_preview_options_pressed));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, preview_options);

	preview_options_popup = memnew(PopupPanel);
	preview_options->add_child(preview_options_popup);
	VBoxContainer *preview_options_vbox = memnew(VBoxContainer);
	preview_options_vbox->set_custom_minimum_size(Size2(240, 0) * EDSCALE);
	preview_options_popup->add_child(preview_options_vbox);

	Label *density_label = memnew(Label);
	density_label->set_text(TTR("Texel Density"));
	density_label->set_theme_type_variation(SNAME("HeaderSmall"));
	preview_options_vbox->add_child(density_label);

	target_density_spinbox = memnew(SpinBox);
	target_density_spinbox->set_min(0.01);
	target_density_spinbox->set_max(1024.0);
	target_density_spinbox->set_step(0.01);
	target_density_spinbox->set_allow_greater(true);
	target_density_spinbox->set_suffix(TTR(" texels/unit"));
	target_density_spinbox->set_value(target_density);
	target_density_spinbox->connect(SceneStringName(value_changed), callable_mp(this, &LightmapGIEditorPlugin::_target_density_changed));
	preview_options_vbox->add_child(target_density_spinbox);
	lightmap = nullptr;

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->add_filter("*.lmbake", TTR("LightMap Bake"));
	file_dialog->set_title(TTR("Select lightmap bake file:"));
	file_dialog->connect("file_selected", callable_mp(this, &LightmapGIEditorPlugin::_bake_select_file));
	bake->add_child(file_dialog);
}

LightmapGIEditorPlugin::~LightmapGIEditorPlugin() {
	_clear_preview();
}
