/**************************************************************************/
/*  shader_baker_export_plugin.cpp                                        */
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

#include "shader_baker_export_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/sprite_3d.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/rendering_shader_container.h"

// Ensure that AlphaCut is the same between the two classes so we can share the code to detect transparency.
static_assert(ENUM_MEMBERS_EQUAL(SpriteBase3D::ALPHA_CUT_DISABLED, Label3D::ALPHA_CUT_DISABLED));
static_assert(ENUM_MEMBERS_EQUAL(SpriteBase3D::ALPHA_CUT_DISCARD, Label3D::ALPHA_CUT_DISCARD));
static_assert(ENUM_MEMBERS_EQUAL(SpriteBase3D::ALPHA_CUT_OPAQUE_PREPASS, Label3D::ALPHA_CUT_OPAQUE_PREPASS));
static_assert(ENUM_MEMBERS_EQUAL(SpriteBase3D::ALPHA_CUT_HASH, Label3D::ALPHA_CUT_HASH));
static_assert(ENUM_MEMBERS_EQUAL(SpriteBase3D::ALPHA_CUT_MAX, Label3D::ALPHA_CUT_MAX));

String ShaderBakerExportPlugin::get_name() const {
	return "ShaderBaker";
}

bool ShaderBakerExportPlugin::_is_active(const Vector<String> &p_features) const {
	// Shader baker should only work when a RendererRD driver is active, as the embedded shaders won't be found otherwise.
	return RendererSceneRenderRD::get_singleton() != nullptr && RendererRD::MaterialStorage::get_singleton() != nullptr && p_features.has("shader_baker");
}

bool ShaderBakerExportPlugin::_initialize_container_format(const Ref<EditorExportPlatform> &p_platform, const Ref<EditorExportPreset> &p_preset) {
	shader_container_driver = p_preset->get_project_setting("rendering/rendering_device/driver");
	ERR_FAIL_COND_V_MSG(shader_container_driver.is_empty(), false, "Invalid `rendering/rendering_device/driver` setting, disabling shader baking.");

	for (Ref<ShaderBakerExportPluginPlatform> platform : platforms) {
		if (platform->matches_driver(shader_container_driver)) {
			shader_container_format = platform->create_shader_container_format(p_platform, p_preset);
			ERR_FAIL_NULL_V_MSG(shader_container_format, false, "Unable to create shader container format for the export platform.");
			return true;
		}
	}

	return false;
}

void ShaderBakerExportPlugin::_cleanup_container_format() {
	if (shader_container_format != nullptr) {
		memdelete(shader_container_format);
		shader_container_format = nullptr;
	}
}

bool ShaderBakerExportPlugin::_initialize_cache_directory() {
	shader_cache_export_path = get_export_base_path().path_join("shader_baker").path_join(shader_cache_platform_name).path_join(shader_container_driver);

	if (!DirAccess::dir_exists_absolute(shader_cache_export_path)) {
		Error err = DirAccess::make_dir_recursive_absolute(shader_cache_export_path);
		ERR_FAIL_COND_V_MSG(err != OK, false, "Can't create shader cache folder for exporting.");
	}

	return true;
}

bool ShaderBakerExportPlugin::_begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) {
	if (!_is_active(p_features)) {
		return false;
	}

	if (!_initialize_container_format(p_platform, get_export_preset())) {
		return false;
	}

	if (Engine::get_singleton()->is_generate_spirv_debug_info_enabled()) {
		WARN_PRINT("Shader baker can't generate a compatible shader when run with --generate-spirv-debug-info. Restart the editor without this argument if you want to bake shaders.");
		return false;
	}

	shader_cache_platform_name = p_platform->get_os_name();
	shader_cache_renderer_name = RendererSceneRenderRD::get_singleton()->get_name();
	tasks_processed = 0;
	tasks_total = 0;
	tasks_cancelled = false;

	StringBuilder to_hash;
	to_hash.append("[GodotVersionNumber]");
	to_hash.append(GODOT_VERSION_NUMBER);
	to_hash.append("[GodotVersionHash]");
	to_hash.append(GODOT_VERSION_HASH);
	to_hash.append("[Renderer]");
	to_hash.append(shader_cache_renderer_name);
	customization_configuration_hash = to_hash.as_string().hash64();

	BitField<RenderingShaderLibrary::FeatureBits> renderer_features = {};
#ifndef XR_DISABLED
	bool xr_enabled = GLOBAL_GET("xr/shaders/enabled");
	renderer_features.set_flag(RenderingShaderLibrary::FEATURE_ADVANCED_BIT);
	if (xr_enabled) {
		renderer_features.set_flag(RenderingShaderLibrary::FEATURE_MULTIVIEW_BIT);
	}
#endif // XR_DISABLED

	int vrs_mode = GLOBAL_GET("rendering/vrs/mode");
	if (vrs_mode != 0) {
		renderer_features.set_flag(RenderingShaderLibrary::FEATURE_VRS_BIT);
	}

	// Both FP16 and FP32 variants should be included.
	renderer_features.set_flag(RenderingShaderLibrary::FEATURE_FP16_BIT);
	renderer_features.set_flag(RenderingShaderLibrary::FEATURE_FP32_BIT);

	RendererSceneRenderRD::get_singleton()->enable_features(renderer_features);

	// Included all shaders created by renderers and effects.
	ShaderRD::shaders_embedded_set_lock();
	const ShaderRD::ShaderVersionPairSet &pair_set = ShaderRD::shaders_embedded_set_get();
	for (Pair<ShaderRD *, RID> pair : pair_set) {
		_customize_shader_version(pair.first, pair.second);
	}

	ShaderRD::shaders_embedded_set_unlock();

	// Include all shaders created by embedded materials.
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	material_storage->shader_embedded_set_lock();
	const HashSet<RID> &rid_set = material_storage->shader_embedded_set_get();
	for (RID rid : rid_set) {
		RendererRD::MaterialStorage::ShaderData *shader_data = material_storage->shader_get_data(rid);
		if (shader_data != nullptr) {
			Pair<ShaderRD *, RID> shader_version_pair = shader_data->get_native_shader_and_version();
			if (shader_version_pair.first != nullptr) {
				_customize_shader_version(shader_version_pair.first, shader_version_pair.second);
			}
		}
	}

	material_storage->shader_embedded_set_unlock();

	return true;
}

bool ShaderBakerExportPlugin::_begin_customize_scenes(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) {
	if (!_is_active(p_features)) {
		return false;
	}

	if (shader_container_format == nullptr) {
		// Resource customization failed to initialize.
		return false;
	}

	return true;
}

void ShaderBakerExportPlugin::_end_customize_resources() {
	if (!_initialize_cache_directory()) {
		return;
	}

	// Run a progress bar that waits for all shader baking tasks to finish.
	bool progress_active = true;
	EditorProgress editor_progress("baking_shaders", TTR("Baking shaders"), tasks_total);
	editor_progress.step("Baking...", 0);
	while (progress_active) {
		uint32_t tasks_for_progress = 0;
		{
			MutexLock lock(tasks_mutex);
			if (tasks_processed >= tasks_total) {
				progress_active = false;
			} else {
				tasks_condition.wait(lock);
				tasks_for_progress = tasks_processed;
			}
		}

		if (progress_active && editor_progress.step("Baking...", tasks_for_progress)) {
			// User skipped the shader baker, we just don't pack the shaders in the project.
			tasks_cancelled = true;
			progress_active = false;
		}
	}

	String shader_cache_user_dir = ShaderRD::get_shader_cache_user_dir();
	for (const ShaderGroupItem &group_item : shader_group_items) {
		// Wait for all shader compilation tasks of the group to be finished.
		for (WorkerThreadPool::TaskID task_id : group_item.variant_tasks) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(task_id);
		}

		if (!tasks_cancelled) {
			WorkResult work_result;
			{
				MutexLock lock(shader_work_results_mutex);
				work_result = shader_work_results[group_item.cache_path];
			}

			PackedByteArray cache_file_bytes = ShaderRD::save_shader_cache_bytes(group_item.variants, work_result.variant_data);
			add_file(shader_cache_user_dir.path_join(group_item.cache_path), cache_file_bytes, false);

			String cache_file_path = shader_cache_export_path.path_join(group_item.cache_path);
			if (!DirAccess::exists(cache_file_path)) {
				DirAccess::make_dir_recursive_absolute(cache_file_path.get_base_dir());
			}

			Ref<FileAccess> cache_file_access = FileAccess::open(cache_file_path, FileAccess::WRITE);
			if (cache_file_access.is_valid()) {
				cache_file_access->store_buffer(cache_file_bytes);
			}
		}
	}

	if (!tasks_cancelled) {
		String file_cache_path = shader_cache_export_path.path_join("file_cache");
		Ref<FileAccess> cache_list_access = FileAccess::open(file_cache_path, FileAccess::READ_WRITE);
		if (cache_list_access.is_null()) {
			cache_list_access = FileAccess::open(file_cache_path, FileAccess::WRITE);
		}

		if (cache_list_access.is_valid()) {
			String cache_list_line;
			while (cache_list_line = cache_list_access->get_line(), !cache_list_line.is_empty()) {
				// Only add if it wasn't already added.
				if (!shader_paths_processed.has(cache_list_line)) {
					PackedByteArray cache_file_bytes = FileAccess::get_file_as_bytes(shader_cache_export_path.path_join(cache_list_line));
					if (!cache_file_bytes.is_empty()) {
						add_file(shader_cache_user_dir.path_join(cache_list_line), cache_file_bytes, false);
					}
				}

				shader_paths_processed.erase(cache_list_line);
			}

			for (const String &shader_path : shader_paths_processed) {
				cache_list_access->store_line(shader_path);
			}

			cache_list_access->close();
		}
	}

	shader_paths_processed.clear();
	shader_work_results.clear();
	shader_group_items.clear();

	_cleanup_container_format();
}

Ref<Resource> ShaderBakerExportPlugin::_customize_resource(const Ref<Resource> &p_resource, const String &p_path) {
	RendererRD::MaterialStorage *singleton = RendererRD::MaterialStorage::get_singleton();
	DEV_ASSERT(singleton != nullptr);

	Ref<Material> material = p_resource;
	if (material.is_valid()) {
		RID material_rid = material->get_rid();
		if (material_rid.is_valid()) {
			RendererRD::MaterialStorage::ShaderData *shader_data = singleton->material_get_shader_data(material_rid);
			if (shader_data != nullptr) {
				Pair<ShaderRD *, RID> shader_version_pair = shader_data->get_native_shader_and_version();
				if (shader_version_pair.first != nullptr) {
					_customize_shader_version(shader_version_pair.first, shader_version_pair.second);
				}
			}
		}
	}

	return Ref<Resource>();
}

Node *ShaderBakerExportPlugin::_customize_scene(Node *p_root, const String &p_path) {
	LocalVector<Node *> nodes_to_visit;
	nodes_to_visit.push_back(p_root);
	while (!nodes_to_visit.is_empty()) {
		// Visit all nodes recursively in the scene to find the Label3Ds and Sprite3Ds.
		Node *node = nodes_to_visit[nodes_to_visit.size() - 1];
		nodes_to_visit.remove_at(nodes_to_visit.size() - 1);

		Label3D *label_3d = Object::cast_to<Label3D>(node);
		Sprite3D *sprite_3d = Object::cast_to<Sprite3D>(node);
		if (label_3d != nullptr || sprite_3d != nullptr) {
			// Create materials for Label3D and Sprite3D, which are normally generated at runtime on demand.
			HashMap<StringName, Variant> properties;

			// These must match the defaults set by Sprite3D/Label3D.
			properties["transparent"] = true; // Label3D doesn't have this property, but it is always true anyway.
			properties["shaded"] = false;
			properties["double_sided"] = true;
			properties["no_depth_test"] = false;
			properties["fixed_size"] = false;
			properties["billboard"] = StandardMaterial3D::BILLBOARD_DISABLED;
			properties["texture_filter"] = StandardMaterial3D::TEXTURE_FILTER_LINEAR_WITH_MIPMAPS;
			properties["alpha_antialiasing_mode"] = StandardMaterial3D::ALPHA_ANTIALIASING_OFF;
			properties["alpha_cut"] = SpriteBase3D::ALPHA_CUT_DISABLED;

			List<PropertyInfo> property_list;
			node->get_property_list(&property_list);
			for (const PropertyInfo &info : property_list) {
				bool valid = false;
				Variant property = node->get(info.name, &valid);
				if (valid) {
					properties[info.name] = property;
				}
			}

			// This must follow the logic in Sprite3D::draw_texture_rect().
			BaseMaterial3D::Transparency mat_transparency = BaseMaterial3D::Transparency::TRANSPARENCY_DISABLED;
			if (properties["transparent"]) {
				SpriteBase3D::AlphaCutMode acm = SpriteBase3D::AlphaCutMode(int(properties["alpha_cut"]));
				if (acm == SpriteBase3D::ALPHA_CUT_DISCARD) {
					mat_transparency = BaseMaterial3D::Transparency::TRANSPARENCY_ALPHA_SCISSOR;
				} else if (acm == SpriteBase3D::ALPHA_CUT_OPAQUE_PREPASS) {
					mat_transparency = BaseMaterial3D::Transparency::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS;
				} else if (acm == SpriteBase3D::ALPHA_CUT_HASH) {
					mat_transparency = BaseMaterial3D::Transparency::TRANSPARENCY_ALPHA_HASH;
				} else {
					mat_transparency = BaseMaterial3D::Transparency::TRANSPARENCY_ALPHA;
				}
			}

			StandardMaterial3D::BillboardMode billboard_mode = StandardMaterial3D::BillboardMode(int(properties["billboard"]));
			Ref<Material> sprite_3d_material = StandardMaterial3D::get_material_for_2d(bool(properties["shaded"]), mat_transparency, bool(properties["double_sided"]), billboard_mode == StandardMaterial3D::BILLBOARD_ENABLED, billboard_mode == StandardMaterial3D::BILLBOARD_FIXED_Y, false, bool(properties["no_depth_test"]), bool(properties["fixed_size"]), BaseMaterial3D::TextureFilter(int(properties["texture_filter"])), BaseMaterial3D::AlphaAntiAliasing(int(properties["alpha_antialiasing_mode"])));
			_customize_resource(sprite_3d_material, String());

			if (label_3d != nullptr) {
				// Generate variants with and without MSDF support since we don't have access to the font here.
				Ref<Material> label_3d_material = StandardMaterial3D::get_material_for_2d(bool(properties["shaded"]), mat_transparency, bool(properties["double_sided"]), billboard_mode == StandardMaterial3D::BILLBOARD_ENABLED, billboard_mode == StandardMaterial3D::BILLBOARD_FIXED_Y, true, bool(properties["no_depth_test"]), bool(properties["fixed_size"]), BaseMaterial3D::TextureFilter(int(properties["texture_filter"])), BaseMaterial3D::AlphaAntiAliasing(int(properties["alpha_antialiasing_mode"])));
				_customize_resource(label_3d_material, String());
			}
		}

		// Visit children.
		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			nodes_to_visit.push_back(node->get_child(i));
		}
	}

	return nullptr;
}

uint64_t ShaderBakerExportPlugin::_get_customization_configuration_hash() const {
	return customization_configuration_hash;
}

void ShaderBakerExportPlugin::_customize_shader_version(ShaderRD *p_shader, RID p_version) {
	const int64_t variant_count = p_shader->get_variant_count();
	const int64_t group_count = p_shader->get_group_count();
	LocalVector<ShaderGroupItem> group_items;
	group_items.resize(group_count);

	RBSet<uint32_t> groups_to_compile;
	for (int64_t i = 0; i < group_count; i++) {
		if (!p_shader->is_group_enabled(i)) {
			continue;
		}

		String cache_path = p_shader->version_get_cache_file_relative_path(p_version, i, shader_container_driver);
		if (shader_paths_processed.has(cache_path)) {
			continue;
		}

		shader_paths_processed.insert(cache_path);
		groups_to_compile.insert(i);

		group_items[i].cache_path = cache_path;
		group_items[i].variants = p_shader->get_group_to_variants(i);

		{
			MutexLock lock(shader_work_results_mutex);
			shader_work_results[cache_path].variant_data.resize(variant_count);
		}
	}

	for (int64_t i = 0; i < variant_count; i++) {
		int group = p_shader->get_variant_to_group(i);
		if (!p_shader->is_variant_enabled(i) || !groups_to_compile.has(group)) {
			continue;
		}

		WorkItem work_item;
		work_item.cache_path = group_items[group].cache_path;
		work_item.shader_name = p_shader->get_name();
		work_item.stage_sources = p_shader->version_build_variant_stage_sources(p_version, i);
		work_item.dynamic_buffers = p_shader->get_dynamic_buffers();
		work_item.variant = i;

		WorkerThreadPool::TaskID task_id = WorkerThreadPool::get_singleton()->add_template_task(this, &ShaderBakerExportPlugin::_process_work_item, work_item);
		group_items[group].variant_tasks.push_back(task_id);
		tasks_total++;
	}

	for (uint32_t i : groups_to_compile) {
		shader_group_items.push_back(group_items[i]);
	}
}

void ShaderBakerExportPlugin::_process_work_item(WorkItem p_work_item) {
	if (!tasks_cancelled) {
		// Only process the item if the tasks haven't been cancelled by the user yet.
		Vector<RD::ShaderStageSPIRVData> spirv_data = ShaderRD::compile_stages(p_work_item.stage_sources, p_work_item.dynamic_buffers);
		if (unlikely(spirv_data.is_empty())) {
			ERR_PRINT("Unable to retrieve SPIR-V data for shader.");
		} else {
			Ref<RenderingShaderContainer> shader_container = shader_container_format->create_container();

			// Compile shader binary from SPIR-V.
			bool code_compiled = shader_container->set_code_from_spirv(p_work_item.shader_name, spirv_data);
			if (unlikely(!code_compiled)) {
				ERR_PRINT("Failed to compile code to native for SPIR-V.");
			} else {
				PackedByteArray shader_bytes = shader_container->to_bytes();
				{
					MutexLock lock(shader_work_results_mutex);
					shader_work_results[p_work_item.cache_path].variant_data.ptrw()[p_work_item.variant] = shader_bytes;
				}
			}
		}
	}

	{
		MutexLock lock(tasks_mutex);
		tasks_processed++;
	}

	tasks_condition.notify_one();
}

ShaderBakerExportPlugin::ShaderBakerExportPlugin() {
	// Do nothing.
}

ShaderBakerExportPlugin::~ShaderBakerExportPlugin() {
	// Do nothing.
}

void ShaderBakerExportPlugin::add_platform(Ref<ShaderBakerExportPluginPlatform> p_platform) {
	platforms.push_back(p_platform);
}

void ShaderBakerExportPlugin::remove_platform(Ref<ShaderBakerExportPluginPlatform> p_platform) {
	platforms.erase(p_platform);
}
