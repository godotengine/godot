/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "register_types.h"

#ifndef _3D_DISABLED

#include "extensions/gltf_light.h"
#include "extensions/gltf_spec_gloss.h"
#include "gltf_document.h"
#include "gltf_document_extension.h"
#include "gltf_document_extension_convert_importer_mesh.h"
#include "gltf_state.h"
#include "structures/gltf_accessor.h"
#include "structures/gltf_animation.h"
#include "structures/gltf_buffer_view.h"
#include "structures/gltf_camera.h"
#include "structures/gltf_mesh.h"
#include "structures/gltf_node.h"
#include "structures/gltf_skeleton.h"
#include "structures/gltf_skin.h"
#include "structures/gltf_texture.h"

#ifdef TOOLS_ENABLED
#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_scene_exporter_gltf_plugin.h"
#include "editor/editor_scene_importer_blend.h"
#include "editor/editor_scene_importer_fbx.h"
#include "editor/editor_scene_importer_gltf.h"
#include "editor/editor_settings.h"

static void _editor_init() {
	Ref<EditorSceneFormatImporterGLTF> import_gltf;
	import_gltf.instantiate();
	ResourceImporterScene::add_importer(import_gltf);

	// Blend to glTF importer.

	bool blend_enabled = GLOBAL_GET("filesystem/import/blender/enabled");
	// Defined here because EditorSettings doesn't exist in `register_gltf_types` yet.
	EDITOR_DEF_RST("filesystem/import/blender/blender3_path", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,
			"filesystem/import/blender/blender3_path", PROPERTY_HINT_GLOBAL_DIR));
	if (blend_enabled) {
		Ref<EditorSceneFormatImporterBlend> importer;
		importer.instantiate();
		ResourceImporterScene::add_importer(importer);

		Ref<EditorFileSystemImportFormatSupportQueryBlend> blend_import_query;
		blend_import_query.instantiate();
		EditorFileSystem::get_singleton()->add_import_format_support_query(blend_import_query);
	}

	// FBX to glTF importer.

	bool fbx_enabled = GLOBAL_GET("filesystem/import/fbx/enabled");
	// Defined here because EditorSettings doesn't exist in `register_gltf_types` yet.
	String fbx2gltf_path = EDITOR_DEF_RST("filesystem/import/fbx/fbx2gltf_path", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,
			"filesystem/import/fbx/fbx2gltf_path", PROPERTY_HINT_GLOBAL_FILE));
	if (fbx_enabled) {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (fbx2gltf_path.is_empty()) {
			WARN_PRINT("FBX file import is enabled, but no FBX2glTF path is configured. FBX files will not be imported.");
		} else if (!da->file_exists(fbx2gltf_path)) {
			WARN_PRINT("FBX file import is enabled, but the FBX2glTF path doesn't point to a valid FBX2glTF executable. FBX files will not be imported.");
		} else {
			Ref<EditorSceneFormatImporterFBX> importer;
			importer.instantiate();
			ResourceImporterScene::add_importer(importer);
		}
	}
}
#endif // TOOLS_ENABLED

void initialize_gltf_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// glTF API available at runtime.
		GDREGISTER_CLASS(GLTFAccessor);
		GDREGISTER_CLASS(GLTFAnimation);
		GDREGISTER_CLASS(GLTFBufferView);
		GDREGISTER_CLASS(GLTFCamera);
		GDREGISTER_CLASS(GLTFDocument);
		GDREGISTER_CLASS(GLTFDocumentExtension);
		GDREGISTER_CLASS(GLTFDocumentExtensionConvertImporterMesh);
		GDREGISTER_CLASS(GLTFLight);
		GDREGISTER_CLASS(GLTFMesh);
		GDREGISTER_CLASS(GLTFNode);
		GDREGISTER_CLASS(GLTFSkeleton);
		GDREGISTER_CLASS(GLTFSkin);
		GDREGISTER_CLASS(GLTFSpecGloss);
		GDREGISTER_CLASS(GLTFState);
		GDREGISTER_CLASS(GLTFTexture);
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		// Editor-specific API.
		ClassDB::APIType prev_api = ClassDB::get_current_api();
		ClassDB::set_current_api(ClassDB::API_EDITOR);

		GDREGISTER_CLASS(EditorSceneFormatImporterGLTF);
		EditorPlugins::add_by_type<SceneExporterGLTFPlugin>();

		// Project settings defined here so doctool finds them.
		GLOBAL_DEF_RST("filesystem/import/blender/enabled", true);
		GLOBAL_DEF_RST("filesystem/import/fbx/enabled", true);
		GDREGISTER_CLASS(EditorSceneFormatImporterBlend);
		GDREGISTER_CLASS(EditorSceneFormatImporterFBX);

		ClassDB::set_current_api(prev_api);
		EditorNode::add_init_callback(_editor_init);
	}

#endif // TOOLS_ENABLED
}

void uninitialize_gltf_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

#endif // _3D_DISABLED
