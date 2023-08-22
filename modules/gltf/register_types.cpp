/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "extensions/gltf_document_extension_convert_importer_mesh.h"
#include "extensions/gltf_document_extension_texture_ktx.h"
#include "extensions/gltf_document_extension_texture_webp.h"
#include "extensions/gltf_spec_gloss.h"
#include "extensions/physics/gltf_document_extension_physics.h"
#include "gltf_document.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_import_blend_runner.h"
#include "editor/editor_scene_exporter_gltf_plugin.h"
#include "editor/editor_scene_importer_blend.h"
#include "editor/editor_scene_importer_fbx.h"
#include "editor/editor_scene_importer_gltf.h"

#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"

static void _editor_init() {
	Ref<EditorSceneFormatImporterGLTF> import_gltf;
	import_gltf.instantiate();
	ResourceImporterScene::add_importer(import_gltf);

	// Blend to glTF importer.

	bool blend_enabled = GLOBAL_GET("filesystem/import/blender/enabled");
	String blender3_path = EDITOR_GET("filesystem/import/blender/blender3_path");
	if (blend_enabled) {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (blender3_path.is_empty()) {
			WARN_PRINT(TTR("Blend file import is enabled in the project settings, but no Blender path is configured in the editor settings. Blend files will not be imported."));
		} else if (!da->dir_exists(blender3_path)) {
			WARN_PRINT(TTR("Blend file import is enabled, but the Blender path doesn't point to an accessible directory. Blend files will not be imported."));
		} else {
			Ref<EditorSceneFormatImporterBlend> importer;
			importer.instantiate();
			ResourceImporterScene::add_importer(importer);

			Ref<EditorFileSystemImportFormatSupportQueryBlend> blend_import_query;
			blend_import_query.instantiate();
			EditorFileSystem::get_singleton()->add_import_format_support_query(blend_import_query);
		}
	}
	memnew(EditorImportBlendRunner);
	EditorNode::get_singleton()->add_child(EditorImportBlendRunner::get_singleton());

	// FBX to glTF importer.

	bool fbx_enabled = GLOBAL_GET("filesystem/import/fbx/enabled");
	if (fbx_enabled) {
		Ref<EditorSceneFormatImporterFBX> importer;
		importer.instantiate();
		ResourceImporterScene::get_scene_singleton()->add_importer(importer);

		Ref<EditorFileSystemImportFormatSupportQueryFBX> fbx_import_query;
		fbx_import_query.instantiate();
		EditorFileSystem::get_singleton()->add_import_format_support_query(fbx_import_query);
	}
}
#endif // TOOLS_ENABLED

#define GLTF_REGISTER_DOCUMENT_EXTENSION(m_doc_ext_class) \
	Ref<m_doc_ext_class> extension_##m_doc_ext_class;     \
	extension_##m_doc_ext_class.instantiate();            \
	GLTFDocument::register_gltf_document_extension(extension_##m_doc_ext_class);

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
		GDREGISTER_CLASS(GLTFPhysicsBody);
		GDREGISTER_CLASS(GLTFPhysicsShape);
		GDREGISTER_CLASS(GLTFSkeleton);
		GDREGISTER_CLASS(GLTFSkin);
		GDREGISTER_CLASS(GLTFSpecGloss);
		GDREGISTER_CLASS(GLTFState);
		GDREGISTER_CLASS(GLTFTexture);
		GDREGISTER_CLASS(GLTFTextureSampler);
		// Register GLTFDocumentExtension classes with GLTFDocument.
		GLTF_REGISTER_DOCUMENT_EXTENSION(GLTFDocumentExtensionPhysics);
		GLTF_REGISTER_DOCUMENT_EXTENSION(GLTFDocumentExtensionTextureKTX);
		GLTF_REGISTER_DOCUMENT_EXTENSION(GLTFDocumentExtensionTextureWebP);
		bool is_editor = ::Engine::get_singleton()->is_editor_hint();
		if (!is_editor) {
			GLTF_REGISTER_DOCUMENT_EXTENSION(GLTFDocumentExtensionConvertImporterMesh);
		}
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		// Editor-specific API.
		ClassDB::APIType prev_api = ClassDB::get_current_api();
		ClassDB::set_current_api(ClassDB::API_EDITOR);

		GDREGISTER_CLASS(EditorSceneFormatImporterGLTF);
		EditorPlugins::add_by_type<SceneExporterGLTFPlugin>();

		// Project settings defined here so doctool finds them.
		GLOBAL_DEF_RST_BASIC("filesystem/import/blender/enabled", true);
		GLOBAL_DEF_RST_BASIC("filesystem/import/fbx/enabled", true);
		GDREGISTER_CLASS(EditorSceneFormatImporterBlend);
		GDREGISTER_CLASS(EditorSceneFormatImporterFBX);
		// Can't (a priori) run external app on these platforms.
		GLOBAL_DEF_RST("filesystem/import/blender/enabled.android", false);
		GLOBAL_DEF_RST("filesystem/import/blender/enabled.web", false);
		GLOBAL_DEF_RST("filesystem/import/fbx/enabled.android", false);
		GLOBAL_DEF_RST("filesystem/import/fbx/enabled.web", false);

		ClassDB::set_current_api(prev_api);
		EditorNode::add_init_callback(_editor_init);
	}

#endif // TOOLS_ENABLED
}

void uninitialize_gltf_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GLTFDocument::unregister_all_gltf_document_extensions();
}
