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

#include "structures/usd_animation.h"
#include "structures/usd_camera.h"
#include "structures/usd_document.h"
#include "structures/usd_light.h"
#include "structures/usd_material.h"
#include "structures/usd_materialx_converter.h"
#include "structures/usd_mesh.h"
#include "structures/usd_node.h"
#include "structures/usd_skeleton.h"
#include "structures/usd_state.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scene_importer_usd.h"

#include "editor/editor_node.h"

static void _editor_init() {
	Ref<EditorSceneFormatImporterUSD> import_usd;
	import_usd.instantiate();
	ResourceImporterScene::add_scene_importer(import_usd);
}
#endif // TOOLS_ENABLED

void initialize_usd_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(USDNode);
		GDREGISTER_CLASS(USDMesh);
		GDREGISTER_CLASS(USDMaterial);
		GDREGISTER_CLASS(USDMaterialXConverter);
		GDREGISTER_CLASS(USDLight);
		GDREGISTER_CLASS(USDCamera);
		GDREGISTER_CLASS(USDSkeleton);
		GDREGISTER_CLASS(USDAnimation);
		GDREGISTER_CLASS(USDState);
		GDREGISTER_CLASS(USDDocument);
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		GDREGISTER_CLASS(EditorSceneFormatImporterUSD);

		EditorNode::add_init_callback(_editor_init);
	}
#endif // TOOLS_ENABLED
}

void uninitialize_usd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
