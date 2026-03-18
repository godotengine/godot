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

#include "multiplayer_debugger.h"
#include "multiplayer_spawner.h"
#include "multiplayer_synchronizer.h"
#include "scene_multiplayer.h"
#include "scene_replication_interface.h"
#include "scene_rpc_interface.h"

#ifdef TOOLS_ENABLED
#include "editor/multiplayer_editor_plugin.h"
#endif

void initialize_multiplayer_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(SceneReplicationConfig);
		GDREGISTER_CLASS(MultiplayerSpawner);
		GDREGISTER_CLASS(MultiplayerSynchronizer);
		GDREGISTER_CLASS(OfflineMultiplayerPeer);
		GDREGISTER_CLASS(SceneMultiplayer);
		if (GD_IS_CLASS_ENABLED(MultiplayerAPI)) {
			MultiplayerAPI::set_default_interface("SceneMultiplayer");
			MultiplayerDebugger::initialize();
		}
	}
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<MultiplayerEditorPlugin>();
	}
#endif
}

void uninitialize_multiplayer_module(ModuleInitializationLevel p_level) {
	if (GD_IS_CLASS_ENABLED(MultiplayerAPI)) {
		MultiplayerDebugger::deinitialize();
	}
}
