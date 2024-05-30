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

#include "src/retarget_animation_player.h"
#include "src/retarget_animation_tree.h"
#include "src/retarget_pose_transporter.h"
#include "src/retarget_profile.h"
#include "src/retarget_utility.h"

#ifdef TOOLS_ENABLED
#include "editor/post_import_plugin_realtime_retarget.h"
#include "editor/realtime_retarget_editor_plugin.h"
#endif

void initialize_realtime_retarget_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		GDREGISTER_CLASS(RetargetUtility);
	}
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(RetargetProfile);
		GDREGISTER_CLASS(RetargetProfileGlobalAll);
		GDREGISTER_CLASS(RetargetProfileLocalAll);
		GDREGISTER_CLASS(RetargetProfileAbsoluteAll);
		GDREGISTER_CLASS(RetargetProfileLocalFingersGlobalOthers);
		GDREGISTER_CLASS(RetargetProfileLocalLimbsGlobalOthers);
		GDREGISTER_CLASS(RetargetProfileAbsoluteFingersGlobalOthers);
		GDREGISTER_CLASS(RetargetProfileAbsoluteLimbsGlobalOthers);
		GDREGISTER_CLASS(RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers);
		GDREGISTER_CLASS(RetargetAnimationPlayer);
		GDREGISTER_CLASS(RetargetAnimationTree);
		GDREGISTER_CLASS(RetargetPoseTransporter);
	}
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<RetargetAnimationPlayerEditorPlugin>();
		EditorPlugins::add_by_type<RealtimeRetargetEditorPlugin>();
	}
#endif
}

void uninitialize_realtime_retarget_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
