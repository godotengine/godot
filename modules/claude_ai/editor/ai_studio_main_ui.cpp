/**************************************************************************/
/*  ai_studio_main_ui.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                        AI-Powered Game Engine                          */
/*                         Built on Godot Engine                          */
/**************************************************************************/

#include "ai_studio_main_ui.h"

#include "claude_ai_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/rich_text_label.h"

void AIStudioMainUI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("show_main_view"), &AIStudioMainUI::show_main_view);
	ClassDB::bind_method(D_METHOD("hide_main_view"), &AIStudioMainUI::hide_main_view);
	ClassDB::bind_method(D_METHOD("is_main_view_visible"), &AIStudioMainUI::is_main_view_visible);
}

AIStudioMainUI::AIStudioMainUI() {
	set_name("AIStudioMainUI");
}

AIStudioMainUI::~AIStudioMainUI() {
}

void AIStudioMainUI::show_main_view() {
	is_main_view_active = true;
}

void AIStudioMainUI::hide_main_view() {
	is_main_view_active = false;
}

