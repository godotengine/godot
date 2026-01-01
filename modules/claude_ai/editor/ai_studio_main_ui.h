/**************************************************************************/
/*  ai_studio_main_ui.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                        AI-Powered Game Engine                          */
/*                         Built on Godot Engine                          */
/**************************************************************************/

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/control.h"

class ClaudeAIDock;

/// Main UI controller for AI Studio
/// This integrates Claude AI as a core feature of the editor
class AIStudioMainUI : public Control {
	GDCLASS(AIStudioMainUI, Control)

private:
	bool is_main_view_active = false;

protected:
	static void _bind_methods();

public:
	AIStudioMainUI();
	~AIStudioMainUI();

	void show_main_view();
	void hide_main_view();
	bool is_main_view_visible() const { return is_main_view_active; }
};

