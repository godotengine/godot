/**
 * limbo_string_names.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_STRING_NAMES_H
#define LIMBO_STRING_NAMES_H

#ifdef LIMBOAI_MODULE
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/typedefs.h"
#include "modules/register_module_types.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/variant/string.hpp"
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string_name.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class LimboStringNames {
public:

	static LimboStringNames *singleton;

	static void create() { singleton = memnew(LimboStringNames); }
	static void free() {
		memdelete(singleton);
		singleton = nullptr;
	}

	LimboStringNames();

public:
	_FORCE_INLINE_ static LimboStringNames *get_singleton() { return singleton; }

	StringName _enter;
	StringName _exit;
	StringName _generate_name;
	StringName _get_configuration_warnings;
	StringName _replace_task;
	StringName _setup;
	StringName _tick;
	StringName _update_banners;
	StringName _update;
	StringName _weight_;
	StringName accent_color;
	StringName ActionCopy;
	StringName ActionCut;
	StringName ActionPaste;
	StringName active_state_changed;
	StringName add_child_at_index;
	StringName add_child;
	StringName Add;
	StringName AnimationFilter;
	StringName Back;
	StringName behavior_tree_finished;
	StringName behavior_tree;
	StringName BehaviorTree;
	StringName bold;
	StringName BTAlwaysFail;
	StringName BTAlwaysSucceed;
	StringName button_down;
	StringName button_up;
	StringName call_deferred;
	StringName changed;
	StringName connect;
	StringName dark_color_1;
	StringName dark_color_2;
	StringName Debug;
	StringName disabled_font_color;
	StringName doc_italic;
	StringName draw;
	StringName Duplicate;
	StringName Edit;
	StringName EditAddRemove;
	StringName Editor;
	StringName EditorFonts;
	StringName EditorIcons;
	StringName EditorStyles;
	StringName emit_changed;
	StringName entered;
	StringName error_value;
	StringName EVENT_FAILURE;
	StringName EVENT_FINISHED;
	StringName EVENT_SUCCESS;
	StringName exited;
	StringName favorite_tasks_changed;
	StringName Favorites;
	StringName focus_exited;
	StringName font_color;
	StringName font_size;
	StringName font;
	StringName Forward;
	StringName gui_input;
	StringName GuiOptionArrow;
	StringName GuiTreeArrowDown;
	StringName GuiTreeArrowRight;
	StringName HeaderSmall;
	StringName Help;
	StringName icon_max_width;
	StringName id_pressed;
	StringName Info;
	StringName item_collapsed;
	StringName item_selected;
	StringName LimboDeselectAll;
	StringName LimboExtraClock;
	StringName LimboExtractSubtree;
	StringName LimboPercent;
	StringName LimboSelectAll;
	StringName LimboVarAdd;
	StringName LimboVarEmpty;
	StringName LimboVarError;
	StringName LimboVarExists;
	StringName LimboVarNotFound;
	StringName LimboVarPrivate;
	StringName LineEdit;
	StringName Load;
	StringName managed;
	StringName mode_changed;
	StringName mouse_entered;
	StringName mouse_exited;
	StringName MoveDown;
	StringName MoveUp;
	StringName New;
	StringName NewRoot;
	StringName NodeWarning;
	StringName NonFavorite;
	StringName normal;
	StringName panel;
	StringName plan_changed;
	StringName popup_hide;
	StringName pressed;
	StringName probability_clicked;
	StringName refresh;
	StringName Reload;
	StringName remove_child;
	StringName Remove;
	StringName Rename;
	StringName request_open_in_screen;
	StringName rmb_pressed;
	StringName Save;
	StringName Script;
	StringName ScriptCreate;
	StringName Search;
	StringName separation;
	StringName set_custom_name;
	StringName set_root_task;
	StringName set_v_scroll;
	StringName setup;
	StringName started;
	StringName StatusWarning;
	StringName stopped;
	StringName task_activated;
	StringName task_button_pressed;
	StringName task_button_rmb;
	StringName task_dragged;
	StringName task_meta;
	StringName task_selected;
	StringName text_changed;
	StringName text_submitted;
	StringName timeout;
	StringName toggled;
	StringName Tools;
	StringName Tree;
	StringName TripleBar;
	StringName update_task;
	StringName update_tree;
	StringName updated;
	StringName visibility_changed;
	StringName window_visibility_changed;

	String repeat_forever;
	String output_var_prefix;

	NodePath node_pp;
};

#define LW_NAME(m_arg) LimboStringNames::get_singleton()->m_arg

#endif // LIMBO_STRING_NAMES_H
