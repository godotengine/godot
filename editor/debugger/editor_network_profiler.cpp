/*************************************************************************/
/*  editor_network_profiler.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_network_profiler.h"

#include "core/os/os.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void EditorNetworkProfiler::_bind_methods() {
	ADD_SIGNAL(MethodInfo("enable_profiling", PropertyInfo(Variant::BOOL, "enable")));
}

void EditorNetworkProfiler::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		activate->set_icon(get_theme_icon("Play", "EditorIcons"));
		clear_button->set_icon(get_theme_icon("Clear", "EditorIcons"));
		incoming_bandwidth_text->set_right_icon(get_theme_icon("ArrowDown", "EditorIcons"));
		outgoing_bandwidth_text->set_right_icon(get_theme_icon("ArrowUp", "EditorIcons"));

		// This needs to be done here to set the faded color when the profiler is first opened
		incoming_bandwidth_text->add_theme_color_override("font_color_uneditable", get_theme_color("font_color", "Editor") * Color(1, 1, 1, 0.5));
		outgoing_bandwidth_text->add_theme_color_override("font_color_uneditable", get_theme_color("font_color", "Editor") * Color(1, 1, 1, 0.5));
	}
}

void EditorNetworkProfiler::_update_frame() {
	counters_display->clear();

	TreeItem *root = counters_display->create_item();

	for (Map<ObjectID, DebuggerMarshalls::MultiplayerNodeInfo>::Element *E = nodes_data.front(); E; E = E->next()) {
		TreeItem *node = counters_display->create_item(root);

		for (int j = 0; j < counters_display->get_columns(); ++j) {
			node->set_text_align(j, j > 0 ? TreeItem::ALIGN_RIGHT : TreeItem::ALIGN_LEFT);
		}

		node->set_text(0, E->get().node_path);
		node->set_text(1, E->get().incoming_rpc == 0 ? "-" : itos(E->get().incoming_rpc));
		node->set_text(2, E->get().incoming_rset == 0 ? "-" : itos(E->get().incoming_rset));
		node->set_text(3, E->get().outgoing_rpc == 0 ? "-" : itos(E->get().outgoing_rpc));
		node->set_text(4, E->get().outgoing_rset == 0 ? "-" : itos(E->get().outgoing_rset));
	}
}

void EditorNetworkProfiler::_activate_pressed() {
	if (activate->is_pressed()) {
		activate->set_icon(get_theme_icon("Stop", "EditorIcons"));
		activate->set_text(TTR("Stop"));
	} else {
		activate->set_icon(get_theme_icon("Play", "EditorIcons"));
		activate->set_text(TTR("Start"));
	}
	emit_signal("enable_profiling", activate->is_pressed());
}

void EditorNetworkProfiler::_clear_pressed() {
	nodes_data.clear();
	set_bandwidth(0, 0);
	if (frame_delay->is_stopped()) {
		frame_delay->set_wait_time(0.1);
		frame_delay->start();
	}
}

void EditorNetworkProfiler::add_node_frame_data(const DebuggerMarshalls::MultiplayerNodeInfo p_frame) {
	if (!nodes_data.has(p_frame.node)) {
		nodes_data.insert(p_frame.node, p_frame);
	} else {
		nodes_data[p_frame.node].incoming_rpc += p_frame.incoming_rpc;
		nodes_data[p_frame.node].incoming_rset += p_frame.incoming_rset;
		nodes_data[p_frame.node].outgoing_rpc += p_frame.outgoing_rpc;
		nodes_data[p_frame.node].outgoing_rset += p_frame.outgoing_rset;
	}

	if (frame_delay->is_stopped()) {
		frame_delay->set_wait_time(0.1);
		frame_delay->start();
	}
}

void EditorNetworkProfiler::set_bandwidth(int p_incoming, int p_outgoing) {
	incoming_bandwidth_text->set_text(vformat(TTR("%s/s"), String::humanize_size(p_incoming)));
	outgoing_bandwidth_text->set_text(vformat(TTR("%s/s"), String::humanize_size(p_outgoing)));

	// Make labels more prominent when the bandwidth is greater than 0 to attract user attention
	incoming_bandwidth_text->add_theme_color_override(
			"font_color_uneditable",
			get_theme_color("font_color", "Editor") * Color(1, 1, 1, p_incoming > 0 ? 1 : 0.5));
	outgoing_bandwidth_text->add_theme_color_override(
			"font_color_uneditable",
			get_theme_color("font_color", "Editor") * Color(1, 1, 1, p_outgoing > 0 ? 1 : 0.5));
}

bool EditorNetworkProfiler::is_profiling() {
	return activate->is_pressed();
}

EditorNetworkProfiler::EditorNetworkProfiler() {
	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_theme_constant_override("separation", 8 * EDSCALE);
	add_child(hb);

	activate = memnew(Button);
	activate->set_toggle_mode(true);
	activate->set_text(TTR("Start"));
	activate->connect("pressed", callable_mp(this, &EditorNetworkProfiler::_activate_pressed));
	hb->add_child(activate);

	clear_button = memnew(Button);
	clear_button->set_text(TTR("Clear"));
	clear_button->connect("pressed", callable_mp(this, &EditorNetworkProfiler::_clear_pressed));
	hb->add_child(clear_button);

	hb->add_spacer();

	Label *lb = memnew(Label);
	lb->set_text(TTR("Down"));
	hb->add_child(lb);

	incoming_bandwidth_text = memnew(LineEdit);
	incoming_bandwidth_text->set_editable(false);
	incoming_bandwidth_text->set_custom_minimum_size(Size2(120, 0) * EDSCALE);
	incoming_bandwidth_text->set_align(LineEdit::Align::ALIGN_RIGHT);
	hb->add_child(incoming_bandwidth_text);

	Control *down_up_spacer = memnew(Control);
	down_up_spacer->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
	hb->add_child(down_up_spacer);

	lb = memnew(Label);
	lb->set_text(TTR("Up"));
	hb->add_child(lb);

	outgoing_bandwidth_text = memnew(LineEdit);
	outgoing_bandwidth_text->set_editable(false);
	outgoing_bandwidth_text->set_custom_minimum_size(Size2(120, 0) * EDSCALE);
	outgoing_bandwidth_text->set_align(LineEdit::Align::ALIGN_RIGHT);
	hb->add_child(outgoing_bandwidth_text);

	// Set initial texts in the incoming/outgoing bandwidth labels
	set_bandwidth(0, 0);

	counters_display = memnew(Tree);
	counters_display->set_custom_minimum_size(Size2(300, 0) * EDSCALE);
	counters_display->set_v_size_flags(SIZE_EXPAND_FILL);
	counters_display->set_hide_folding(true);
	counters_display->set_hide_root(true);
	counters_display->set_columns(5);
	counters_display->set_column_titles_visible(true);
	counters_display->set_column_title(0, TTR("Node"));
	counters_display->set_column_expand(0, true);
	counters_display->set_column_min_width(0, 60 * EDSCALE);
	counters_display->set_column_title(1, TTR("Incoming RPC"));
	counters_display->set_column_expand(1, false);
	counters_display->set_column_min_width(1, 120 * EDSCALE);
	counters_display->set_column_title(2, TTR("Incoming RSET"));
	counters_display->set_column_expand(2, false);
	counters_display->set_column_min_width(2, 120 * EDSCALE);
	counters_display->set_column_title(3, TTR("Outgoing RPC"));
	counters_display->set_column_expand(3, false);
	counters_display->set_column_min_width(3, 120 * EDSCALE);
	counters_display->set_column_title(4, TTR("Outgoing RSET"));
	counters_display->set_column_expand(4, false);
	counters_display->set_column_min_width(4, 120 * EDSCALE);
	add_child(counters_display);

	frame_delay = memnew(Timer);
	frame_delay->set_wait_time(0.1);
	frame_delay->set_one_shot(true);
	add_child(frame_delay);
	frame_delay->connect("timeout", callable_mp(this, &EditorNetworkProfiler::_update_frame));
}
