/*************************************************************************/
/*  editor_playback_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_playback_plugin.h"

#include "camera_matrix.h"
#include "core/os/input.h"
#include "editor/animation_editor.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/script_editor_debugger.h"
#include "os/keyboard.h"
#include "print_string.h"
#include "project_settings.h"
#include "scene/3d/camera.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/packed_scene.h"
#include "sort.h"

static int _get_key_modifier_setting(const String &p_property) {

	switch (EditorSettings::get_singleton()->get(p_property).operator int()) {

		case 0: return 0;
		case 1: return KEY_SHIFT;
		case 2: return KEY_ALT;
		case 3: return KEY_META;
		case 4: return KEY_CONTROL;
	}
	return 0;
}

static int _get_key_modifier(Ref<InputEventWithModifiers> e) {
	if (e->get_shift())
		return KEY_SHIFT;
	if (e->get_alt())
		return KEY_ALT;
	if (e->get_control())
		return KEY_CONTROL;
	if (e->get_metakey())
		return KEY_META;
	return 0;
}

void EditorPlaybackViewport::_smouseenter() {

	if (!surface->has_focus() && (!get_focus_owner() || !get_focus_owner()->is_text_field()))
		surface->grab_focus();
}

void EditorPlaybackViewport::_smouseexit() {
}

void EditorPlaybackViewport::_sinput(const Ref<InputEvent> &p_event) {

	{
		EditorNode *en = editor;
		EditorPluginList *force_input_forwarding_list = en->get_editor_plugins_force_input_forwarding();
		if (!force_input_forwarding_list->empty()) {
		}
	}
	{
		EditorNode *en = editor;
		EditorPluginList *over_plugin_list = en->get_editor_plugins_over();
		if (!over_plugin_list->empty()) {
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
	}
}

void EditorPlaybackViewport::set_message(String p_message, float p_time) {

	message = p_message;
	message_time = p_time;
}

void EditorPlaybackViewport::_notification(int p_what) {

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		bool visible = is_visible_in_tree();

		set_process(visible);
	}

	if (p_what == NOTIFICATION_RESIZED) {
	}

	if (p_what == NOTIFICATION_PROCESS) {

		real_t delta = get_process_delta_time();

		bool changed = false;
		bool exist = false;

		if (message_time > 0) {

			if (message != last_message) {
				surface->update();
				last_message = message;
			}

			message_time -= get_physics_process_delta_time();
			if (message_time < 0)
				surface->update();
		}

		//update shadow atlas if changed

		int shadowmap_size = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/size");
		int atlas_q0 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_0_subdiv");
		int atlas_q1 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_1_subdiv");
		int atlas_q2 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_2_subdiv");
		int atlas_q3 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_3_subdiv");

		viewport->set_shadow_atlas_size(shadowmap_size);
		viewport->set_shadow_atlas_quadrant_subdiv(0, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q0));
		viewport->set_shadow_atlas_quadrant_subdiv(1, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q1));
		viewport->set_shadow_atlas_quadrant_subdiv(2, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q2));
		viewport->set_shadow_atlas_quadrant_subdiv(3, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q3));

		//update msaa if changed

		int msaa_mode = ProjectSettings::get_singleton()->get("rendering/quality/filters/msaa");
		viewport->set_msaa(Viewport::MSAA(msaa_mode));

		bool hdr = ProjectSettings::get_singleton()->get("rendering/quality/depth/hdr");
		viewport->set_hdr(hdr);

		editor->get_scene_root()->set_size_override(true, get_size());

		prev_size = surface->get_size();

		_update_camera_state();
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		surface->connect("draw", this, "_draw");
		surface->connect("gui_input", this, "_sinput");
		surface->connect("mouse_entered", this, "_smouseenter");
		surface->connect("mouse_exited", this, "_smouseexit");
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {
	}

	if (p_what == NOTIFICATION_MOUSE_ENTER) {
	}

	if (p_what == NOTIFICATION_DRAW) {
	}
}

void EditorPlaybackViewport::_current_camera_changed(Object *p_camera) {
	camera_pointer = Object::cast_to<Camera>(p_camera);
}

void EditorPlaybackViewport::_update_camera_state() {
	if (camera_pointer) {
		camera->show();

		camera->set_global_transform(camera_pointer->get_global_transform());
		if (camera_pointer->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
			camera->set_perspective(camera_pointer->get_fov(), camera_pointer->get_znear(), camera_pointer->get_zfar());
		} else {
			camera->set_orthogonal(camera_pointer->get_size(), camera_pointer->get_znear(), camera_pointer->get_zfar());
		}
		camera->set_environment(camera_pointer->get_environment());
		camera->set_cull_mask(camera_pointer->get_cull_mask());

		;
	} else {
		camera->hide();
	}
}

void EditorPlaybackViewport::_draw() {

	if (surface->has_focus()) {
		Size2 size = surface->get_size();
		Rect2 r = Rect2(Point2(), size);
		get_stylebox("Focus", "EditorStyles")->draw(surface->get_canvas_item(), r);
	}

	RID ci = surface->get_canvas_item();

	if (message_time > 0) {
		Ref<Font> font = get_font("font", "Label");
		Point2 msgpos = Point2(5, get_size().y - 20);
		font->draw(ci, msgpos + Point2(1, 1), message, Color(0, 0, 0, 0.8));
		font->draw(ci, msgpos + Point2(-1, -1), message, Color(0, 0, 0, 0.8));
		font->draw(ci, msgpos, message, Color(1, 1, 1, 1));
	}

	/*if (previewing) {

		Size2 ss = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
		float aspect = ss.aspect();
		Size2 s = get_size();

		Rect2 draw_rect;

		switch (previewing->get_keep_aspect_mode()) {
			case Camera::KEEP_WIDTH: {

				draw_rect.size = Size2(s.width, s.width / aspect);
				draw_rect.position.x = 0;
				draw_rect.position.y = (s.height - draw_rect.size.y) * 0.5;

			} break;
			case Camera::KEEP_HEIGHT: {

				draw_rect.size = Size2(s.height * aspect, s.height);
				draw_rect.position.y = 0;
				draw_rect.position.x = (s.width - draw_rect.size.x) * 0.5;

			} break;
		}

		draw_rect = Rect2(Vector2(), s).clip(draw_rect);

		stroke_rect(*surface, draw_rect, Color(0.6, 0.6, 0.1, 0.5), 2.0);

	} else {
	}*/
}

void EditorPlaybackViewport::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_draw"), &EditorPlaybackViewport::_draw);
	ClassDB::bind_method(D_METHOD("_smouseenter"), &EditorPlaybackViewport::_smouseenter);
	ClassDB::bind_method(D_METHOD("_smouseexit"), &EditorPlaybackViewport::_smouseexit);
	ClassDB::bind_method(D_METHOD("_sinput"), &EditorPlaybackViewport::_sinput);
	ClassDB::bind_method(D_METHOD("_current_camera_changed"), &EditorPlaybackViewport::_current_camera_changed);

	ADD_SIGNAL(MethodInfo("toggle_maximize_view", PropertyInfo(Variant::OBJECT, "viewport")));
}

void EditorPlaybackViewport::reset() {

	message_time = 0;
	message = "";
	last_message = "";
	name = "";
}

void EditorPlaybackViewport::focus_selection() {
	Vector3 center;
	int count = 0;

	if (count != 0) {
		center /= float(count);
	}
}

EditorPlaybackViewport::EditorPlaybackViewport(EditorPlayback *p_editor_playback, EditorNode *p_editor) {

	editor = p_editor;
	editor_data = editor->get_scene_tree_dock()->get_editor_data();
	message_time = 0;

	editor_playback = p_editor_playback;

	ViewportContainer *c = memnew(ViewportContainer);
	viewport_container = c;
	c->set_stretch(true);
	add_child(c);
	c->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	c->show();
	viewport = memnew(Viewport);
	viewport->set_disable_input(false);
	viewport->set_world_2d(editor->get_scene_root()->get_world_2d());
	c->add_child(viewport);

	camera = memnew(Camera);
	camera->set_disable_gizmo(true);
	viewport->add_child(camera);
	camera->make_current();
	camera->hide();

	surface = memnew(Control);
	surface->set_drag_forwarding(this);
	add_child(surface);
	surface->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	surface->set_clip_contents(true);
	surface->set_focus_mode(FOCUS_ALL);

	camera_pointer = NULL;

	// Setup camera
	Viewport *vp = editor->get_scene_root();
	if (vp) {
		_current_camera_changed(vp->get_camera());
		vp->connect("current_camera_changed", this, "_current_camera_changed");
	}

	name = "";
}

EditorPlaybackViewport::~EditorPlaybackViewport() {
	Viewport *vp = editor->get_scene_root();
	if (vp) {
		vp->disconnect("current_camera_changed", this, "_current_camera_changed");
	}
}

//////////////////////////////////////////////////////////////

void EditorPlaybackViewportContainer::_gui_input(const Ref<InputEvent> &p_event) {
}

void EditorPlaybackViewportContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_MOUSE_ENTER || p_what == NOTIFICATION_MOUSE_EXIT) {

		mouseover = (p_what == NOTIFICATION_MOUSE_ENTER);
		update();
	}

	if (p_what == NOTIFICATION_DRAW && mouseover) {
	}

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		EditorPlaybackViewport *viewport;
		viewport = Object::cast_to<EditorPlaybackViewport>(get_child(0));

		Size2 size = get_size();

		if (size.x < 10 || size.y < 10) {
			viewport->hide();
			return;
		} else {
			viewport->show();
		}

		fit_child_in_rect(viewport, Rect2(Vector2(), size));
	}
}

void EditorPlaybackViewportContainer::_bind_methods() {

	ClassDB::bind_method("_gui_input", &EditorPlaybackViewportContainer::_gui_input);
}

EditorPlaybackViewportContainer::EditorPlaybackViewportContainer() {

	mouseover = false;
	ratio_h = 0.5;
	ratio_v = 0.5;
}

///////////////////////////////////////////////////////////////////

EditorPlayback *EditorPlayback::singleton = NULL;

Dictionary EditorPlayback::get_state() const {

	Dictionary d;

	int vc = 1;

	d["viewport_mode"] = vc;
	Array vpdata;

	d["viewports"] = vpdata;

	return d;
}
void EditorPlayback::set_state(const Dictionary &p_state) {

	Dictionary d = p_state;

	if (d.has("viewports")) {
		Array vp = d["viewports"];
		ERR_FAIL_COND(vp.size() > VIEWPORTS_COUNT);
	}
}

void EditorPlayback::_unhandled_key_input(Ref<InputEvent> p_event) {

	if (!is_visible_in_tree() || get_viewport()->gui_has_modal_stack())
		return;
}
void EditorPlayback::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {

		get_tree()->connect("node_removed", this, "_node_removed");
		EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor()->connect("node_changed", this, "_refresh_menu_icons");
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
	}
}

void EditorPlayback::_toggle_maximize_view(Object *p_viewport) {
	if (!p_viewport)
		return;

	EditorPlaybackViewport *current_viewport = Object::cast_to<EditorPlaybackViewport>(p_viewport);
	if (!current_viewport)
		return;

	bool maximized = false;

	if (current_viewport->get_global_rect() == viewport_base->get_global_rect())
		maximized = true;

	if (!maximized) {
		viewport->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	} else {
		viewport->show();
	}
}

void EditorPlayback::_node_removed(Node *p_node) {
}

void EditorPlayback::_bind_methods() {

	//ClassDB::bind_method("_gui_input",&EditorPlayback::_gui_input);
	ClassDB::bind_method("_unhandled_key_input", &EditorPlayback::_unhandled_key_input);
	ClassDB::bind_method("_node_removed", &EditorPlayback::_node_removed);
	ClassDB::bind_method("_toggle_maximize_view", &EditorPlayback::_toggle_maximize_view);
}

void EditorPlayback::clear() {

	viewport->reset();
}

EditorPlayback::EditorPlayback(EditorNode *p_editor) {

	singleton = this;
	editor = p_editor;
	editor_selection = editor->get_editor_selection();
	editor_selection->add_editor_plugin(this);

	VBoxContainer *vbc = this;

	hbc_menu = memnew(HBoxContainer);
	vbc->add_child(hbc_menu);

	viewport_base = memnew(EditorPlaybackViewportContainer);
	viewport_base->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(viewport_base);

	viewport = memnew(EditorPlaybackViewport(this, editor));
	viewport->connect("toggle_maximize_view", this, "_toggle_maximize_view");
	viewport_base->add_child(viewport);

	set_process_unhandled_key_input(true);
	add_to_group("_editor_playback_group");
}

EditorPlayback::~EditorPlayback() {
}

void EditorPlaybackPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		editor_playback->show();
		VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport_rid(), false);
		editor->set_use_fixed_window_size_override(false);
		editor_playback->set_process(true);
		editor_playback->grab_focus();

	} else {

		editor_playback->hide();
		VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport_rid(), true);
		editor->set_use_fixed_window_size_override(true);
		editor_playback->set_process(false);
	}
}
void EditorPlaybackPlugin::edit(Object *p_object) {
}

bool EditorPlaybackPlugin::handles(Object *p_object) const {

	return false;
}

Dictionary EditorPlaybackPlugin::get_state() const {
	return editor_playback->get_state();
}

void EditorPlaybackPlugin::set_state(const Dictionary &p_state) {

	editor_playback->set_state(p_state);
}

void EditorPlaybackPlugin::_bind_methods() {
}

EditorPlaybackPlugin::EditorPlaybackPlugin(EditorNode *p_node) {

	editor = p_node;
	editor_playback = memnew(EditorPlayback(p_node));
	editor_playback->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(editor_playback);

	editor_playback->hide();
}

EditorPlaybackPlugin::~EditorPlaybackPlugin() {
}
