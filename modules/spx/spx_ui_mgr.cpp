/**************************************************************************/
/*  spx_ui_mgr.cpp                                                        */
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

#include "spx_ui_mgr.h"

#include "scene/main/canvas_layer.h"
#include "scene/resources/packed_scene.h"

#define SPX_CALLBACK SpxEngine::get_singleton()->get_callbacks()
#define check_and_get_node_r(VALUE) \
	auto node = get_node(obj);\
	if (node == nullptr) {\
		print_error("try to get property of a null node gid=" + itos(obj)); \
	return VALUE; \
}

#define check_and_get_node_v() \
	auto node = get_node(obj);\
	if (node == nullptr) {\
		print_error("try to get property of a null node gid=" + itos(obj)); \
	return ; \
}


SpxUi *SpxUiMgr::get_node(GdObj obj) {
	if (id_objects.has(obj)) {
		return id_objects[obj];
	}
	return nullptr;
}

ESpxUiType SpxUiMgr::get_node_type(Node *obj) {
	if (obj == nullptr) {
		return ESpxUiType::None;
	}
	if (dynamic_cast<SpxLabel *>(obj)) {
		return ESpxUiType::Label;
	}
	if (dynamic_cast<SpxButton *>(obj)) {
		return ESpxUiType::Button;
	}
	if (dynamic_cast<SpxImage *>(obj)) {
		return ESpxUiType::Image;
	}
	if (dynamic_cast<SpxToggle *>(obj)) {
		return ESpxUiType::Toggle;
	}
	if (dynamic_cast<SpxInput *>(obj)) {
		return ESpxUiType::Input;
	}


	if (dynamic_cast<SpxControl *>(obj)) {
		return ESpxUiType::Control;
	}
	return ESpxUiType::None;
}

void SpxUiMgr::on_click(ISpxUi *node) {
	SPX_CALLBACK->func_on_ui_clicked(node->get_gid());
}

void SpxUiMgr::on_awake() {
	SpxBaseMgr::on_awake();
	owner = memnew(CanvasLayer);
	owner->set_name(get_class_name());
	get_spx_root()->add_child(owner);
}

void SpxUiMgr::on_node_destroy(SpxUi *node) {
	if (id_objects.erase(node->get_gid())) {
		SPX_CALLBACK->func_on_ui_destroyed(node->get_gid());
	}
}

Control *SpxUiMgr::create_control(GdString path) {
	const String path_str = SpxStr(path);
	Control *node = nullptr;
	if (path_str != "") {
		Ref<PackedScene> scene = ResourceLoader::load(path_str);
		if (scene.is_null()) {
			print_error("Failed to load sprite scene " + path_str);
			return NULL_OBJECT_ID;
		} else {
			node = dynamic_cast<Control *>(scene->instantiate());
			if (node == nullptr) {
				print_error("Failed to load sprite scene , type invalid " + path_str);
			}
		}
	}
	return node;
}



SpxUi *SpxUiMgr::on_create_node(Control *control, GdInt type,bool is_attach) {
	SpxUi *node = memnew(SpxUi);
	if(is_attach) {
		owner->add_child(control);
	}
	node->set_type(type);
	node->set_control_item(control);
	node->set_gid(get_unique_id());
	node->on_start();
	uiMgr->id_objects[node->get_gid()] = node;
	SPX_CALLBACK->func_on_ui_ready(node->get_gid());
	return node;
}

#define CREATE_UI_NODE(TYPE) \
	Spx##TYPE *control = dynamic_cast<Spx##TYPE *>(create_control(path)); \
	if (control == nullptr) { \
		control = memnew(Spx##TYPE); \
	} \
	auto node = on_create_node(control,(GdInt)ESpxUiType::TYPE);

GdObj SpxUiMgr::create_node(GdString path) {
	auto control = create_control(path);
	if (control == nullptr) {
		print_error("Failed to create node " + SpxStr(path));
		return NULL_OBJECT_ID;
	}
	auto type = get_node_type(control);
	auto node = on_create_node(control, (GdInt)type);
	return node->get_gid();
}

GdObj SpxUiMgr::create_button(GdString path, GdString text) {
	CREATE_UI_NODE(Button)
	control->set_text(SpxStr(text));
	return node->get_gid();
}

GdObj SpxUiMgr::create_label(GdString path, GdString text) {
	CREATE_UI_NODE(Label)
	control->set_text(SpxStr(text));
	return node->get_gid();
}

GdObj SpxUiMgr::create_image(GdString path) {
	CREATE_UI_NODE(Image)
	return node->get_gid();
}

GdObj SpxUiMgr::create_toggle(GdString path, GdBool value) {
	CREATE_UI_NODE(Toggle)
	return node->get_gid();
}

GdObj SpxUiMgr::create_slider(GdString path, GdFloat value) {
	return 0;
}


GdObj SpxUiMgr::create_input(GdString path, GdString text) {
	return 0;
}

GdBool SpxUiMgr::destroy_node(GdObj obj) {
	check_and_get_node_r(false)
	node->queue_free();
	return true;
}
GdObj SpxUiMgr::bind_node(GdObj obj, GdString rel_path) {
	auto parent_node  = get_node(obj);
	if(parent_node == nullptr) {
		print_line("bind_node error :can not find a node ", obj);
		return NULL_OBJECT_ID;
	}
	auto path = SpxStr(rel_path);
	if(! parent_node->control->has_node(path)) {
		print_line("bind_node error :can not find a child node ", obj, path);
		return NULL_OBJECT_ID;
	}
	auto child = parent_node->control->get_node(path);
	auto type = get_node_type(child);
	if(type == ESpxUiType::None) {
		print_line("bind_node error : unknown type ", obj, path);
		return NULL_OBJECT_ID;
	}
	auto node = on_create_node( (Control*) child, (GdInt)type,false);
	return node->get_gid();
}

GdInt SpxUiMgr::get_type(GdObj obj) {
	check_and_get_node_r(0)
	return node->get_type();
}

void SpxUiMgr::set_interactable(GdObj obj, GdBool interactable) {
	check_and_get_node_v()
	node->set_interactable(interactable);
}

GdBool SpxUiMgr::get_interactable(GdObj obj) {
	check_and_get_node_r(false)
	return node->is_interactable();
}

void SpxUiMgr::set_text(GdObj obj, GdString text) {
	check_and_get_node_v()
	node->set_text(text);
}

GdString SpxUiMgr::get_text(GdObj obj) {
	check_and_get_node_r(GdString())
	return node->get_text();
}

void SpxUiMgr::set_rect(GdObj obj, GdRect2 rect) {
	check_and_get_node_v()
	node->set_rect(rect);
}

GdRect2 SpxUiMgr::get_rect(GdObj obj) {
	check_and_get_node_r(GdRect2())
	return node->get_rect();
}

void SpxUiMgr::set_color(GdObj obj, GdColor color) {
	check_and_get_node_v()
	node->set_color(color);
}

GdColor SpxUiMgr::get_color(GdObj obj) {
	check_and_get_node_r(GdColor())
	return node->get_color();
}

void SpxUiMgr::set_font_size(GdObj obj, GdInt size) {
	check_and_get_node_v()
	node->set_font_size(size);
}

GdInt SpxUiMgr::get_font_size(GdObj obj) {
	check_and_get_node_r(0)
	return node->get_font_size();
}

void SpxUiMgr::set_visible(GdObj obj, GdBool visible) {
	check_and_get_node_v()
	node->set_visible(visible);
}

GdBool SpxUiMgr::get_visible(GdObj obj) {
	check_and_get_node_r(false)
	return node->get_visible();
}

void SpxUiMgr::set_texture(GdObj obj, GdString path) {
	check_and_get_node_v()
	node->set_texture(path);
}

GdString SpxUiMgr::get_texture(GdObj obj) {
	check_and_get_node_r(nullptr)
	return node->get_texture();
}


GdInt SpxUiMgr::get_layout_direction(GdObj obj) {
	check_and_get_node_r(0)
	return node->get_layout_direction();
}
void SpxUiMgr::set_layout_direction(GdObj obj,GdInt value) {
	check_and_get_node_v()
	node->set_layout_direction(value);
}
GdInt SpxUiMgr::get_layout_mode(GdObj obj) {
	check_and_get_node_r(0)
	return node->get_layout_mode();
}
void SpxUiMgr::set_layout_mode(GdObj obj,GdInt value) {
	check_and_get_node_v()
	node->set_layout_mode(value);
}
GdInt SpxUiMgr::get_anchors_preset(GdObj obj) {
	check_and_get_node_r(0)
	return node->get_anchors_preset();
}
void SpxUiMgr::set_anchors_preset(GdObj obj, GdInt value) {
	check_and_get_node_v()
	node->set_anchors_preset(value);
}

GdVec2 SpxUiMgr::get_scale(GdObj obj) {
	check_and_get_node_r(GdVec2())
	return node->get_scale();

}
void SpxUiMgr::set_scale(GdObj obj, GdVec2 value) {
	check_and_get_node_v()
	node->set_scale(value);
}
GdVec2 SpxUiMgr::get_position(GdObj obj) {
	check_and_get_node_r(GdVec2())
	return node->get_position();

}
void SpxUiMgr::set_position(GdObj obj, GdVec2 value) {
	check_and_get_node_v()
	node->set_position(value);
}
GdVec2 SpxUiMgr::get_size(GdObj obj) {
	check_and_get_node_r(GdVec2())
	return node->get_size();

}
void SpxUiMgr::set_size(GdObj obj, GdVec2 value) {
	check_and_get_node_v()
	node->set_size(value);
}

GdVec2 SpxUiMgr::get_global_position(GdObj obj) {
	check_and_get_node_r(GdVec2())
	return node->get_global_position();
}
void SpxUiMgr::set_global_position(GdObj obj, GdVec2 value) {
	check_and_get_node_v()
	node->set_global_position(value);
}
GdFloat SpxUiMgr::get_rotation(GdObj obj) {
	check_and_get_node_r(GdFloat())
	return node->get_rotation();

}
void SpxUiMgr::set_rotation(GdObj obj, GdFloat value) {
	check_and_get_node_v()
	node->set_rotation(value);
}
GdBool SpxUiMgr::get_flip(GdObj obj, GdBool horizontal) {
	check_and_get_node_r(GdBool())
	return node->get_flip(horizontal);
}
void SpxUiMgr::set_flip(GdObj obj, GdBool horizontal, GdBool is_flip) {
	check_and_get_node_v()
	node->set_flip(horizontal,is_flip);
}
