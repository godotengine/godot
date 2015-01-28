/*************************************************************************/
/*  canvas_item.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "canvas_item.h"
#include "servers/visual_server.h"
#include "scene/main/viewport.h"
#include "scene/main/canvas_layer.h"
#include "message_queue.h"
#include "scene/scene_string_names.h"
#include "scene/resources/font.h"
#include "scene/resources/texture.h"
#include "scene/resources/style_box.h"

bool CanvasItem::is_visible() const {

	if (!is_inside_tree())
		return false;

	const CanvasItem *p=this;

	while(p) {		
		if (p->hidden)
			return false;
		p=p->get_parent_item();
	}


	return true;
}

bool CanvasItem::is_hidden() const {

	/*if (!is_inside_scene())
		return false;*/

	return hidden;
}

void CanvasItem::_propagate_visibility_changed(bool p_visible) {

	notification(NOTIFICATION_VISIBILITY_CHANGED);

	if (p_visible)
		update(); //todo optimize
	else
		emit_signal(SceneStringNames::get_singleton()->hide);
	_block();

	for(int i=0;i<get_child_count();i++) {

		CanvasItem *c=get_child(i)->cast_to<CanvasItem>();

		if (c && c->hidden!=p_visible) //should the toplevels stop propagation? i think so but..
			c->_propagate_visibility_changed(p_visible);
	}

	_unblock();

}

void CanvasItem::show() {

	if (!hidden)
		return;


	hidden=false;
	VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,true);

	if (!is_inside_tree())
		return;

	if (is_visible()) {
		_propagate_visibility_changed(true);
	}
}


void CanvasItem::hide() {

	if (hidden)
		return;

	bool propagate=is_inside_tree() && is_visible();
	hidden=true;
	VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,false);

	if (!is_inside_tree())
		return;
	if (propagate)
		_propagate_visibility_changed(false);

}


Variant CanvasItem::edit_get_state() const {


	return Variant();
}
void CanvasItem::edit_set_state(const Variant& p_state) {


}

void CanvasItem::edit_set_rect(const Rect2& p_edit_rect) {

	//used by editors, implement at will
}

void CanvasItem::edit_rotate(float p_rot) {


}

Size2 CanvasItem::edit_get_minimum_size() const {

	return Size2(-1,-1); //no limit
}

void CanvasItem::_update_callback() {



	if (!is_inside_tree()) {
		pending_update=false;
		return;
	}


	VisualServer::get_singleton()->canvas_item_clear(get_canvas_item());
	//todo updating = true - only allow drawing here
	if (is_visible()) { //todo optimize this!!
		if (first_draw) {
			notification(NOTIFICATION_VISIBILITY_CHANGED);
			first_draw=false;
		}
		drawing=true;
		notification(NOTIFICATION_DRAW);
		emit_signal(SceneStringNames::get_singleton()->draw);
		if (get_script_instance()) {
			Variant::CallError err;
			get_script_instance()->call_multilevel_reversed(SceneStringNames::get_singleton()->_draw,NULL,0);
		}
		drawing=false;

	}
	//todo updating = false
	pending_update=false; // don't change to false until finished drawing (avoid recursive update)
}

Matrix32 CanvasItem::get_global_transform_with_canvas() const {

	const CanvasItem *ci = this;
	Matrix32 xform;
	const CanvasItem *last_valid=NULL;

	while(ci) {

		last_valid=ci;
		xform = ci->get_transform() * xform;
		ci=ci->get_parent_item();
	}

	if (last_valid->canvas_layer)
		return last_valid->canvas_layer->get_transform() * xform;
	else
		return xform;
}

Matrix32 CanvasItem::get_global_transform() const {


	if (global_invalid) {

		const CanvasItem *pi = get_parent_item();
		if (pi)
			global_transform = pi->get_global_transform() * get_transform();
		else
			global_transform = get_transform();

		global_invalid=false;
	}

	return global_transform;

}


void CanvasItem::_queue_sort_children() {

	if (pending_children_sort)
		return;

	pending_children_sort=true;
	MessageQueue::get_singleton()->push_call(this,"_sort_children");
}

void CanvasItem::_sort_children() {

	pending_children_sort=false;

	if (!is_inside_tree())
		return;

	for(int i=0;i<get_child_count();i++) {

		Node *n = get_child(i);
		CanvasItem *ci=n->cast_to<CanvasItem>();

		if (ci) {
			if (ci->toplevel || ci->group!="")
				continue;
			VisualServer::get_singleton()->canvas_item_raise(n->cast_to<CanvasItem>()->canvas_item);
		}
	}
}

void CanvasItem::_raise_self() {

	if (!is_inside_tree())
		return;

	VisualServer::get_singleton()->canvas_item_raise(canvas_item);
}


void CanvasItem::_enter_canvas() {

	if ((!get_parent() || !get_parent()->cast_to<CanvasItem>()) || toplevel) {

		Node *n = this;
		Viewport *viewport=NULL;
		canvas_layer=NULL;

		while(n) {

			if (n->cast_to<Viewport>()) {

				viewport = n->cast_to<Viewport>();
				break;
			}
			if (!canvas_layer && n->cast_to<CanvasLayer>()) {

				canvas_layer = n->cast_to<CanvasLayer>();
			}
			n=n->get_parent();
		}

		RID canvas;
		if (canvas_layer)
			canvas=canvas_layer->get_world_2d()->get_canvas();
		else
			canvas=viewport->find_world_2d()->get_canvas();

		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,canvas);

		group = "root_canvas"+itos(canvas.get_id());

		add_to_group(group);
		get_tree()->call_group(SceneTree::GROUP_CALL_UNIQUE,group,"_raise_self");

	} else {

		CanvasItem *parent = get_parent_item();
		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,parent->get_canvas_item());
		parent->_queue_sort_children();
	}

	pending_update=false;
	update();

	notification(NOTIFICATION_ENTER_CANVAS);

}

void CanvasItem::_exit_canvas() {

	notification(NOTIFICATION_EXIT_CANVAS,true); //reverse the notification
	VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,RID());
	canvas_layer=NULL;
	group="";

}


void CanvasItem::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {

			first_draw=true;
			pending_children_sort=false;
			if (get_parent()) {
				CanvasItem *ci = get_parent()->cast_to<CanvasItem>();
				if (ci)
					C=ci->children_items.push_back(this);
			}
			_enter_canvas();
			if (!block_transform_notify && !xform_change.in_list()) {
				get_tree()->xform_change_list.add(&xform_change);
			}
		} break;
		case NOTIFICATION_MOVED_IN_PARENT: {


			if (group!="") {
				get_tree()->call_group(SceneTree::GROUP_CALL_UNIQUE,group,"_raise_self");
			} else {
				CanvasItem *p = get_parent_item();
				ERR_FAIL_COND(!p);
				p->_queue_sort_children();
			}


		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (xform_change.in_list())
				get_tree()->xform_change_list.remove(&xform_change);
			_exit_canvas();
			if (C) {
				get_parent()->cast_to<CanvasItem>()->children_items.erase(C);
				C=NULL;
			}
		} break;
		case NOTIFICATION_DRAW: {

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {


		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			emit_signal(SceneStringNames::get_singleton()->visibility_changed);
		} break;

	}
}

void CanvasItem::_set_visible_(bool p_visible) {

	if (p_visible)
		show();
	else
		hide();
}
bool CanvasItem::_is_visible_() const {

	return !is_hidden();
}


void CanvasItem::update() {

	if (!is_inside_tree())
		return;
	if (pending_update)
		return;

	pending_update=true;

	MessageQueue::get_singleton()->push_call(this,"_update_callback");
}

void CanvasItem::set_opacity(float p_opacity) {

	opacity=p_opacity;
	VisualServer::get_singleton()->canvas_item_set_opacity(canvas_item,opacity);

}
float CanvasItem::get_opacity() const {

	return opacity;
}


void CanvasItem::set_as_toplevel(bool p_toplevel) {

	if (toplevel==p_toplevel)
		return;

	if (!is_inside_tree()) {
		toplevel=p_toplevel;
		return;
	}

	_exit_canvas();
	toplevel=p_toplevel;
	_enter_canvas();
}

bool CanvasItem::is_set_as_toplevel() const {

	return toplevel;
}

CanvasItem *CanvasItem::get_parent_item() const {

	if (toplevel)
		return NULL;

	Node *parent = get_parent();
	if (!parent)
		return NULL;

	return parent->cast_to<CanvasItem>();
}


void CanvasItem::set_self_opacity(float p_self_opacity) {

	self_opacity=p_self_opacity;
	VisualServer::get_singleton()->canvas_item_set_self_opacity(canvas_item,self_opacity);

}
float CanvasItem::get_self_opacity() const {

	return self_opacity;
}

void CanvasItem::set_blend_mode(BlendMode p_blend_mode) {

	ERR_FAIL_INDEX(p_blend_mode,5);
	blend_mode=p_blend_mode;
	VisualServer::get_singleton()->canvas_item_set_blend_mode(canvas_item,VS::MaterialBlendMode(blend_mode));

}

CanvasItem::BlendMode CanvasItem::get_blend_mode() const {

	return blend_mode;
}



void CanvasItem::item_rect_changed() {

	update();
	emit_signal(SceneStringNames::get_singleton()->item_rect_changed);
}


void CanvasItem::draw_line(const Point2& p_from, const Point2& p_to,const Color& p_color,float p_width) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	VisualServer::get_singleton()->canvas_item_add_line(canvas_item,p_from,p_to,p_color,p_width);
}

void CanvasItem::draw_rect(const Rect2& p_rect, const Color& p_color) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	VisualServer::get_singleton()->canvas_item_add_rect(canvas_item,p_rect,p_color);

}

void CanvasItem::draw_circle(const Point2& p_pos, float p_radius, const Color& p_color) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	VisualServer::get_singleton()->canvas_item_add_circle(canvas_item,p_pos,p_radius,p_color);

}

void CanvasItem::draw_texture(const Ref<Texture>& p_texture,const Point2& p_pos) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_texture.is_null());

	p_texture->draw(canvas_item,p_pos);
}

void CanvasItem::draw_texture_rect(const Ref<Texture>& p_texture,const Rect2& p_rect, bool p_tile,const Color& p_modulate) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect(canvas_item,p_rect,p_tile,p_modulate);

}
void CanvasItem::draw_texture_rect_region(const Ref<Texture>& p_texture,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}
	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect_region(canvas_item,p_rect,p_src_rect,p_modulate);
}

void CanvasItem::draw_style_box(const Ref<StyleBox>& p_style_box,const Rect2& p_rect) {
	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_style_box.is_null());

	p_style_box->draw(canvas_item,p_rect);

}
void CanvasItem::draw_primitive(const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, Ref<Texture> p_texture,float p_width) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_primitive(canvas_item,p_points,p_colors,p_uvs,rid,p_width);
}
void CanvasItem::draw_set_transform(const Point2& p_offset, float p_rot, const Size2& p_scale) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	Matrix32 xform(p_rot,p_offset);
	xform.scale_basis(p_scale);
	VisualServer::get_singleton()->canvas_item_set_transform(canvas_item,xform);
}

void CanvasItem::draw_polygon(const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, Ref<Texture> p_texture) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_polygon(canvas_item,p_points,p_colors,p_uvs,rid);


}

void CanvasItem::draw_colored_polygon(const Vector<Point2>& p_points, const Color& p_color,const Vector<Point2>& p_uvs, Ref<Texture> p_texture) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	Vector<Color> colors;
	colors.push_back(p_color);
	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_polygon(canvas_item,p_points,colors,p_uvs,rid);
}

void CanvasItem::draw_string(const Ref<Font>& p_font,const Point2& p_pos, const String& p_text,const Color& p_modulate,int p_clip_w) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_font.is_null());
	p_font->draw(canvas_item,p_pos,p_text,p_modulate,p_clip_w);

}

float CanvasItem::draw_char(const Ref<Font>& p_font,const Point2& p_pos, const String& p_char,const String& p_next,const Color& p_modulate) {

	if (!drawing) {
		ERR_EXPLAIN("Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
		ERR_FAIL_V(0);
	}

	ERR_FAIL_COND_V(p_char.length()!=1,0);
	ERR_FAIL_COND_V(p_font.is_null(),0);

	return p_font->draw_char(canvas_item,p_pos,p_char[0],p_next.c_str()[0],p_modulate);

}


void CanvasItem::_notify_transform(CanvasItem *p_node) {

	if (p_node->xform_change.in_list() && p_node->global_invalid)
		return; //nothing to do

	p_node->global_invalid=true;

	if (!p_node->xform_change.in_list()) {
		if (!p_node->block_transform_notify) {
			if (p_node->is_inside_tree())
				get_tree()->xform_change_list.add(&p_node->xform_change);
		}
	}


	for(List<CanvasItem*>::Element *E=p_node->children_items.front();E;E=E->next()) {

		CanvasItem* ci=E->get();
		if (ci->toplevel)
			continue;
		_notify_transform(ci);
	}
}


Rect2 CanvasItem::get_viewport_rect() const {

	ERR_FAIL_COND_V(!is_inside_tree(),Rect2());
	return get_viewport()->get_visible_rect();
}

RID CanvasItem::get_canvas() const {

	ERR_FAIL_COND_V(!is_inside_tree(),RID());

	if (canvas_layer)
		return canvas_layer->get_world_2d()->get_canvas();
	else
		return get_viewport()->find_world_2d()->get_canvas();


}

CanvasItem *CanvasItem::get_toplevel() const {

	CanvasItem *ci=const_cast<CanvasItem*>(this);
	while(!ci->toplevel && ci->get_parent() && ci->get_parent()->cast_to<CanvasItem>()) {
		ci=ci->get_parent()->cast_to<CanvasItem>();
	}

	return ci;
}


Ref<World2D> CanvasItem::get_world_2d() const {

	ERR_FAIL_COND_V(!is_inside_tree(),Ref<World2D>());

	CanvasItem *tl=get_toplevel();

	if (tl->canvas_layer) {
		return tl->canvas_layer->get_world_2d();
	} else if (tl->get_viewport()) {
		return tl->get_viewport()->find_world_2d();
	} else {
		return Ref<World2D>();
	}

}

RID CanvasItem::get_viewport_rid() const {

	ERR_FAIL_COND_V(!is_inside_tree(),RID());
	return get_viewport()->get_viewport();
}

void CanvasItem::set_block_transform_notify(bool p_enable) {
	block_transform_notify=p_enable;
}

bool CanvasItem::is_block_transform_notify_enabled() const {

	return block_transform_notify;
}

void CanvasItem::set_draw_behind_parent(bool p_enable) {

	if (behind==p_enable)
		return;
	behind=p_enable;
	VisualServer::get_singleton()->canvas_item_set_on_top(canvas_item,!behind);

}

bool CanvasItem::is_draw_behind_parent_enabled() const{

	return behind;
}

void CanvasItem::set_shader(const Ref<Shader>& p_shader) {

	ERR_FAIL_COND(p_shader.is_valid() && p_shader->get_mode()!=Shader::MODE_CANVAS_ITEM);

#ifdef TOOLS_ENABLED

	if (shader.is_valid()) {
		shader->disconnect("changed",this,"_shader_changed");
	}
#endif
	shader=p_shader;

#ifdef TOOLS_ENABLED

	if (shader.is_valid()) {
		shader->connect("changed",this,"_shader_changed");
	}
#endif

	RID rid;
	if (shader.is_valid())
		rid=shader->get_rid();
	VS::get_singleton()->canvas_item_set_shader(canvas_item,rid);
	_change_notify(); //properties for shader exposed
}

void CanvasItem::set_use_parent_shader(bool p_use_parent_shader) {

	use_parent_shader=p_use_parent_shader;
	VS::get_singleton()->canvas_item_set_use_parent_shader(canvas_item,p_use_parent_shader);
}

bool CanvasItem::get_use_parent_shader() const{

	return use_parent_shader;
}

Ref<Shader> CanvasItem::get_shader() const{

	return shader;
}

void CanvasItem::set_shader_param(const StringName& p_param,const Variant& p_value) {

	VS::get_singleton()->canvas_item_set_shader_param(canvas_item,p_param,p_value);
}

Variant CanvasItem::get_shader_param(const StringName& p_param) const {

	return VS::get_singleton()->canvas_item_get_shader_param(canvas_item,p_param);
}

bool CanvasItem::_set(const StringName& p_name, const Variant& p_value) {

	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (pr) {
			set_shader_param(pr,p_value);
			return true;
		}
	}
	return false;
}

bool CanvasItem::_get(const StringName& p_name,Variant &r_ret) const{

	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (pr) {
			r_ret=get_shader_param(pr);
			return true;
		}
	}
	return false;

}
void CanvasItem::_get_property_list( List<PropertyInfo> *p_list) const{

	if (shader.is_valid()) {
		shader->get_param_list(p_list);
	}
}

#ifdef TOOLS_ENABLED
void CanvasItem::_shader_changed() {

	_change_notify();
}
#endif

void CanvasItem::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_sort_children"),&CanvasItem::_sort_children);
	ObjectTypeDB::bind_method(_MD("_raise_self"),&CanvasItem::_raise_self);
	ObjectTypeDB::bind_method(_MD("_update_callback"),&CanvasItem::_update_callback);
	ObjectTypeDB::bind_method(_MD("_set_visible_"),&CanvasItem::_set_visible_);
	ObjectTypeDB::bind_method(_MD("_is_visible_"),&CanvasItem::_is_visible_);

	ObjectTypeDB::bind_method(_MD("edit_set_state","state"),&CanvasItem::edit_set_state);
	ObjectTypeDB::bind_method(_MD("edit_get"),&CanvasItem::edit_get_state);
	ObjectTypeDB::bind_method(_MD("edit_set_rect","rect"),&CanvasItem::edit_set_rect);
	ObjectTypeDB::bind_method(_MD("edit_rotate","degrees"),&CanvasItem::edit_rotate);

	ObjectTypeDB::bind_method(_MD("get_item_rect"),&CanvasItem::get_item_rect);
	//ObjectTypeDB::bind_method(_MD("get_transform"),&CanvasItem::get_transform);
	ObjectTypeDB::bind_method(_MD("get_canvas_item"),&CanvasItem::get_canvas_item);

	ObjectTypeDB::bind_method(_MD("is_visible"),&CanvasItem::is_visible);
	ObjectTypeDB::bind_method(_MD("is_hidden"),&CanvasItem::is_hidden);
	ObjectTypeDB::bind_method(_MD("show"),&CanvasItem::show);
	ObjectTypeDB::bind_method(_MD("hide"),&CanvasItem::hide);

	ObjectTypeDB::bind_method(_MD("update"),&CanvasItem::update);

	ObjectTypeDB::bind_method(_MD("set_as_toplevel","enable"),&CanvasItem::set_as_toplevel);
	ObjectTypeDB::bind_method(_MD("is_set_as_toplevel"),&CanvasItem::is_set_as_toplevel);

	ObjectTypeDB::bind_method(_MD("set_blend_mode","blend_mode"),&CanvasItem::set_blend_mode);
	ObjectTypeDB::bind_method(_MD("get_blend_mode"),&CanvasItem::get_blend_mode);

	ObjectTypeDB::bind_method(_MD("set_opacity","opacity"),&CanvasItem::set_opacity);
	ObjectTypeDB::bind_method(_MD("get_opacity"),&CanvasItem::get_opacity);
	ObjectTypeDB::bind_method(_MD("set_self_opacity","self_opacity"),&CanvasItem::set_self_opacity);
	ObjectTypeDB::bind_method(_MD("get_self_opacity"),&CanvasItem::get_self_opacity);

	ObjectTypeDB::bind_method(_MD("set_draw_behind_parent","enabe"),&CanvasItem::set_draw_behind_parent);
	ObjectTypeDB::bind_method(_MD("is_draw_behind_parent_enabled"),&CanvasItem::is_draw_behind_parent_enabled);

	ObjectTypeDB::bind_method(_MD("_set_on_top","on_top"),&CanvasItem::_set_on_top);
	ObjectTypeDB::bind_method(_MD("_is_on_top"),&CanvasItem::_is_on_top);
#ifdef TOOLS_ENABLED
	ObjectTypeDB::bind_method(_MD("_shader_changed"),&CanvasItem::_shader_changed);
#endif
	//ObjectTypeDB::bind_method(_MD("get_transform"),&CanvasItem::get_transform);

	ObjectTypeDB::bind_method(_MD("draw_line","from","to","color","width"),&CanvasItem::draw_line,DEFVAL(1.0));
	ObjectTypeDB::bind_method(_MD("draw_rect","rect","color"),&CanvasItem::draw_rect);
	ObjectTypeDB::bind_method(_MD("draw_circle","pos","radius","color"),&CanvasItem::draw_circle);
	ObjectTypeDB::bind_method(_MD("draw_texture","texture:Texture","pos"),&CanvasItem::draw_texture);
	ObjectTypeDB::bind_method(_MD("draw_texture_rect","texture:Texture","rect","tile","modulate"),&CanvasItem::draw_texture_rect,DEFVAL(false),DEFVAL(Color(1,1,1)));
	ObjectTypeDB::bind_method(_MD("draw_texture_rect_region","texture:Texture","rect","src_rect","modulate"),&CanvasItem::draw_texture_rect_region,DEFVAL(Color(1,1,1)));
	ObjectTypeDB::bind_method(_MD("draw_style_box","style_box:StyleBox","rect"),&CanvasItem::draw_style_box);
	ObjectTypeDB::bind_method(_MD("draw_primitive","points","colors","uvs","texture:Texture","width"),&CanvasItem::draw_primitive,DEFVAL(Array()),DEFVAL(Ref<Texture>()),DEFVAL(1.0));
	ObjectTypeDB::bind_method(_MD("draw_polygon","points","colors","uvs","texture:Texture"),&CanvasItem::draw_polygon,DEFVAL(Array()),DEFVAL(Ref<Texture>()));
	ObjectTypeDB::bind_method(_MD("draw_colored_polygon","points","color","uvs","texture:Texture"),&CanvasItem::draw_colored_polygon,DEFVAL(Array()),DEFVAL(Ref<Texture>()));
	ObjectTypeDB::bind_method(_MD("draw_string","font:Font","pos","text","modulate","clip_w"),&CanvasItem::draw_string,DEFVAL(Color(1,1,1)),DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("draw_char","font:Font","pos","char","next","modulate"),&CanvasItem::draw_char,DEFVAL(Color(1,1,1)));

	ObjectTypeDB::bind_method(_MD("draw_set_transform","pos","rot","scale"),&CanvasItem::draw_set_transform);
	ObjectTypeDB::bind_method(_MD("get_transform"),&CanvasItem::get_transform);
	ObjectTypeDB::bind_method(_MD("get_global_transform"),&CanvasItem::get_global_transform);
	ObjectTypeDB::bind_method(_MD("get_viewport_transform"),&CanvasItem::get_viewport_transform);
	ObjectTypeDB::bind_method(_MD("get_viewport_rect"),&CanvasItem::get_viewport_rect);
	ObjectTypeDB::bind_method(_MD("get_canvas"),&CanvasItem::get_canvas);
	ObjectTypeDB::bind_method(_MD("get_world_2d"),&CanvasItem::get_world_2d);
	//ObjectTypeDB::bind_method(_MD("get_viewport"),&CanvasItem::get_viewport);

	ObjectTypeDB::bind_method(_MD("set_shader","shader"),&CanvasItem::set_shader);
	ObjectTypeDB::bind_method(_MD("get_shader"),&CanvasItem::get_shader);
	ObjectTypeDB::bind_method(_MD("set_use_parent_shader","enable"),&CanvasItem::set_use_parent_shader);
	ObjectTypeDB::bind_method(_MD("get_use_parent_shader"),&CanvasItem::get_use_parent_shader);

	BIND_VMETHOD(MethodInfo("_draw"));

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"visibility/visible"), _SCS("_set_visible_"),_SCS("_is_visible_") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"visibility/opacity",PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_opacity"),_SCS("get_opacity") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"visibility/self_opacity",PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_self_opacity"),_SCS("get_self_opacity") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::BOOL,"visibility/behind_parent"), _SCS("set_draw_behind_parent"),_SCS("is_draw_behind_parent_enabled") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"visibility/on_top",PROPERTY_HINT_NONE,"",0), _SCS("_set_on_top"),_SCS("_is_on_top") ); //compatibility

	ADD_PROPERTYNZ( PropertyInfo(Variant::INT,"visibility/blend_mode",PROPERTY_HINT_ENUM, "Mix,Add,Sub,Mul,PMAlpha"), _SCS("set_blend_mode"),_SCS("get_blend_mode") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::OBJECT,"shader/shader",PROPERTY_HINT_RESOURCE_TYPE, "CanvasItemShader,CanvasItemShaderGraph"), _SCS("set_shader"),_SCS("get_shader") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::BOOL,"shader/use_parent"), _SCS("set_use_parent_shader"),_SCS("get_use_parent_shader") );
	//exporting these two things doesn't really make much sense i think
	//ADD_PROPERTY( PropertyInfo(Variant::BOOL,"transform/toplevel"), _SCS("set_as_toplevel"),_SCS("is_set_as_toplevel") );
	//ADD_PROPERTY(PropertyInfo(Variant::BOOL,"transform/notify"),_SCS("set_transform_notify"),_SCS("is_transform_notify_enabled"));

	ADD_SIGNAL( MethodInfo("draw") );
	ADD_SIGNAL( MethodInfo("visibility_changed") );
	ADD_SIGNAL( MethodInfo("hide") );
	ADD_SIGNAL( MethodInfo("item_rect_changed") );



	BIND_CONSTANT( BLEND_MODE_MIX );
	BIND_CONSTANT( BLEND_MODE_ADD );
	BIND_CONSTANT( BLEND_MODE_SUB );
	BIND_CONSTANT( BLEND_MODE_MUL );
	BIND_CONSTANT( BLEND_MODE_PREMULT_ALPHA );


	BIND_CONSTANT( NOTIFICATION_DRAW);
	BIND_CONSTANT( NOTIFICATION_VISIBILITY_CHANGED );
	BIND_CONSTANT( NOTIFICATION_ENTER_CANVAS );
	BIND_CONSTANT( NOTIFICATION_EXIT_CANVAS );
	BIND_CONSTANT( NOTIFICATION_TRANSFORM_CHANGED );


}

Matrix32 CanvasItem::get_canvas_transform() const {

	ERR_FAIL_COND_V(!is_inside_tree(),Matrix32());

	if (canvas_layer)
		return canvas_layer->get_transform();
	else
		return get_viewport()->get_canvas_transform();

}

Matrix32 CanvasItem::get_viewport_transform() const {

	ERR_FAIL_COND_V(!is_inside_tree(),Matrix32());

	if (canvas_layer) {

		if (get_viewport()) {
			return get_viewport()->get_final_transform() * canvas_layer->get_transform();
		} else {
			return canvas_layer->get_transform();
		}

	} else if (get_viewport()) {
		return get_viewport()->get_final_transform() * get_viewport()->get_canvas_transform();
	}

	return Matrix32();

}


CanvasItem::CanvasItem() : xform_change(this) {


	canvas_item=VisualServer::get_singleton()->canvas_item_create();
	hidden=false;
	pending_update=false;
	opacity=1;
	self_opacity=1;
	toplevel=false;	
	pending_children_sort=false;
	first_draw=false;
	blend_mode=BLEND_MODE_MIX;
	drawing=false;
	behind=false;
	block_transform_notify=false;
//	viewport=NULL;
	canvas_layer=NULL;
	use_parent_shader=false;
	global_invalid=true;

	C=NULL;

}

CanvasItem::~CanvasItem() {

	VisualServer::get_singleton()->free(canvas_item);
}
