/*************************************************************************/
/*  sprite_region_editor_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Author: Mariano Suligoy                                               */
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

#include "sprite_region_editor_plugin.h"
#include "scene/gui/check_box.h"
#include "os/input.h"
#include "os/keyboard.h"

void SpriteRegionEditor::_region_draw()
{
	Ref<Texture> base_tex = node->get_texture();
	if (base_tex.is_null())
		return;

	Matrix32 mtx;
	mtx.elements[2]=-draw_ofs;
	mtx.scale_basis(Vector2(draw_zoom,draw_zoom));

	VS::get_singleton()->canvas_item_set_clip(edit_draw->get_canvas_item(),true);
	VS::get_singleton()->canvas_item_add_set_transform(edit_draw->get_canvas_item(),mtx);
	edit_draw->draw_texture(base_tex,Point2());
	VS::get_singleton()->canvas_item_add_set_transform(edit_draw->get_canvas_item(),Matrix32());

	if (snap_show_grid) {
		Size2 s = edit_draw->get_size();
		int last_cell;

		if (snap_step.x!=0) {
			for(int i=0;i<s.width;i++) {
				int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(i,0)).x-snap_offset.x)/snap_step.x));
				if (i==0)
					last_cell=cell;
				if (last_cell!=cell)
					edit_draw->draw_line(Point2(i,0),Point2(i,s.height),Color(0.3,0.7,1,0.3));
				last_cell=cell;
			}
		}

		if (snap_step.y!=0) {
			for(int i=0;i<s.height;i++) {
				int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(0,i)).y-snap_offset.y)/snap_step.y));
				if (i==0)
					last_cell=cell;
				if (last_cell!=cell)
					edit_draw->draw_line(Point2(0,i),Point2(s.width,i),Color(0.3,0.7,1,0.3));
				last_cell=cell;
			}
		}
	}

	Ref<Texture> select_handle = get_icon("EditorHandle","EditorIcons");

	Rect2 scroll_rect(Point2(),mtx.basis_xform(base_tex->get_size()));
	scroll_rect.expand_to(mtx.basis_xform(edit_draw->get_size()));

	Vector2 endpoints[4]={
		mtx.basis_xform(rect.pos),
		mtx.basis_xform(rect.pos+Vector2(rect.size.x,0)),
		mtx.basis_xform(rect.pos+rect.size),
		mtx.basis_xform(rect.pos+Vector2(0,rect.size.y))
	};

	for(int i=0;i<4;i++) {

		int prev = (i+3)%4;
		int next = (i+1)%4;

		Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
		ofs*=1.4144*(select_handle->get_size().width/2);

		edit_draw->draw_line(endpoints[i]-draw_ofs, endpoints[next]-draw_ofs, Color(0.9,0.5,0.5), 2);

		edit_draw->draw_texture(select_handle,(endpoints[i]+ofs-(select_handle->get_size()/2)).floor()-draw_ofs);

		ofs = (endpoints[next]-endpoints[i])/2;
		ofs += (endpoints[next]-endpoints[i]).tangent().normalized()*(select_handle->get_size().width/2);

		edit_draw->draw_texture(select_handle,(endpoints[i]+ofs-(select_handle->get_size()/2)).floor()-draw_ofs);

		scroll_rect.expand_to(endpoints[i]);
	}

	scroll_rect=scroll_rect.grow(200);
	updating_scroll=true;
	hscroll->set_min(scroll_rect.pos.x);
	hscroll->set_max(scroll_rect.pos.x+scroll_rect.size.x);
	hscroll->set_page(edit_draw->get_size().x);
	hscroll->set_val(draw_ofs.x);
	hscroll->set_step(0.001);

	vscroll->set_min(scroll_rect.pos.y);
	vscroll->set_max(scroll_rect.pos.y+scroll_rect.size.y);
	vscroll->set_page(edit_draw->get_size().y);
	vscroll->set_val(draw_ofs.y);
	vscroll->set_step(0.001);
	updating_scroll=false;
}

void SpriteRegionEditor::_region_input(const InputEvent& p_input)
{
	Matrix32 mtx;
	mtx.elements[2]=-draw_ofs;
	mtx.scale_basis(Vector2(draw_zoom,draw_zoom));

	Vector2 endpoints[8]={
		mtx.xform(rect.pos)+Vector2(-4,-4),
		mtx.xform(rect.pos+Vector2(rect.size.x/2,0))+Vector2(0,-4),
		mtx.xform(rect.pos+Vector2(rect.size.x,0))+Vector2(4,-4),
		mtx.xform(rect.pos+Vector2(rect.size.x,rect.size.y/2))+Vector2(4,0),
		mtx.xform(rect.pos+rect.size)+Vector2(4,4),
		mtx.xform(rect.pos+Vector2(rect.size.x/2,rect.size.y))+Vector2(0,4),
		mtx.xform(rect.pos+Vector2(0,rect.size.y))+Vector2(-4,4),
		mtx.xform(rect.pos+Vector2(0,rect.size.y/2))+Vector2(-4,0)
	};

	if (p_input.type==InputEvent::MOUSE_BUTTON) {


		const InputEventMouseButton &mb=p_input.mouse_button;

		if (mb.button_index==BUTTON_LEFT) {


			if (mb.pressed) {

				drag_from=mtx.affine_inverse().xform(Vector2(mb.x,mb.y));
				drag_from=snap_point(drag_from);
				drag=true;
				rect_prev=node->get_region_rect();

				drag_index=-1;
				for(int i=0;i<8;i++) {

					Vector2 tuv=endpoints[i];
					if (tuv.distance_to(Vector2(mb.x,mb.y))<8) {
						drag_index=i;
						creating = false;
					}
				}

				if (drag_index==-1) {
					creating = true;
					rect = Rect2(drag_from,Size2());
				}

			} else if (drag) {

				undo_redo->create_action("Set region_rect");
				undo_redo->add_do_method(node,"set_region_rect",node->get_region_rect());
				undo_redo->add_undo_method(node,"set_region_rect",rect_prev);
				undo_redo->add_do_method(edit_draw,"update");
				undo_redo->add_undo_method(edit_draw,"update");
				undo_redo->commit_action();

				drag=false;
			}

		} else if (mb.button_index==BUTTON_RIGHT && mb.pressed) {

			if (drag) {
				drag=false;
				node->set_region_rect(rect_prev);
				rect=rect_prev;
				edit_draw->update();
			}

		} else if (mb.button_index==BUTTON_WHEEL_UP && mb.pressed) {

			zoom->set_val( zoom->get_val()/0.9 );
		} else if (mb.button_index==BUTTON_WHEEL_DOWN && mb.pressed) {

			zoom->set_val( zoom->get_val()*0.9);
		}

	} else if (p_input.type==InputEvent::MOUSE_MOTION) {

		const InputEventMouseMotion &mm=p_input.mouse_motion;

		if (mm.button_mask&BUTTON_MASK_MIDDLE || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {

			Vector2 draged(mm.relative_x,mm.relative_y);
			hscroll->set_val( hscroll->get_val()-draged.x );
			vscroll->set_val( vscroll->get_val()-draged.y );

		} else if (drag) {

			Vector2 new_pos = mtx.affine_inverse().xform(Vector2(mm.x,mm.y));
			new_pos = snap_point(new_pos);

			if (creating) {
				rect = Rect2(drag_from,Size2());
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
				edit_draw->update();
				return;
			}

			switch(drag_index) {
			case 0: {
				Vector2 p=rect_prev.pos+rect_prev.size;
				rect = Rect2(p,Size2());
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 1: {
				Vector2 p=rect_prev.pos+Vector2(0,rect_prev.size.y);
				rect = Rect2(p,Size2(rect_prev.size.x,0));
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 2: {
				Vector2 p=rect_prev.pos+Vector2(0,rect_prev.size.y);
				rect = Rect2(p,Size2());
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 3: {
				Vector2 p=rect_prev.pos;
				rect = Rect2(p,Size2(0,rect_prev.size.y));
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 4: {
				Vector2 p=rect_prev.pos;
				rect = Rect2(p,Size2());
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 5: {
				Vector2 p=rect_prev.pos;
				rect = Rect2(p,Size2(rect_prev.size.x,0));
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 6: {
				Vector2 p=rect_prev.pos+Vector2(rect_prev.size.x,0);
				rect = Rect2(p,Size2());
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;
			case 7: {
				Vector2 p=rect_prev.pos+Vector2(rect_prev.size.x,0);
				rect = Rect2(p,Size2(0,rect_prev.size.y));
				rect.expand_to(new_pos);
				node->set_region_rect(rect);
			} break;

			}
			edit_draw->update();
		}

	}
}

void SpriteRegionEditor::_scroll_changed(float)
{
	if (updating_scroll)
		return;

	draw_ofs.x=hscroll->get_val();
	draw_ofs.y=vscroll->get_val();
	draw_zoom=zoom->get_val();
	print_line("_scroll_changed");
	edit_draw->update();
}

void SpriteRegionEditor::_set_use_snap(bool p_use)
{
	use_snap=p_use;
}

void SpriteRegionEditor::_set_show_grid(bool p_show)
{
	snap_show_grid=p_show;
	edit_draw->update();
}

void SpriteRegionEditor::_set_snap_off_x(float p_val)
{
	snap_offset.x=p_val;
	edit_draw->update();
}

void SpriteRegionEditor::_set_snap_off_y(float p_val)
{
	snap_offset.y=p_val;
	edit_draw->update();
}

void SpriteRegionEditor::_set_snap_step_x(float p_val)
{
	snap_step.x=p_val;
	edit_draw->update();
}

void SpriteRegionEditor::_set_snap_step_y(float p_val)
{
	snap_step.y=p_val;
	edit_draw->update();
}

void SpriteRegionEditor::_notification(int p_what)
{
	switch(p_what) {

		case NOTIFICATION_READY: {
			edit_node->set_icon( get_icon("RegionEdit","EditorIcons"));
			b_snap_grid->set_icon( get_icon("Grid", "EditorIcons"));
			b_snap_enable->set_icon( get_icon("Snap", "EditorIcons"));
			icon_zoom->set_texture( get_icon("Zoom", "EditorIcons"));
		} break;
	}
}

void SpriteRegionEditor::_node_removed(Node *p_node)
{
	if(p_node==node) {
		node=NULL;
		hide();
	}
}

void SpriteRegionEditor::_bind_methods()
{
	ObjectTypeDB::bind_method(_MD("_edit_node"),&SpriteRegionEditor::_edit_node);
	ObjectTypeDB::bind_method(_MD("_region_draw"),&SpriteRegionEditor::_region_draw);
	ObjectTypeDB::bind_method(_MD("_region_input"),&SpriteRegionEditor::_region_input);
	ObjectTypeDB::bind_method(_MD("_scroll_changed"),&SpriteRegionEditor::_scroll_changed);
	ObjectTypeDB::bind_method(_MD("_node_removed"),&SpriteRegionEditor::_node_removed);
	ObjectTypeDB::bind_method(_MD("_set_use_snap"),&SpriteRegionEditor::_set_use_snap);
	ObjectTypeDB::bind_method(_MD("_set_show_grid"),&SpriteRegionEditor::_set_show_grid);
	ObjectTypeDB::bind_method(_MD("_set_snap_off_x"),&SpriteRegionEditor::_set_snap_off_x);
	ObjectTypeDB::bind_method(_MD("_set_snap_off_y"),&SpriteRegionEditor::_set_snap_off_y);
	ObjectTypeDB::bind_method(_MD("_set_snap_step_x"),&SpriteRegionEditor::_set_snap_step_x);
	ObjectTypeDB::bind_method(_MD("_set_snap_step_y"),&SpriteRegionEditor::_set_snap_step_y);
}

void SpriteRegionEditor::edit(Node *p_sprite)
{
	if (p_sprite) {
		node=p_sprite->cast_to<Sprite>();
		node->connect("exit_tree",this,"_node_removed",varray(p_sprite),CONNECT_ONESHOT);
	} else {
		if (node)
			node->disconnect("exit_tree",this,"_node_removed");
		node=NULL;
	}

}
void SpriteRegionEditor::_edit_node()
{
	if (node->get_texture().is_null()) {

		error->set_text("No texture in this sprite.\nSet a texture to be able to edit Region.");
		error->popup_centered_minsize();
		return;
	}

	rect=node->get_region_rect();
	dlg_editor->popup_centered_ratio(0.85);
}

inline float _snap_scalar(float p_offset, float p_step, float p_target) {
	return p_step != 0 ? Math::stepify(p_target - p_offset, p_step) + p_offset : p_target;
}

Vector2 SpriteRegionEditor::snap_point(Vector2 p_target) const {
	if (use_snap) {
		p_target.x = _snap_scalar(snap_offset.x, snap_step.x, p_target.x);
		p_target.y = _snap_scalar(snap_offset.y, snap_step.y, p_target.y);
	}
	p_target = p_target.snapped(Size2(1, 1));

	return p_target;
}

SpriteRegionEditor::SpriteRegionEditor(EditorNode* p_editor)
{
	node=NULL;
	editor=p_editor;
	undo_redo = editor->get_undo_redo();

	snap_step=Vector2(10,10);
	use_snap=false;
	snap_show_grid=false;
	drag=false;

	add_child( memnew( VSeparator ));
	edit_node = memnew( ToolButton );
	add_child(edit_node);
	edit_node->connect("pressed",this,"_edit_node");

	dlg_editor = memnew( AcceptDialog );
	add_child(dlg_editor);
	dlg_editor->set_title("Sprite Region Editor");
	dlg_editor->set_self_opacity(0.9);

	VBoxContainer *main_vb = memnew( VBoxContainer );
	dlg_editor->add_child(main_vb);
	dlg_editor->set_child_rect(main_vb);
	HBoxContainer *hb_tools = memnew( HBoxContainer );
	main_vb->add_child(hb_tools);

	b_snap_enable = memnew( ToolButton );
	hb_tools->add_child(b_snap_enable);
	b_snap_enable->set_text("Snap");
	b_snap_enable->set_focus_mode(FOCUS_NONE);
	b_snap_enable->set_toggle_mode(true);
	b_snap_enable->set_pressed(use_snap);
	b_snap_enable->set_tooltip("Enable Snap");
	b_snap_enable->connect("toggled",this,"_set_use_snap");

	b_snap_grid = memnew( ToolButton );
	hb_tools->add_child(b_snap_grid);
	b_snap_grid->set_text("Grid");
	b_snap_grid->set_focus_mode(FOCUS_NONE);
	b_snap_grid->set_toggle_mode(true);
	b_snap_grid->set_pressed(snap_show_grid);
	b_snap_grid->set_tooltip("Show Grid");
	b_snap_grid->connect("toggled",this,"_set_show_grid");

	hb_tools->add_child( memnew( VSeparator ));
	hb_tools->add_child( memnew( Label("Grid Offset:") ) );

	sb_off_x = memnew( SpinBox );
	sb_off_x->set_min(-256);
	sb_off_x->set_max(256);
	sb_off_x->set_step(1);
	sb_off_x->set_val(snap_offset.x);
	sb_off_x->set_suffix("px");
	sb_off_x->connect("value_changed", this, "_set_snap_off_x");
	hb_tools->add_child(sb_off_x);

	sb_off_y = memnew( SpinBox );
	sb_off_y->set_min(-256);
	sb_off_y->set_max(256);
	sb_off_y->set_step(1);
	sb_off_y->set_val(snap_offset.y);
	sb_off_y->set_suffix("px");
	sb_off_y->connect("value_changed", this, "_set_snap_off_y");
	hb_tools->add_child(sb_off_y);

	hb_tools->add_child( memnew( VSeparator ));
	hb_tools->add_child( memnew( Label("Grid Step:") ) );

	sb_step_x = memnew( SpinBox );
	sb_step_x->set_min(-256);
	sb_step_x->set_max(256);
	sb_step_x->set_step(1);
	sb_step_x->set_val(snap_step.x);
	sb_step_x->set_suffix("px");
	sb_step_x->connect("value_changed", this, "_set_snap_step_x");
	hb_tools->add_child(sb_step_x);

	sb_step_y = memnew( SpinBox );
	sb_step_y->set_min(-256);
	sb_step_y->set_max(256);
	sb_step_y->set_step(1);
	sb_step_y->set_val(snap_step.y);
	sb_step_y->set_suffix("px");
	sb_step_y->connect("value_changed", this, "_set_snap_step_y");
	hb_tools->add_child(sb_step_y);

	HBoxContainer *main_hb = memnew( HBoxContainer );
	main_vb->add_child(main_hb);
	edit_draw = memnew( Control );
	main_hb->add_child(edit_draw);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);
	edit_draw->set_h_size_flags(SIZE_EXPAND_FILL);


	hb_tools->add_child( memnew( VSeparator ));
	icon_zoom = memnew( TextureFrame );
	hb_tools->add_child(icon_zoom);

	zoom = memnew( HSlider );
	zoom->set_min(0.01);
	zoom->set_max(4);
	zoom->set_val(1);
	zoom->set_step(0.01);
	hb_tools->add_child(zoom);
	zoom->set_custom_minimum_size(Size2(200,0));
	zoom_value = memnew( SpinBox );
	zoom->share(zoom_value);
	zoom_value->set_custom_minimum_size(Size2(50,0));
	hb_tools->add_child(zoom_value);
	zoom->connect("value_changed",this,"_scroll_changed");



	vscroll = memnew( VScrollBar);
	main_hb->add_child(vscroll);
	vscroll->connect("value_changed",this,"_scroll_changed");
	hscroll = memnew( HScrollBar );
	main_vb->add_child(hscroll);
	hscroll->connect("value_changed",this,"_scroll_changed");

	edit_draw->connect("draw",this,"_region_draw");
	edit_draw->connect("input_event",this,"_region_input");
	draw_zoom=1.0;
	updating_scroll=false;

	error = memnew( AcceptDialog);
	add_child(error);

}

void SpriteRegionEditorPlugin::edit(Object *p_node)
{
	region_editor->edit(p_node->cast_to<Node>());
}

bool SpriteRegionEditorPlugin::handles(Object *p_node) const
{
	return p_node->is_type("Sprite");
}

void SpriteRegionEditorPlugin::make_visible(bool p_visible)
{
	if (p_visible) {
		region_editor->show();
	} else {
		region_editor->hide();
		region_editor->edit(NULL);
	}
}


Dictionary SpriteRegionEditorPlugin::get_state() const {

	Dictionary state;
	state["zoom"]=region_editor->zoom->get_val();
	state["snap_offset"]=region_editor->snap_offset;
	state["snap_step"]=region_editor->snap_step;
	state["use_snap"]=region_editor->use_snap;
	state["snap_show_grid"]=region_editor->snap_show_grid;
	return state;
}

void SpriteRegionEditorPlugin::set_state(const Dictionary& p_state){

	Dictionary state=p_state;
	if (state.has("zoom")) {
		region_editor->zoom->set_val(p_state["zoom"]);
	}

	if (state.has("snap_step")) {
		Vector2 s = state["snap_step"];
		region_editor->sb_step_x->set_val(s.x);
		region_editor->sb_step_y->set_val(s.y);
		region_editor->snap_step = s;
	}

	if (state.has("snap_offset")) {
		Vector2 ofs = state["snap_offset"];
		region_editor->sb_off_x->set_val(ofs.x);
		region_editor->sb_off_y->set_val(ofs.y);
		region_editor->snap_offset = ofs;
	}

	if (state.has("use_snap")) {
		region_editor->use_snap=state["use_snap"];
		region_editor->b_snap_enable->set_pressed(state["use_snap"]);
	}

	if (state.has("snap_show_grid")) {
		region_editor->snap_show_grid=state["snap_show_grid"];
		region_editor->b_snap_grid->set_pressed(state["snap_show_grid"]);
	}
}

SpriteRegionEditorPlugin::SpriteRegionEditorPlugin(EditorNode *p_node)
{
	editor = p_node;
	region_editor= memnew ( SpriteRegionEditor(p_node) );
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(region_editor);

	region_editor->hide();
}

