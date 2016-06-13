/*************************************************************************/
/*  texture_region_editor_plugin.cpp                                      */
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

#include "texture_region_editor_plugin.h"
#include "scene/gui/check_box.h"
#include "os/input.h"
#include "os/keyboard.h"

void TextureRegionEditor::_region_draw()
{
	Ref<Texture> base_tex = NULL;
	if(node_type == "Sprite" && node_sprite)
		base_tex = node_sprite->get_texture();
	else if(node_type == "Patch9Frame" && node_patch9)
		base_tex = node_patch9->get_texture();
	else if(node_type == "StyleBoxTexture" && obj_styleBox)
		base_tex = obj_styleBox->get_texture();
	else if(node_type == "AtlasTexture" && atlas_tex)
		base_tex = atlas_tex->get_atlas();
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
	Color color(0.9,0.5,0.5);
	if(this->editing_region == REGION_PATCH_MARGIN)
		color = Color(0.21, 0.79, 0.31);
	for(int i=0;i<4;i++) {

		int prev = (i+3)%4;
		int next = (i+1)%4;

		Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
		ofs*=1.4144*(select_handle->get_size().width/2);

		edit_draw->draw_line(endpoints[i]-draw_ofs, endpoints[next]-draw_ofs, color , 2);

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

void TextureRegionEditor::_region_input(const InputEvent& p_input)
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
				if(node_type == "Sprite" && node_sprite )
					rect_prev=node_sprite->get_region_rect();
				else if(node_type == "AtlasTexture" && atlas_tex)
					rect_prev=atlas_tex->get_region();
				else if(node_type == "Patch9Frame" && node_patch9)
					rect_prev=node_patch9->get_region_rect();
				else if(node_type == "StyleBoxTexture" && obj_styleBox)
					rect_prev=obj_styleBox->get_region_rect();

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
				if(editing_region == REGION_TEXTURE_REGION) {
					undo_redo->create_action("Set region_rect");
					if(node_type == "Sprite" && node_sprite ){
						undo_redo->add_do_method(node_sprite ,"set_region_rect",node_sprite->get_region_rect());
						undo_redo->add_undo_method(node_sprite,"set_region_rect",rect_prev);
					}
					else if(node_type == "AtlasTexture" && atlas_tex ){
						undo_redo->add_do_method(atlas_tex ,"set_region",atlas_tex->get_region());
						undo_redo->add_undo_method(atlas_tex,"set_region",rect_prev);
					}
					else if(node_type == "Patch9Frame" && node_patch9){
						undo_redo->add_do_method(node_patch9 ,"set_region_rect",node_patch9->get_region_rect());
						undo_redo->add_undo_method(node_patch9,"set_region_rect",rect_prev);
					}
					else if(node_type == "StyleBoxTexture" && obj_styleBox){
						undo_redo->add_do_method(obj_styleBox ,"set_region_rect",obj_styleBox->get_region_rect());
						undo_redo->add_undo_method(obj_styleBox,"set_region_rect",rect_prev);
					}
					undo_redo->add_do_method(edit_draw,"update");
					undo_redo->add_undo_method(edit_draw,"update");
					undo_redo->commit_action();
				}
				drag=false;
			}

		} else if (mb.button_index==BUTTON_RIGHT && mb.pressed) {

			if (drag) {
				drag=false;
				apply_rect(rect_prev);
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
				apply_rect(rect);
				edit_draw->update();
				return;
			}

			switch(drag_index) {
			case 0: {
					Vector2 p=rect_prev.pos+rect_prev.size;
					rect = Rect2(p,Size2());
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 1: {
					Vector2 p=rect_prev.pos+Vector2(0,rect_prev.size.y);
					rect = Rect2(p,Size2(rect_prev.size.x,0));
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 2: {
					Vector2 p=rect_prev.pos+Vector2(0,rect_prev.size.y);
					rect = Rect2(p,Size2());
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 3: {
					Vector2 p=rect_prev.pos;
					rect = Rect2(p,Size2(0,rect_prev.size.y));
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 4: {
					Vector2 p=rect_prev.pos;
					rect = Rect2(p,Size2());
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 5: {
					Vector2 p=rect_prev.pos;
					rect = Rect2(p,Size2(rect_prev.size.x,0));
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 6: {
					Vector2 p=rect_prev.pos+Vector2(rect_prev.size.x,0);
					rect = Rect2(p,Size2());
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;
			case 7: {
					Vector2 p=rect_prev.pos+Vector2(rect_prev.size.x,0);
					rect = Rect2(p,Size2(0,rect_prev.size.y));
					rect.expand_to(new_pos);
					apply_rect(rect);
				} break;

			}
			edit_draw->update();
		}

	}
}

void TextureRegionEditor::_scroll_changed(float)
{
	if (updating_scroll)
		return;

	draw_ofs.x=hscroll->get_val();
	draw_ofs.y=vscroll->get_val();
	draw_zoom=zoom->get_val();
	print_line("_scroll_changed");
	edit_draw->update();
}

void TextureRegionEditor::_set_use_snap(bool p_use)
{
	use_snap=p_use;
}

void TextureRegionEditor::_set_show_grid(bool p_show)
{
	snap_show_grid=p_show;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_off_x(float p_val)
{
	snap_offset.x=p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_off_y(float p_val)
{
	snap_offset.y=p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_step_x(float p_val)
{
	snap_step.x=p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_step_y(float p_val)
{
	snap_step.y=p_val;
	edit_draw->update();
}

void TextureRegionEditor::apply_rect(const Rect2& rect){

	if(this->editing_region == REGION_TEXTURE_REGION) {
		if(node_sprite)
			node_sprite->set_region_rect(rect);
		else if(node_patch9)
			node_patch9->set_region_rect(rect);
		else if(obj_styleBox)
			obj_styleBox->set_region_rect(rect);
		else if(atlas_tex)
			atlas_tex->set_region(rect);
	}
	else if(this->editing_region == REGION_PATCH_MARGIN) {
		if(node_patch9) {
			node_patch9->set_patch_margin(MARGIN_LEFT, rect.pos.x - tex_region.pos.x);
			node_patch9->set_patch_margin(MARGIN_RIGHT, tex_region.pos.x+tex_region.size.width-(rect.pos.x+rect.size.width));
			node_patch9->set_patch_margin(MARGIN_TOP, rect.pos.y - tex_region.pos.y);
			node_patch9->set_patch_margin(MARGIN_BOTTOM, tex_region.pos.y+tex_region.size.height-(rect.pos.y+rect.size.height));
		}
		else if(obj_styleBox) {
			obj_styleBox->set_margin_size(MARGIN_LEFT, rect.pos.x - tex_region.pos.x);
			obj_styleBox->set_margin_size(MARGIN_RIGHT, tex_region.pos.x+tex_region.size.width-(rect.pos.x+rect.size.width));
			obj_styleBox->set_margin_size(MARGIN_TOP, rect.pos.y - tex_region.pos.y);
			obj_styleBox->set_margin_size(MARGIN_BOTTOM, tex_region.pos.y+tex_region.size.height-(rect.pos.y+rect.size.height));
		}
	}
}

void TextureRegionEditor::_notification(int p_what)
{
	switch(p_what) {
	case NOTIFICATION_READY: {
			region_button->set_icon( get_icon("RegionEdit","EditorIcons"));
			margin_button->set_icon( get_icon("Patch9Frame", "EditorIcons"));
			b_snap_grid->set_icon( get_icon("Grid", "EditorIcons"));
			b_snap_enable->set_icon( get_icon("Snap", "EditorIcons"));
			icon_zoom->set_texture( get_icon("Zoom", "EditorIcons"));
		} break;
	}
}

void TextureRegionEditor::_node_removed(Object *p_obj)
{
	if(p_obj == node_sprite || p_obj == node_patch9 || p_obj == obj_styleBox || p_obj == atlas_tex) {
		node_patch9  = NULL;
		node_sprite  = NULL;
		obj_styleBox = NULL;
		atlas_tex    = NULL;
		hide();
	}
}

void TextureRegionEditor::_bind_methods()
{
	ObjectTypeDB::bind_method(_MD("_edit_node"),&TextureRegionEditor::_edit_node);
	ObjectTypeDB::bind_method(_MD("_edit_region"),&TextureRegionEditor::_edit_region);
	ObjectTypeDB::bind_method(_MD("_edit_margin"),&TextureRegionEditor::_edit_margin);
	ObjectTypeDB::bind_method(_MD("_region_draw"),&TextureRegionEditor::_region_draw);
	ObjectTypeDB::bind_method(_MD("_region_input"),&TextureRegionEditor::_region_input);
	ObjectTypeDB::bind_method(_MD("_scroll_changed"),&TextureRegionEditor::_scroll_changed);
	ObjectTypeDB::bind_method(_MD("_node_removed"),&TextureRegionEditor::_node_removed);
	ObjectTypeDB::bind_method(_MD("_set_use_snap"),&TextureRegionEditor::_set_use_snap);
	ObjectTypeDB::bind_method(_MD("_set_show_grid"),&TextureRegionEditor::_set_show_grid);
	ObjectTypeDB::bind_method(_MD("_set_snap_off_x"),&TextureRegionEditor::_set_snap_off_x);
	ObjectTypeDB::bind_method(_MD("_set_snap_off_y"),&TextureRegionEditor::_set_snap_off_y);
	ObjectTypeDB::bind_method(_MD("_set_snap_step_x"),&TextureRegionEditor::_set_snap_step_x);
	ObjectTypeDB::bind_method(_MD("_set_snap_step_y"),&TextureRegionEditor::_set_snap_step_y);
}

void TextureRegionEditor::edit(Object *p_obj)
{
	if (p_obj) {
		margin_button->hide();
		node_type = p_obj->get_type();
		if(node_type == "Sprite"){
			node_sprite = p_obj->cast_to<Sprite>();
			node_patch9 = NULL;
			obj_styleBox = NULL;
			atlas_tex   = NULL;
		}
		else if(node_type == "AtlasTexture") {
			atlas_tex   = p_obj->cast_to<AtlasTexture>();
			node_sprite = NULL;
			node_patch9 = NULL;
			obj_styleBox = NULL;
		}
		else if(node_type == "Patch9Frame") {
			node_patch9 = p_obj->cast_to<Patch9Frame>();
			node_sprite = NULL;
			obj_styleBox = NULL;
			atlas_tex = NULL;
			margin_button->show();
		}
		else if(node_type == "StyleBoxTexture") {
			obj_styleBox = p_obj->cast_to<StyleBoxTexture>();
			node_sprite = NULL;
			node_patch9 = NULL;
			atlas_tex = NULL;
			margin_button->show();
		}
		p_obj->connect("exit_tree",this,"_node_removed",varray(p_obj),CONNECT_ONESHOT);
	} else {
		if(node_sprite)
			node_sprite->disconnect("exit_tree",this,"_node_removed");
		else if(atlas_tex)
			atlas_tex->disconnect("exit_tree",this,"_node_removed");
		else if(node_patch9)
			node_patch9->disconnect("exit_tree",this,"_node_removed");
		else if(obj_styleBox)
			obj_styleBox->disconnect("exit_tree",this,"_node_removed");
		node_sprite  = NULL;
		node_patch9  = NULL;
		obj_styleBox = NULL;
		atlas_tex    = NULL;
	}
}

void TextureRegionEditor::_edit_region()
{
	this->_edit_node(REGION_TEXTURE_REGION);
	dlg_editor->set_title(TTR("Texture Region Editor"));
}

void TextureRegionEditor::_edit_margin()
{
	this->_edit_node(REGION_PATCH_MARGIN);
	dlg_editor->set_title(TTR("Scale Region Editor"));
}

void TextureRegionEditor::_edit_node(int region)
{
	Ref<Texture> texture = NULL;
	if(node_type == "Sprite" && node_sprite )
		texture = node_sprite->get_texture();
	else if(node_type == "Patch9Frame" && node_patch9 )
		texture = node_patch9->get_texture();
	else if(node_type == "StyleBoxTexture" && obj_styleBox)
		texture = obj_styleBox->get_texture();
	else if(node_type == "AtlasTexture" && atlas_tex)
		texture = atlas_tex->get_atlas();

	if (texture.is_null()) {
		error->set_text(TTR("No texture in this node.\nSet a texture to be able to edit region."));
		error->popup_centered_minsize();
		return;
	}

	if(node_type == "Sprite" && node_sprite )
		tex_region = node_sprite->get_region_rect();
	else if(node_type == "Patch9Frame" && node_patch9 )
		tex_region = node_patch9->get_region_rect();
	else if(node_type == "StyleBoxTexture" && obj_styleBox)
		tex_region = obj_styleBox->get_region_rect();
	else if(node_type == "AtlasTexture" && atlas_tex)
		tex_region = atlas_tex->get_region();
	rect = tex_region;

	if(region == REGION_PATCH_MARGIN) {
		if(node_patch9){
			Patch9Frame *node = node_patch9;
			rect.pos += Point2(node->get_patch_margin(MARGIN_LEFT),node->get_patch_margin(MARGIN_TOP));
			rect.size -= Size2(node->get_patch_margin(MARGIN_RIGHT)+node->get_patch_margin(MARGIN_LEFT), node->get_patch_margin(MARGIN_BOTTOM)+node->get_patch_margin(MARGIN_TOP));
		}
		else if(obj_styleBox) {
			StyleBoxTexture * node = obj_styleBox;
			rect.pos += Point2(node->get_margin_size(MARGIN_LEFT),node->get_margin_size(MARGIN_TOP));
			rect.size -= Size2(node->get_margin_size(MARGIN_RIGHT)+node->get_margin_size(MARGIN_LEFT), node->get_margin_size(MARGIN_BOTTOM)+node->get_margin_size(MARGIN_TOP));
		}
	}

	dlg_editor->popup_centered_ratio(0.85);
	dlg_editor->get_ok()->release_focus();

	editing_region = region;
}

inline float _snap_scalar(float p_offset, float p_step, float p_target) {
	return p_step != 0 ? Math::stepify(p_target - p_offset, p_step) + p_offset : p_target;
}

Vector2 TextureRegionEditor::snap_point(Vector2 p_target) const {
	if (use_snap) {
		p_target.x = _snap_scalar(snap_offset.x, snap_step.x, p_target.x);
		p_target.y = _snap_scalar(snap_offset.y, snap_step.y, p_target.y);
	}
	p_target = p_target.snapped(Size2(1, 1));

	return p_target;
}

TextureRegionEditor::TextureRegionEditor(EditorNode* p_editor)
{
	node_sprite = NULL;
	node_patch9 = NULL;
	atlas_tex   = NULL;
	editor=p_editor;
	undo_redo = editor->get_undo_redo();

	snap_step=Vector2(10,10);
	use_snap=false;
	snap_show_grid=false;
	drag=false;

	add_child( memnew( VSeparator ));
	region_button = memnew( ToolButton );
	add_child(region_button);
	region_button->set_tooltip(TTR("Texture Region Editor"));
	region_button->connect("pressed",this,"_edit_region");

	margin_button = memnew( ToolButton );
	add_child(margin_button);
	margin_button->set_tooltip(TTR("Scale Region Editor"));
	margin_button->connect("pressed",this,"_edit_margin");

	dlg_editor = memnew( AcceptDialog );
	add_child(dlg_editor);
	dlg_editor->set_self_opacity(0.9);

	VBoxContainer *main_vb = memnew( VBoxContainer );
	dlg_editor->add_child(main_vb);
	dlg_editor->set_child_rect(main_vb);
	HBoxContainer *hb_tools = memnew( HBoxContainer );
	main_vb->add_child(hb_tools);

	b_snap_enable = memnew( ToolButton );
	hb_tools->add_child(b_snap_enable);
	b_snap_enable->set_text(TTR("Snap"));
	b_snap_enable->set_focus_mode(FOCUS_NONE);
	b_snap_enable->set_toggle_mode(true);
	b_snap_enable->set_pressed(use_snap);
	b_snap_enable->set_tooltip(TTR("Enable Snap"));
	b_snap_enable->connect("toggled",this,"_set_use_snap");

	b_snap_grid = memnew( ToolButton );
	hb_tools->add_child(b_snap_grid);
	b_snap_grid->set_text(TTR("Grid"));
	b_snap_grid->set_focus_mode(FOCUS_NONE);
	b_snap_grid->set_toggle_mode(true);
	b_snap_grid->set_pressed(snap_show_grid);
	b_snap_grid->set_tooltip(TTR("Show Grid"));
	b_snap_grid->connect("toggled",this,"_set_show_grid");

	hb_tools->add_child( memnew( VSeparator ));
	hb_tools->add_child( memnew( Label(TTR("Grid Offset:")) ) );

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
	hb_tools->add_child( memnew( Label(TTR("Grid Step:")) ) );

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

void TextureRegionEditorPlugin::edit(Object *p_node)
{
	region_editor->edit(p_node);
}

bool TextureRegionEditorPlugin::handles(Object *p_obj) const
{
	return p_obj->is_type("Sprite") || p_obj->is_type("Patch9Frame") || p_obj->is_type("StyleBoxTexture") || p_obj->is_type("AtlasTexture");
}

void TextureRegionEditorPlugin::make_visible(bool p_visible)
{
	if (p_visible) {
		region_editor->show();
	} else {
		region_editor->hide();
		region_editor->edit(NULL);
	}
}


Dictionary TextureRegionEditorPlugin::get_state() const {

	Dictionary state;
	state["zoom"]=region_editor->zoom->get_val();
	state["snap_offset"]=region_editor->snap_offset;
	state["snap_step"]=region_editor->snap_step;
	state["use_snap"]=region_editor->use_snap;
	state["snap_show_grid"]=region_editor->snap_show_grid;
	return state;
}

void TextureRegionEditorPlugin::set_state(const Dictionary& p_state){

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

TextureRegionEditorPlugin::TextureRegionEditorPlugin(EditorNode *p_node)
{
	editor = p_node;
	region_editor= memnew ( TextureRegionEditor(p_node) );
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(region_editor);

	region_editor->hide();
}
