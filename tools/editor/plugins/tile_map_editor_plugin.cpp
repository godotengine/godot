/*************************************************************************/
/*  tile_map_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "tile_map_editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "os/keyboard.h"

#include "canvas_item_editor_plugin.h"
#include "os/file_access.h"
#include "tools/editor/editor_settings.h"
#include "os/input.h"
#include "method_bind_ext.inc"

void TileMapEditor::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_READY: {

			transpose->set_icon( get_icon("Transpose","EditorIcons"));
			mirror_x->set_icon( get_icon("MirrorX","EditorIcons"));
			mirror_y->set_icon( get_icon("MirrorY","EditorIcons"));
			rotate_0->set_icon( get_icon("Rotate0","EditorIcons"));
			rotate_90->set_icon( get_icon("Rotate90","EditorIcons"));
			rotate_180->set_icon( get_icon("Rotate180","EditorIcons"));
			rotate_270->set_icon( get_icon("Rotate270","EditorIcons"));

		} break;
	}

}

void TileMapEditor::_canvas_mouse_enter()  {

	mouse_over=true;
	canvas_item_editor->update();


}

void TileMapEditor::_canvas_mouse_exit()  {

	mouse_over=false;
	canvas_item_editor->update();

}

int TileMapEditor::get_selected_tile() const {
	int item = palette->get_current();
	if (item==-1)
		return TileMap::INVALID_CELL;
	return palette->get_item_metadata(item);
}

void TileMapEditor::set_selected_tile(int p_tile) {
	for (int i = 0; i < palette->get_item_count(); i++) {
		if (palette->get_item_metadata(i).operator int() == p_tile) {
			palette->select(i,true);
			palette->ensure_current_is_visible();
			break;
		}
	}
}

// Wrapper to workaround five arg limit of undo/redo methods
void TileMapEditor::_set_cell_shortened(const Point2& p_pos,int p_value,bool p_flip_h, bool p_flip_v, bool p_transpose) {
	ERR_FAIL_COND(!node);
	node->set_cell(floor(p_pos.x), floor(p_pos.y), p_value, p_flip_h, p_flip_v, p_transpose);
}

void TileMapEditor::_set_cell(const Point2i& p_pos,int p_value,bool p_flip_h, bool p_flip_v, bool p_transpose,bool p_with_undo) {

	ERR_FAIL_COND(!node);

	bool prev_flip_h=node->is_cell_x_flipped(p_pos.x,p_pos.y);
	bool prev_flip_v=node->is_cell_y_flipped(p_pos.x,p_pos.y);
	bool prev_transpose=node->is_cell_transposed(p_pos.x,p_pos.y);
	int prev_val=node->get_cell(p_pos.x,p_pos.y);

	if (p_value==prev_val && p_flip_h==prev_flip_h && p_flip_v==prev_flip_v && p_transpose==prev_transpose)
		return; //check that it's actually different


	if (p_with_undo) {
		undo_redo->add_do_method(node,"set_cellv",Point2(p_pos),p_value,p_flip_h,p_flip_v,p_transpose);
		undo_redo->add_undo_method(node,"set_cellv",Point2(p_pos),prev_val,prev_flip_h,prev_flip_v,prev_transpose);
	} else {

		node->set_cell(p_pos.x,p_pos.y,p_value,p_flip_h,p_flip_v,p_transpose);

	}

}

void TileMapEditor::_set_display_mode(int p_mode) {
	if (display_mode == p_mode) {
		return;
	}

	switch (p_mode) {
		case DISPLAY_THUMBNAIL: {
			button_thumbnail->set_pressed(true);
			button_list->set_pressed(false);
		} break;
		case DISPLAY_LIST: {
			button_thumbnail->set_pressed(false);
			button_list->set_pressed(true);
		} break;
	}

	display_mode = p_mode;

	_update_palette();
}

void TileMapEditor::_update_palette() {

	if (!node)
		return;

	palette->clear();

	Ref<TileSet> tileset=node->get_tileset();
	if (!tileset.is_valid())
		return;

	List<int> tiles;
	tileset->get_tile_list(&tiles);

	if (display_mode == DISPLAY_THUMBNAIL) {
		palette->set_max_columns(0);
		palette->set_icon_mode(ItemList::ICON_MODE_TOP);
	} else if (display_mode == DISPLAY_LIST) {
		palette->set_max_columns(1);
		palette->set_icon_mode(ItemList::ICON_MODE_LEFT);
	}

	palette->set_max_text_lines(2);

	for(List<int>::Element *E=tiles.front();E;E=E->next()) {
		palette->add_item("");

		Ref<Texture> tex = tileset->tile_get_texture(E->get());

		if (tex.is_valid()) {
			Rect2 region = tileset->tile_get_region(E->get());

			if (!region.has_no_area()) {
				Image data = VS::get_singleton()->texture_get_data(tex->get_rid());

				Ref<ImageTexture> img = memnew( ImageTexture );
				img->create_from_image(data.get_rect(region));

				palette->set_item_icon(palette->get_item_count()-1, img);
			} else {
				palette->set_item_icon(palette->get_item_count()-1,tex);
			}
		}

		if (tileset->tile_get_name(E->get())!="") {
			palette->set_item_text(palette->get_item_count()-1, tileset->tile_get_name(E->get()));
		} else {
			palette->set_item_text(palette->get_item_count()-1, "#"+itos(E->get()));
		}

		palette->set_item_metadata(palette->get_item_count()-1, E->get());
	}
}

void TileMapEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		node=NULL;
		hide();
	}

}
void TileMapEditor::_menu_option(int p_option) {

	switch(p_option) {


	}
}

struct _TileMapEditorCopyData {
	Point2i pos;
	int cell;
	bool flip_h;
	bool flip_v;
	bool transpose;
};

bool TileMapEditor::forward_input_event(const InputEvent& p_event) {


	if (!node || !node->get_tileset().is_valid() || !node->is_visible())
		return false;

	Matrix32 xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Matrix32 xform_inv = xform.affine_inverse();
	Vector2 snap = node->get_cell_size();


	switch(p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &mb=p_event.mouse_button;

			if (mb.button_index==BUTTON_LEFT) {


				if (mb.pressed && tool==TOOL_DUPLICATING) {


					List<_TileMapEditorCopyData> dupdata;
					Point2 ofs = over_tile-selection.pos;

					for(int i=selection.pos.y;i<=selection.pos.y+selection.size.y;i++) {

						for(int j=selection.pos.x;j<=selection.pos.x+selection.size.x;j++) {

							_TileMapEditorCopyData tcd;
							tcd.pos=Point2i(j,i);
							tcd.cell=node->get_cell(j,i);
							tcd.flip_h=node->is_cell_x_flipped(j,i);
							tcd.flip_v=node->is_cell_y_flipped(j,i);
							tcd.transpose=node->is_cell_transposed(j,i);
							dupdata.push_back(tcd);


						}
					}

					undo_redo->create_action("Duplicate");
					for (List<_TileMapEditorCopyData>::Element *E=dupdata.front();E;E=E->next()) {


						_set_cell(E->get().pos+ofs,E->get().cell,E->get().flip_h,E->get().flip_v,E->get().transpose,true);
					}
					undo_redo->commit_action();

					tool=TOOL_NONE;
					canvas_item_editor->update();
					selection.pos=over_tile;

				} else if (mb.pressed && tool==TOOL_NONE) {

					if (Input::get_singleton()->is_key_pressed(KEY_SPACE))
						return false; //drag
					if (mb.mod.shift) {

						tool=TOOL_SELECTING;
						selection_begin =node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));
						selection.pos=selection_begin;
						selection.size=Point2(0,0);
						selection_active=true;
						canvas_item_editor->update();
						return true;

					} else if (mb.mod.control) {
						tool=TOOL_PICKING;
						set_selected_tile(node->get_cell(over_tile.x, over_tile.y));
						mirror_x->set_pressed(node->is_cell_x_flipped(over_tile.x, over_tile.y));
						mirror_y->set_pressed(node->is_cell_y_flipped(over_tile.x, over_tile.y));
						transpose->set_pressed(node->is_cell_transposed(over_tile.x, over_tile.y));
						_update_transform_buttons();
						canvas_item_editor->update();
						return true;
					} else {
						int id = get_selected_tile();
						if (id!=TileMap::INVALID_CELL) {
							tool=TOOL_PAINTING;
							Point2i local =node->world_to_map((xform_inv.xform(Point2(mb.x,mb.y))));
							paint_undo.clear();
							paint_undo[local]=_get_op_from_cell(local);
							node->set_cell(local.x,local.y,id,mirror_x->is_pressed(),mirror_y->is_pressed(),transpose->is_pressed());
							return true;
						}
					}
				} else {

					if (tool==TOOL_PAINTING || tool == TOOL_SELECTING || tool == TOOL_PICKING) {

						if (tool==TOOL_PAINTING) {

							if (paint_undo.size()) {
								undo_redo->create_action("Paint TileMap");
								for(Map<Point2i,CellOp>::Element *E=paint_undo.front();E;E=E->next()) {

									Point2i p=E->key();
									undo_redo->add_do_method(node,"set_cellv",Point2(p),node->get_cell(p.x,p.y),node->is_cell_x_flipped(p.x,p.y),node->is_cell_y_flipped(p.x,p.y),node->is_cell_transposed(p.x,p.y));
									undo_redo->add_undo_method(node,"set_cellv",Point2(p),E->get().idx,E->get().xf,E->get().yf,E->get().tr);
								}

								undo_redo->commit_action();
								paint_undo.clear();
							}
						}
						tool=TOOL_NONE;
						return true;
					}
				}
			}

			if (mb.button_index==BUTTON_RIGHT) {

				if (mb.pressed && tool==TOOL_DUPLICATING) {

					tool=TOOL_NONE;
					canvas_item_editor->update();
				} else if (mb.pressed && tool==TOOL_NONE) {

					tool=TOOL_ERASING;
					Point2i local =node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));
					paint_undo.clear();
					paint_undo[local]=_get_op_from_cell(local);
					//node->set_cell(local.x,local.y,id,mirror_x->is_pressed(),mirror_y->is_pressed(),transpose->is_pressed());
					//return true;
					_set_cell(local,TileMap::INVALID_CELL);
					return true;
				} else if (!mb.pressed) {

					if (tool==TOOL_ERASING) {

						if (paint_undo.size()) {
							undo_redo->create_action("Erase TileMap");
							for(Map<Point2i,CellOp>::Element *E=paint_undo.front();E;E=E->next()) {

								Point2i p=E->key();
								//undo_redo->add_do_method(node,"set_cell",p,node->get_cell(p.x,p.y),node->is_cell_x_flipped(p.x,p.y),node->is_cell_y_flipped(p.x,p.y),node->is_cell_transposed(p.x,p.y));
								//_set_cell(p,TileMap::INVALID_CELL,false,false,false,true);
								undo_redo->add_do_method(node,"set_cellv",Point2(p),TileMap::INVALID_CELL,false,false,false);
								undo_redo->add_undo_method(node,"set_cellv",Point2(p),E->get().idx,E->get().xf,E->get().yf,E->get().tr);
							}

							undo_redo->commit_action();
							paint_undo.clear();

						}
						tool=TOOL_NONE;
						return true;
					}
				}
			}

		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &mm=p_event.mouse_motion;

			Point2i new_over_tile = node->world_to_map(xform_inv.xform(Point2(mm.x,mm.y)));//(xform_inv.xform(Point2(mm.x,mm.y))/snap).floor();
			if (new_over_tile!=over_tile) {

				over_tile=new_over_tile;
				canvas_item_editor->update();
			}



			if (tool==TOOL_PAINTING) {

				int id = get_selected_tile();
				if (id!=TileMap::INVALID_CELL) {

					if (!paint_undo.has(over_tile)) {

						paint_undo[over_tile]=_get_op_from_cell(over_tile);
					}
					node->set_cell(over_tile.x,over_tile.y,id,mirror_x->is_pressed(),mirror_y->is_pressed(),transpose->is_pressed());

					return true;
				}
			}

			if (tool==TOOL_SELECTING) {

				Point2i begin=selection_begin;
				Point2i end =over_tile;

				if (begin.x > end.x) {

					SWAP( begin.x, end.x);
				}
				if (begin.y > end.y) {

					SWAP( begin.y, end.y);
				}

				selection.pos=begin;
				selection.size=end-begin;
				canvas_item_editor->update();

				return true;

			}

			if (tool==TOOL_ERASING) {
				Point2i local =over_tile;
				if (!paint_undo.has(over_tile)) {
					paint_undo[over_tile]=_get_op_from_cell(over_tile);
				}
				//node->set_cell(over_tile.x,over_tile.y,id,mirror_x->is_pressed(),mirror_y->is_pressed(),transpose->is_pressed());
				_set_cell(local,TileMap::INVALID_CELL);
				return true;
			}

			if (tool==TOOL_PICKING) {
				set_selected_tile(node->get_cell(over_tile.x, over_tile.y));
				mirror_x->set_pressed(node->is_cell_x_flipped(over_tile.x, over_tile.y));
				mirror_y->set_pressed(node->is_cell_y_flipped(over_tile.x, over_tile.y));
				transpose->set_pressed(node->is_cell_transposed(over_tile.x, over_tile.y));
				_update_transform_buttons();
				canvas_item_editor->update();
				return true;
			}

		} break;
		case InputEvent::KEY: {

			const InputEventKey &k = p_event.key;
			if (!node)
				break;

			if (k.pressed && k.scancode==KEY_DELETE && selection_active && tool==TOOL_NONE) {

				undo_redo->create_action("Delete");
				for(int i=selection.pos.y;i<=selection.pos.y+selection.size.y;i++) {

					for(int j=selection.pos.x;j<=selection.pos.x+selection.size.x;j++) {


						_set_cell(Point2i(j,i),TileMap::INVALID_CELL);
					}
				}
				undo_redo->commit_action();

				selection_active=false;
				canvas_item_editor->update();
				return true;
			}

			if (mouse_over && k.pressed && k.scancode==KEY_A  && tool==TOOL_NONE && !k.mod.command) {

				/*int cell = node->get_cell(over_tile.x,over_tile.y);
				if (cell!=TileMap::INVALID_CELL) {
					bool flip_h = node->is_cell_x_flipped(over_tile.x,over_tile.y);
					bool flip_v = node->is_cell_y_flipped(over_tile.x,over_tile.y);
					_set_cell(over_tile,cell,!flip_h,flip_v);
				}*/

				mirror_x->set_pressed( ! mirror_x->is_pressed() );
				canvas_item_editor->update();
				return true;
			}
			if (mouse_over && k.pressed && k.scancode==KEY_S  && tool==TOOL_NONE && !k.mod.command) {


				/*
				int cell = node->get_cell(over_tile.x,over_tile.y);
				if (cell!=TileMap::INVALID_CELL) {

					bool flip_h = node->is_cell_x_flipped(over_tile.x,over_tile.y);
					bool flip_v = node->is_cell_y_flipped(over_tile.x,over_tile.y);
					_set_cell(over_tile,cell,flip_h,!flip_v);
				}*/

				mirror_y->set_pressed( ! mirror_y->is_pressed() );
				canvas_item_editor->update();
				return true;
			}

			if (mouse_over && selection_active && k.pressed && k.mod.command && k.scancode==KEY_D && tool==TOOL_NONE) {

				tool=TOOL_DUPLICATING;
				canvas_item_editor->update();
				return true;
			}



		} break;
	}

	return false;
}
void TileMapEditor::_canvas_draw() {

	if (!node)
		return;

	Size2 cell_size=node->get_cell_size();
	Matrix32 cell_xf = node->get_cell_transform();

	Matrix32 xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Matrix32 xform_inv = xform.affine_inverse();


	Size2 screen_size=canvas_item_editor->get_size();
	{
		Rect2 aabb;
		aabb.pos=node->world_to_map(xform_inv.xform(Vector2()));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(0,screen_size.height))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(screen_size.width,0))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(screen_size)));
		Rect2i si=aabb.grow(1.0);

		if (node->get_half_offset()!=TileMap::HALF_OFFSET_X) {

			for(int i=(si.pos.x)-1;i<=(si.pos.x+si.size.x);i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(i,si.pos.y)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(i,si.pos.y+si.size.y+1)));

				Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
				canvas_item_editor->draw_line(from,to,col,1);

			}
		} else {


			for(int i=(si.pos.x)-1;i<=(si.pos.x+si.size.x);i++) {

				for(int j=(si.pos.y)-1;j<=(si.pos.y+si.size.y);j++) {

					Vector2 ofs;
					if (ABS(j)&1) {
						ofs=cell_xf[0]*0.5;
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(i,j),true)+ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(i,j+1),true)+ofs);
					Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
					canvas_item_editor->draw_line(from,to,col,1);
				}

			}
		}

		if (node->get_half_offset()!=TileMap::HALF_OFFSET_Y) {

			for(int i=(si.pos.y)-1;i<=(si.pos.y+si.size.y);i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(si.pos.x,i)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(si.pos.x+si.size.x+1,i)));

				Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
				canvas_item_editor->draw_line(from,to,col,1);

			}
		} else {


			for(int i=(si.pos.y)-1;i<=(si.pos.y+si.size.y);i++) {

				for(int j=(si.pos.x)-1;j<=(si.pos.x+si.size.x);j++) {

					Vector2 ofs;
					if (ABS(j)&1) {
						ofs=cell_xf[1]*0.5;
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(j,i),true)+ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(j+1,i),true)+ofs);
					Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
					canvas_item_editor->draw_line(from,to,col,1);
				}

			}



		}
/*
	for(int i=(si.pos.y/cell_size.y)-1;i<=(si.pos.y+si.size.y)/cell_size.y;i++) {

		int ofs = i*cell_size.y;
		Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
		canvas_item_editor->draw_line(xform.xform(Point2(si.pos.x,ofs)),xform.xform(Point2(si.pos.x+si.size.x,ofs)),col,1);*/
	}


	if (selection_active) {

		Vector<Vector2> points;
		points.push_back( xform.xform( node->map_to_world(( selection.pos ) )));
		points.push_back( xform.xform( node->map_to_world((selection.pos+Point2(selection.size.x+1,0)) ) ));
		points.push_back( xform.xform( node->map_to_world((selection.pos+Point2(selection.size.x+1,selection.size.y+1)) ) ));
		points.push_back( xform.xform( node->map_to_world((selection.pos+Point2(0,selection.size.y+1)) ) ));
		Color col=Color(0.2,0.8,1,0.4);

		canvas_item_editor->draw_colored_polygon(points,col);
	}


	if (mouse_over){

		Vector2 endpoints[4]={

			( node->map_to_world(over_tile,true) ) ,
			( node->map_to_world((over_tile+Point2(1,0)),true ) ),
			( node->map_to_world((over_tile+Point2(1,1)),true ) ),
			( node->map_to_world((over_tile+Point2(0,1)),true ) )

		};

		for(int i=0;i<4;i++) {
			if (node->get_half_offset()==TileMap::HALF_OFFSET_X && ABS(over_tile.y)&1)
				endpoints[i]+=cell_xf[0]*0.5;
			if (node->get_half_offset()==TileMap::HALF_OFFSET_Y && ABS(over_tile.x)&1)
				endpoints[i]+=cell_xf[1]*0.5;
			endpoints[i]=xform.xform(endpoints[i]);
		}
		Color col;
		if (node->get_cell(over_tile.x,over_tile.y)!=TileMap::INVALID_CELL)
			col=Color(0.2,0.8,1.0,0.8);
		else
			col=Color(1.0,0.4,0.2,0.8);

		for(int i=0;i<4;i++)
			canvas_item_editor->draw_line(endpoints[i],endpoints[(i+1)%4],col,2);



		if (tool==TOOL_DUPLICATING) {

			Rect2i duplicate=selection;
			duplicate.pos=over_tile;


			Vector<Vector2> points;
			points.push_back( xform.xform( node->map_to_world(duplicate.pos ) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(duplicate.size.x+1,0)) ) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(duplicate.size.x+1,duplicate.size.y+1))) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(0,duplicate.size.y+1))) ));
			Color col=Color(0.2,1.0,0.8,0.4);

			canvas_item_editor->draw_colored_polygon(points,col);

		} else {

			Ref<TileSet> ts = node->get_tileset();


			if (ts.is_valid()) {

				int st = get_selected_tile();
				if (ts->has_tile(st)) {

					Ref<Texture> t = ts->tile_get_texture(st);
					if (t.is_valid()) {
						Vector2 from = node->map_to_world(over_tile)+node->get_cell_draw_offset();
						Rect2 r = ts->tile_get_region(st);
						Size2 sc = xform.get_scale();
						if (mirror_x->is_pressed())
							sc.x*=-1.0;
						if (mirror_y->is_pressed())
							sc.y*=-1.0;

						Rect2 rect;
						if (r==Rect2()) {
							rect=Rect2(from,t->get_size());
						} else {

							rect=Rect2(from,r.get_size());
						}


						if (node->get_tile_origin()==TileMap::TILE_ORIGIN_TOP_LEFT) {
							rect.pos+=ts->tile_get_texture_offset(st);

						} else if (node->get_tile_origin()==TileMap::TILE_ORIGIN_CENTER) {
							rect.pos+=node->get_cell_size()/2;
							Vector2 s = r.size;

							Vector2 center = (s/2) - ts->tile_get_texture_offset(st);


							if (mirror_x->is_pressed())
								rect.pos.x-=s.x-center.x;
							else
								rect.pos.x-=center.x;

							if (mirror_y->is_pressed())
								rect.pos.y-=s.y-center.y;
							else
								rect.pos.y-=center.y;
						}

						rect.pos=xform.xform(rect.pos);
						rect.size*=sc;

						if (r==Rect2()) {

							canvas_item_editor->draw_texture_rect(t,rect,false,Color(1,1,1,0.5),transpose->is_pressed());
						} else {

							canvas_item_editor->draw_texture_rect_region(t,rect,r,Color(1,1,1,0.5),transpose->is_pressed());
						}
					}
				}
			}

		}
	}



}



void TileMapEditor::edit(Node *p_tile_map) {

	if (!canvas_item_editor) {
		canvas_item_editor=CanvasItemEditor::get_singleton()->get_viewport_control();
	}

	if (node)
		node->disconnect("settings_changed",this,"_tileset_settings_changed");
	if (p_tile_map) {

		node=p_tile_map->cast_to<TileMap>();
		if (!canvas_item_editor->is_connected("draw",this,"_canvas_draw"))
			canvas_item_editor->connect("draw",this,"_canvas_draw");
		if (!canvas_item_editor->is_connected("mouse_enter",this,"_canvas_mouse_enter"))
			canvas_item_editor->connect("mouse_enter",this,"_canvas_mouse_enter");
		if (!canvas_item_editor->is_connected("mouse_exit",this,"_canvas_mouse_exit"))
			canvas_item_editor->connect("mouse_exit",this,"_canvas_mouse_exit");

		_update_palette();

	} else {
		node=NULL;

		if (canvas_item_editor->is_connected("draw",this,"_canvas_draw"))
			canvas_item_editor->disconnect("draw",this,"_canvas_draw");
		if (canvas_item_editor->is_connected("mouse_enter",this,"_canvas_mouse_enter"))
			canvas_item_editor->disconnect("mouse_enter",this,"_canvas_mouse_enter");
		if (canvas_item_editor->is_connected("mouse_exit",this,"_canvas_mouse_exit"))
			canvas_item_editor->disconnect("mouse_exit",this,"_canvas_mouse_exit");

		_update_palette();
	}

	if (node)
		node->connect("settings_changed",this,"_tileset_settings_changed");

}

void TileMapEditor::_tileset_settings_changed() {

	_update_palette();
	if (canvas_item_editor)
		canvas_item_editor->update();
}

void TileMapEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_menu_option"),&TileMapEditor::_menu_option);
	ObjectTypeDB::bind_method(_MD("_canvas_draw"),&TileMapEditor::_canvas_draw);
	ObjectTypeDB::bind_method(_MD("_canvas_mouse_enter"),&TileMapEditor::_canvas_mouse_enter);
	ObjectTypeDB::bind_method(_MD("_canvas_mouse_exit"),&TileMapEditor::_canvas_mouse_exit);
	ObjectTypeDB::bind_method(_MD("_tileset_settings_changed"),&TileMapEditor::_tileset_settings_changed);
	ObjectTypeDB::bind_method(_MD("_update_transform_buttons"),&TileMapEditor::_update_transform_buttons);
	ObjectTypeDB::bind_method(_MD("_set_cell_shortened","pos","tile","flip_x","flip_y","transpose"),&TileMapEditor::_set_cell_shortened,DEFVAL(false),DEFVAL(false),DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("_set_display_mode","mode"),&TileMapEditor::_set_display_mode);
}

TileMapEditor::CellOp TileMapEditor::_get_op_from_cell(const Point2i& p_pos)
{
	CellOp op;
	op.idx = node->get_cell(p_pos.x,p_pos.y);
	if (op.idx>=0) {
		if (node->is_cell_x_flipped(p_pos.x,p_pos.y))
			op.xf=true;
		if (node->is_cell_y_flipped(p_pos.x,p_pos.y))
			op.yf=true;
		if (node->is_cell_transposed(p_pos.x,p_pos.y))
			op.tr=true;
	}
	return op;
}

void TileMapEditor::_update_transform_buttons(Object *p_button) {
	//ERR_FAIL_NULL(p_button);
	ToolButton *b=p_button->cast_to<ToolButton>();
	//ERR_FAIL_COND(!b);

	mirror_x->set_block_signals(true);
	mirror_y->set_block_signals(true);
	transpose->set_block_signals(true);
	rotate_0->set_block_signals(true);
	rotate_90->set_block_signals(true);
	rotate_180->set_block_signals(true);
	rotate_270->set_block_signals(true);

	if (b == rotate_0) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(false);
		transpose->set_pressed(false);
	}
	else if (b == rotate_90) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(false);
		transpose->set_pressed(true);
	}
	else if (b == rotate_180) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(true);
		transpose->set_pressed(false);
	}
	else if (b == rotate_270) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(true);
		transpose->set_pressed(true);
	}

	rotate_0->set_pressed(!mirror_x->is_pressed() && !mirror_y->is_pressed() && !transpose->is_pressed());
	rotate_90->set_pressed(mirror_x->is_pressed() && !mirror_y->is_pressed() && transpose->is_pressed());
	rotate_180->set_pressed(mirror_x->is_pressed() && mirror_y->is_pressed() && !transpose->is_pressed());
	rotate_270->set_pressed(!mirror_x->is_pressed() && mirror_y->is_pressed() && transpose->is_pressed());

	mirror_x->set_block_signals(false);
	mirror_y->set_block_signals(false);
	transpose->set_block_signals(false);
	rotate_0->set_block_signals(false);
	rotate_90->set_block_signals(false);
	rotate_180->set_block_signals(false);
	rotate_270->set_block_signals(false);
}

TileMapEditor::TileMapEditor(EditorNode *p_editor) {

	node=NULL;
	canvas_item_editor=NULL;
	editor=p_editor;
	undo_redo = editor->get_undo_redo();

	int mw = EDITOR_DEF("tile_map/palette_min_width",80);
	Control *ec = memnew( Control);
	ec->set_custom_minimum_size(Size2(mw,0));
	add_child(ec);

	HBoxContainer *hb = memnew( HBoxContainer );
	add_child(hb);
	hb->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_spacer(true);

	button_thumbnail = memnew( ToolButton );
	button_thumbnail->set_toggle_mode(true);
	button_thumbnail->set_pressed(true);
	button_thumbnail->set_icon(p_editor->get_gui_base()->get_icon("FileThumbnail","EditorIcons"));
	hb->add_child(button_thumbnail);
	button_thumbnail->connect("pressed", this, "_set_display_mode", varray(DISPLAY_THUMBNAIL));

	button_list = memnew( ToolButton );
	button_list->set_toggle_mode(true);
	button_list->set_pressed(false);
	button_list->set_icon(p_editor->get_gui_base()->get_icon("FileList","EditorIcons"));
	hb->add_child(button_list);
	button_list->connect("pressed", this, "_set_display_mode", varray(DISPLAY_LIST));

	// Add tile palette
	palette = memnew( ItemList );
	palette->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(palette);

	// Add menu items
	canvas_item_editor_hb = memnew( HBoxContainer );
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(canvas_item_editor_hb);
	canvas_item_editor_hb->add_child( memnew( VSeparator ));
	transpose = memnew( ToolButton );
	transpose->set_toggle_mode(true);
	transpose->set_tooltip("Transpose");
	transpose->set_focus_mode(FOCUS_NONE);
	transpose->connect("pressed", this, "_update_transform_buttons", make_binds(transpose));
	canvas_item_editor_hb->add_child(transpose);
	mirror_x = memnew( ToolButton );
	mirror_x->set_toggle_mode(true);
	mirror_x->set_tooltip("Mirror X (A)");
	mirror_x->set_focus_mode(FOCUS_NONE);
	mirror_x->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_x));
	canvas_item_editor_hb->add_child(mirror_x);
	mirror_y = memnew( ToolButton );
	mirror_y->set_toggle_mode(true);
	mirror_y->set_tooltip("Mirror Y (S)");
	mirror_y->set_focus_mode(FOCUS_NONE);
	mirror_y->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_y));
	canvas_item_editor_hb->add_child(mirror_y);
	canvas_item_editor_hb->add_child(memnew(VSeparator));
	rotate_0 = memnew( ToolButton );
	rotate_0->set_toggle_mode(true);
	rotate_0->set_tooltip("Rotate 0 degrees");
	rotate_0->set_focus_mode(FOCUS_NONE);
	rotate_0->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_0));
	canvas_item_editor_hb->add_child(rotate_0);
	rotate_90 = memnew( ToolButton );
	rotate_90->set_toggle_mode(true);
	rotate_90->set_tooltip("Rotate 90 degrees");
	rotate_90->set_focus_mode(FOCUS_NONE);
	rotate_90->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_90));
	canvas_item_editor_hb->add_child(rotate_90);
	rotate_180 = memnew( ToolButton );
	rotate_180->set_toggle_mode(true);
	rotate_180->set_tooltip("Rotate 180 degrees");
	rotate_180->set_focus_mode(FOCUS_NONE);
	rotate_180->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_180));
	canvas_item_editor_hb->add_child(rotate_180);
	rotate_270 = memnew( ToolButton );
	rotate_270->set_toggle_mode(true);
	rotate_270->set_tooltip("Rotate 270 degrees");
	rotate_270->set_focus_mode(FOCUS_NONE);
	rotate_270->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_270));
	canvas_item_editor_hb->add_child(rotate_270);
	canvas_item_editor_hb->hide();

	rotate_0->set_pressed(true);
	tool=TOOL_NONE;
	selection_active=false;
	mouse_over=false;
}


void TileMapEditorPlugin::edit(Object *p_object) {

	tile_map_editor->edit(p_object->cast_to<Node>());
}

bool TileMapEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("TileMap");
}

void TileMapEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		tile_map_editor->show();
		tile_map_editor->get_canvas_item_editor_hb()->show();

	} else {

		tile_map_editor->hide();
		tile_map_editor->get_canvas_item_editor_hb()->hide();
		tile_map_editor->edit(NULL);
	}

}

TileMapEditorPlugin::TileMapEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	tile_map_editor = memnew( TileMapEditor(p_node) );
	CanvasItemEditor::get_singleton()->get_palette_split()->add_child(tile_map_editor);
	CanvasItemEditor::get_singleton()->get_palette_split()->move_child(tile_map_editor,0);

	tile_map_editor->hide();




}


TileMapEditorPlugin::~TileMapEditorPlugin()
{
}

