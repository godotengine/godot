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

			transp->set_icon( get_icon("Transpose","EditorIcons"));
			mirror_x->set_icon( get_icon("MirrorX","EditorIcons"));
			mirror_y->set_icon( get_icon("MirrorY","EditorIcons"));
			rotate_0->set_icon( get_icon("Rotate0","EditorIcons"));
			rotate_90->set_icon( get_icon("Rotate90","EditorIcons"));
			rotate_180->set_icon( get_icon("Rotate180","EditorIcons"));
			rotate_270->set_icon( get_icon("Rotate270","EditorIcons"));

		} break;
	}
}

void TileMapEditor::_menu_option(int p_option) {

	switch(p_option) {

		case OPTION_PICK_TILE: {

			tool=TOOL_PICKING;

			canvas_item_editor->update();
		} break;
		case OPTION_SELECT: {

			tool=TOOL_SELECTING;
			selection_active=false;

			canvas_item_editor->update();
		} break;
		case OPTION_DUPLICATE: {

			_update_copydata();

			if (selection_active) {
				tool=TOOL_DUPLICATING;

				canvas_item_editor->update();
			}
		}
		case OPTION_ERASE_SELECTION: {

			if (!selection_active)
				return;

			undo_redo->create_action("Erase Selection");
			for(int i=rectangle.pos.y;i<=rectangle.pos.y+rectangle.size.y;i++) {
				for(int j=rectangle.pos.x;j<=rectangle.pos.x+rectangle.size.x;j++) {

					_set_cell(Point2i(j, i), TileMap::INVALID_CELL, false, false, false, true);
				}
			}
			undo_redo->commit_action();

			selection_active=false;
			copydata.clear();

			canvas_item_editor->update();
		}
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

	int idx = palette->find_metadata(p_tile);

	if (idx >= 0) {
		palette->select(idx, true);
		palette->ensure_current_is_visible();
	}
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

void TileMapEditor::_text_entered(const String& p_text) {

	canvas_item_editor->grab_focus();
}

void TileMapEditor::_text_changed(const String& p_text) {

	_update_palette();
}

void TileMapEditor::_sbox_input(const InputEvent& p_ie) {

	if (p_ie.type==InputEvent::KEY && (
		p_ie.key.scancode == KEY_UP ||
		p_ie.key.scancode == KEY_DOWN ||
		p_ie.key.scancode == KEY_PAGEUP ||
		p_ie.key.scancode == KEY_PAGEDOWN ) ) {

		palette->call("_input_event", p_ie);
		search_box->accept_event();
	}
}

void TileMapEditor::_update_palette() {

	if (!node)
		return;

	int selected = get_selected_tile();
	palette->clear();

	Ref<TileSet> tileset=node->get_tileset();
	if (tileset.is_null())
		return;

	List<int> tiles;
	tileset->get_tile_list(&tiles);

	if (tiles.empty())
		return;

	palette->set_max_columns(0);
	palette->set_icon_mode(ItemList::ICON_MODE_TOP);
	palette->set_max_text_lines(2);

	String filter = search_box->get_text().strip_edges();

	for(List<int>::Element *E=tiles.front();E;E=E->next()) {

		String name;

		if (tileset->tile_get_name(E->get())!="") {
			name = tileset->tile_get_name(E->get());
		} else {
			name = "#"+itos(E->get());
		}

		if (filter != "" && name.findn(filter) == -1)
			continue;

		palette->add_item(name);

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

		palette->set_item_metadata(palette->get_item_count()-1, E->get());
	}

	if (selected != -1)
		set_selected_tile(selected);
	else
		palette->select(0, true);
}

void TileMapEditor::_pick_tile(const Point2& p_pos) {

	int id = node->get_cell(p_pos.x, p_pos.y);

	if (id==TileMap::INVALID_CELL)
		return;

	if (search_box->get_text().strip_edges() != "") {

		search_box->set_text("");
		_update_palette();
	}

	set_selected_tile(id);

	mirror_x->set_pressed(node->is_cell_x_flipped(p_pos.x, p_pos.y));
	mirror_y->set_pressed(node->is_cell_y_flipped(p_pos.x, p_pos.y));
	transp->set_pressed(node->is_cell_transposed(p_pos.x, p_pos.y));

	_update_transform_buttons();
	canvas_item_editor->update();
}

void TileMapEditor::_select(const Point2i& p_from, const Point2i& p_to) {

	Point2i begin=p_from;
	Point2i end=p_to;

	if (begin.x > end.x) {

		SWAP( begin.x, end.x);
	}
	if (begin.y > end.y) {

		SWAP( begin.y, end.y);
	}

	rectangle.pos=begin;
	rectangle.size=end-begin;

	canvas_item_editor->update();
}

void TileMapEditor::_draw_cell(int p_cell, const Point2i& p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Matrix32& p_xform) {

	Ref<Texture> t = node->get_tileset()->tile_get_texture(p_cell);

	if (t.is_null())
		return;

	Vector2 from = node->map_to_world(p_point)+node->get_cell_draw_offset();
	Vector2 tile_ofs = node->get_tileset()->tile_get_texture_offset(p_cell);

	Rect2 r = node->get_tileset()->tile_get_region(p_cell);
	Size2 sc = p_xform.get_scale();

	Rect2 rect;

	if (r==Rect2()) {
		rect=Rect2(from,t->get_size());
	} else {
		rect=Rect2(from,r.get_size());
	}

	if (rect.size.y > rect.size.x) {
		if ((p_flip_h && (p_flip_v || p_transpose)) || (p_flip_v && !p_transpose))
			tile_ofs.y += rect.size.y - rect.size.x;
	} else if (rect.size.y < rect.size.x) {
		if ((p_flip_v && (p_flip_h || p_transpose)) || (p_flip_h && !p_transpose))
			tile_ofs.x += rect.size.x - rect.size.y;
	}

	if (p_transpose) {
		SWAP(tile_ofs.x, tile_ofs.y);
	}
	if (p_flip_h) {
		sc.x*=-1.0;
		tile_ofs.x*=-1.0;
	}
	if (p_flip_v) {
		sc.y*=-1.0;
		tile_ofs.y*=-1.0;
	}

	if (node->get_tile_origin()==TileMap::TILE_ORIGIN_TOP_LEFT) {
		rect.pos+=tile_ofs;

	} else if (node->get_tile_origin()==TileMap::TILE_ORIGIN_CENTER) {
		rect.pos+=node->get_cell_size()/2;
		Vector2 s = r.size;

		Vector2 center = (s/2) - tile_ofs;


		if (p_flip_h)
			rect.pos.x-=s.x-center.x;
		else
			rect.pos.x-=center.x;

		if (p_flip_v)
			rect.pos.y-=s.y-center.y;
		else
			rect.pos.y-=center.y;
	}

	rect.pos=p_xform.xform(rect.pos);
	rect.size*=sc;

	if (r==Rect2())
		canvas_item_editor->draw_texture_rect(t,rect,false,Color(1,1,1,0.5),p_transpose);
	else
		canvas_item_editor->draw_texture_rect_region(t,rect,r,Color(1,1,1,0.5),p_transpose);
}

void TileMapEditor::_update_copydata() {

	copydata.clear();

	if (!selection_active)
		return;

	for(int i=rectangle.pos.y;i<=rectangle.pos.y+rectangle.size.y;i++) {

		for(int j=rectangle.pos.x;j<=rectangle.pos.x+rectangle.size.x;j++) {

			TileData tcd;

			tcd.cell=node->get_cell(j, i);

			if (tcd.cell!=TileMap::INVALID_CELL) {
				tcd.pos=Point2i(j, i);
				tcd.flip_h=node->is_cell_x_flipped(j,i);
				tcd.flip_v=node->is_cell_y_flipped(j,i);
				tcd.transpose=node->is_cell_transposed(j,i);
			}

			copydata.push_back(tcd);
		}
	}
}

bool TileMapEditor::forward_input_event(const InputEvent& p_event) {

	if (!node || !node->get_tileset().is_valid() || !node->is_visible())
		return false;

	Matrix32 xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Matrix32 xform_inv = xform.affine_inverse();

	switch(p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &mb=p_event.mouse_button;

			if (mb.button_index==BUTTON_LEFT) {

				if (mb.pressed) {

					if (Input::get_singleton()->is_key_pressed(KEY_SPACE))
						return false; //drag

					if (tool==TOOL_NONE) {
						if (mb.mod.shift) {

							tool=TOOL_RECTANGLE_PAINT;
							selection_active=false;
							rectangle_begin=node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));

							return true;
						}
						if (mb.mod.control) {

							tool=TOOL_PICKING;

							_pick_tile(over_tile);

							return true;
						}

						tool=TOOL_PAINTING;
					}

					if (tool==TOOL_PAINTING) {
						int id = get_selected_tile();

						if (id!=TileMap::INVALID_CELL) {

							tool=TOOL_PAINTING;

							paint_undo.clear();
							paint_undo[over_tile]=_get_op_from_cell(over_tile);

							_set_cell(over_tile, id, flip_h, flip_v, transpose);

							return true;
						}
					}

					if (tool==TOOL_PICKING) {

						_pick_tile(over_tile);

						return true;
					}

					if (tool==TOOL_SELECTING) {

						selection_active=true;
						rectangle_begin=node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));

						return true;
					}

					if (tool==TOOL_DUPLICATING) {

						Point2 ofs = over_tile-rectangle.pos;

						undo_redo->create_action("Duplicate");
						for (List<TileData>::Element *E=copydata.front();E;E=E->next()) {

							_set_cell(E->get().pos+ofs,E->get().cell,E->get().flip_h,E->get().flip_v,E->get().transpose,true);
						}
						undo_redo->commit_action();

						tool=TOOL_NONE;
						copydata.clear();

						canvas_item_editor->update();

						return true;
					}
				} else {

					if (tool!=TOOL_NONE) {

						if (tool==TOOL_PAINTING) {

							int id=get_selected_tile();

							if (id!=TileMap::INVALID_CELL && paint_undo.size()) {
								undo_redo->create_action("Paint TileMap");
								for(Map<Point2i,CellOp>::Element *E=paint_undo.front();E;E=E->next()) {

									Point2i p=E->key();
									undo_redo->add_do_method(node,"set_cellv",Point2(p),id,node->is_cell_x_flipped(p.x,p.y),node->is_cell_y_flipped(p.x,p.y),node->is_cell_transposed(p.x,p.y));
									undo_redo->add_undo_method(node,"set_cellv",Point2(p),E->get().idx,E->get().xf,E->get().yf,E->get().tr);
								}

								undo_redo->commit_action();
								paint_undo.clear();
							}
						} else if (tool==TOOL_RECTANGLE_PAINT) {

							int id=get_selected_tile();

							if (id!=TileMap::INVALID_CELL) {

								undo_redo->create_action("Rectangle Paint");
								for(int i=rectangle.pos.y;i<=rectangle.pos.y+rectangle.size.y;i++) {
									for(int j=rectangle.pos.x;j<=rectangle.pos.x+rectangle.size.x;j++) {

										_set_cell(Point2(j, i), id, flip_h, flip_v, transpose, true);
									}
								}
								undo_redo->commit_action();

								canvas_item_editor->update();
							}
						} else if (tool==TOOL_SELECTING) {

							canvas_item_editor->update();
						}

						tool=TOOL_NONE;

						return true;
					}
				}
			} else if (mb.button_index==BUTTON_RIGHT) {

				if (mb.pressed) {

					if (tool==TOOL_SELECTING) {

						tool=TOOL_NONE;
						selection_active=false;

						canvas_item_editor->update();

						return true;
					}

					if (tool==TOOL_DUPLICATING) {

						tool=TOOL_NONE;
						copydata.clear();

						canvas_item_editor->update();

						return true;
					}

					if (tool==TOOL_NONE) {

						if (mb.mod.shift) {

							tool=TOOL_RECTANGLE_ERASE;

							selection_active=false;
							rectangle_begin=node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));
							paint_undo.clear();

						} else {
							tool=TOOL_ERASING;

							Point2i local=node->world_to_map(xform_inv.xform(Point2(mb.x,mb.y)));
							paint_undo.clear();
							paint_undo[local]=_get_op_from_cell(local);

							_set_cell(local, TileMap::INVALID_CELL);
						}

						return true;
					}

				} else {
					if (tool==TOOL_ERASING || tool==TOOL_RECTANGLE_ERASE) {

						if (paint_undo.size()) {
							undo_redo->create_action("Erase TileMap");
							for(Map<Point2i,CellOp>::Element *E=paint_undo.front();E;E=E->next()) {

								Point2i p=E->key();
								undo_redo->add_do_method(node,"set_cellv",Point2(p),TileMap::INVALID_CELL,false,false,false);
								undo_redo->add_undo_method(node,"set_cellv",Point2(p),E->get().idx,E->get().xf,E->get().yf,E->get().tr);
							}

							undo_redo->commit_action();
							paint_undo.clear();
						}

						if (tool==TOOL_RECTANGLE_ERASE) {
							canvas_item_editor->update();
						}
						tool=TOOL_NONE;

						return true;
					}
				}
			}
		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &mm=p_event.mouse_motion;

			Point2i new_over_tile = node->world_to_map(xform_inv.xform(Point2(mm.x,mm.y)));

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
					_set_cell(over_tile, id, flip_h, flip_v, transpose);

					return true;
				}
			}

			if (tool==TOOL_SELECTING) {

				_select(rectangle_begin, over_tile);

				return true;
			}
			if (tool==TOOL_RECTANGLE_PAINT || tool==TOOL_RECTANGLE_ERASE) {

				_select(rectangle_begin, over_tile);

				if (tool==TOOL_RECTANGLE_ERASE) {

					if (paint_undo.size()) {

						for (Map<Point2i, CellOp>::Element *E=paint_undo.front();E;E=E->next()) {

							_set_cell(E->key(), E->get().idx, E->get().xf, E->get().yf, E->get().tr);
						}

						paint_undo.clear();
					}

					for(int i=rectangle.pos.y;i<=rectangle.pos.y+rectangle.size.y;i++) {
						for(int j=rectangle.pos.x;j<=rectangle.pos.x+rectangle.size.x;j++) {

							Point2i tile = Point2i(j, i);

							if (!paint_undo.has(tile))
								paint_undo[tile]=_get_op_from_cell(tile);

							_set_cell(tile, TileMap::INVALID_CELL);
						}
					}
				}

				return true;
			}
			if (tool==TOOL_ERASING) {

				if (!paint_undo.has(over_tile)) {
					paint_undo[over_tile]=_get_op_from_cell(over_tile);
				}

				_set_cell(over_tile, TileMap::INVALID_CELL);

				return true;
			}
			if (tool==TOOL_PICKING && Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {

				_pick_tile(over_tile);

				return true;
			}
		} break;
		case InputEvent::KEY: {

			const InputEventKey &k = p_event.key;

			if (!k.pressed)
				break;

			if (k.scancode==KEY_ESCAPE) {

				if (tool==TOOL_DUPLICATING)
					copydata.clear();
				else if (tool==TOOL_SELECTING || selection_active)
					selection_active=false;

				tool=TOOL_NONE;

				canvas_item_editor->update();

				return true;
			}

			if (tool!=TOOL_NONE)
				return false;

			if (k.scancode==KEY_DELETE) {

				_menu_option(OPTION_ERASE_SELECTION);

				return true;
			}
			if (mouse_over && k.scancode==KEY_A && !k.mod.command) {

				flip_h=!flip_h;
				mirror_x->set_pressed(flip_h);
				canvas_item_editor->update();
				return true;
			}
			if (mouse_over && k.scancode==KEY_S && !k.mod.command) {

				flip_v=!flip_v;
				mirror_y->set_pressed(flip_v);
				canvas_item_editor->update();
				return true;
			}
			if (mouse_over && k.scancode==KEY_F && k.mod.command) {

				search_box->select_all();
				search_box->grab_focus();

				return true;
			}
			if (mouse_over && k.scancode==KEY_B && k.mod.command) {

				tool=TOOL_SELECTING;
				selection_active=false;

				canvas_item_editor->update();

				return true;
			}
			if (mouse_over && k.scancode==KEY_D && k.mod.command) {

				_update_copydata();

				if (selection_active) {
					tool=TOOL_DUPLICATING;

					canvas_item_editor->update();

					return true;
				}
			}
		} break;
	}

	return false;
}
void TileMapEditor::_canvas_draw() {

	if (!node)
		return;

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

			int max_lines=2000; //avoid crash if size too smal

			for(int i=(si.pos.x)-1;i<=(si.pos.x+si.size.x);i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(i,si.pos.y)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(i,si.pos.y+si.size.y+1)));

				Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
				canvas_item_editor->draw_line(from,to,col,1);
				if (max_lines--==0)
					break;
			}
		} else {

			int max_lines=10000; //avoid crash if size too smal

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

					if (max_lines--==0)
						break;

				}

			}
		}

		int max_lines=10000; //avoid crash if size too smal

		if (node->get_half_offset()!=TileMap::HALF_OFFSET_Y) {

			for(int i=(si.pos.y)-1;i<=(si.pos.y+si.size.y);i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(si.pos.x,i)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(si.pos.x+si.size.x+1,i)));

				Color col=i==0?Color(1,0.8,0.2,0.5):Color(1,0.3,0.1,0.2);
				canvas_item_editor->draw_line(from,to,col,1);

				if (max_lines--==0)
					break;

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

					if (max_lines--==0)
						break;

				}
			}
		}
	}

	if (selection_active) {

		Vector<Vector2> points;
		points.push_back( xform.xform( node->map_to_world(( rectangle.pos ) )));
		points.push_back( xform.xform( node->map_to_world((rectangle.pos+Point2(rectangle.size.x+1,0)) ) ));
		points.push_back( xform.xform( node->map_to_world((rectangle.pos+Point2(rectangle.size.x+1,rectangle.size.y+1)) ) ));
		points.push_back( xform.xform( node->map_to_world((rectangle.pos+Point2(0,rectangle.size.y+1)) ) ));

		canvas_item_editor->draw_colored_polygon(points, Color(0.2,0.8,1,0.4));
	}


	if (mouse_over){

		Vector2 endpoints[4]={
			node->map_to_world(over_tile, true),
			node->map_to_world((over_tile+Point2(1,0)), true),
			node->map_to_world((over_tile+Point2(1,1)), true),
			node->map_to_world((over_tile+Point2(0,1)), true)
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


		if (tool==TOOL_SELECTING || tool==TOOL_PICKING) {

			return;

		} else if (tool==TOOL_RECTANGLE_PAINT) {

			int id = get_selected_tile();

			if (id==TileMap::INVALID_CELL)
				return;

			for(int i=rectangle.pos.y;i<=rectangle.pos.y+rectangle.size.y;i++) {
				for(int j=rectangle.pos.x;j<=rectangle.pos.x+rectangle.size.x;j++) {

					_draw_cell(id, Point2(j, i), flip_h, flip_v, transpose, xform);
				}
			}
		} else if (tool==TOOL_DUPLICATING) {

			if (copydata.empty())
				return;

			Ref<TileSet> ts = node->get_tileset();

			if (ts.is_null())
				return;

			Point2 ofs = over_tile-rectangle.pos;

			for (List<TileData>::Element *E=copydata.front();E;E=E->next()) {

				if (!ts->has_tile(E->get().cell))
					continue;

				TileData tcd = E->get();

				_draw_cell(tcd.cell, tcd.pos+ofs, tcd.flip_h, tcd.flip_v, tcd.transpose, xform);
			}

			Rect2i duplicate=rectangle;
			duplicate.pos=over_tile;

			Vector<Vector2> points;
			points.push_back( xform.xform( node->map_to_world(duplicate.pos ) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(duplicate.size.x+1,0)) ) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(duplicate.size.x+1,duplicate.size.y+1))) ));
			points.push_back( xform.xform( node->map_to_world((duplicate.pos+Point2(0,duplicate.size.y+1))) ));

			canvas_item_editor->draw_colored_polygon(points, Color(0.2,1.0,0.8,0.2));

		} else {

			int st = get_selected_tile();

			if (st==TileMap::INVALID_CELL)
				return;

			_draw_cell(st, over_tile, flip_h, flip_v, transpose, xform);
		}
	}

}



void TileMapEditor::edit(Node *p_tile_map) {

	search_box->set_text("");

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

	ObjectTypeDB::bind_method(_MD("_text_entered"),&TileMapEditor::_text_entered);
	ObjectTypeDB::bind_method(_MD("_text_changed"),&TileMapEditor::_text_changed);
	ObjectTypeDB::bind_method(_MD("_sbox_input"),&TileMapEditor::_sbox_input);
	ObjectTypeDB::bind_method(_MD("_menu_option"),&TileMapEditor::_menu_option);
	ObjectTypeDB::bind_method(_MD("_canvas_draw"),&TileMapEditor::_canvas_draw);
	ObjectTypeDB::bind_method(_MD("_canvas_mouse_enter"),&TileMapEditor::_canvas_mouse_enter);
	ObjectTypeDB::bind_method(_MD("_canvas_mouse_exit"),&TileMapEditor::_canvas_mouse_exit);
	ObjectTypeDB::bind_method(_MD("_tileset_settings_changed"),&TileMapEditor::_tileset_settings_changed);
	ObjectTypeDB::bind_method(_MD("_update_transform_buttons"),&TileMapEditor::_update_transform_buttons);
}

TileMapEditor::CellOp TileMapEditor::_get_op_from_cell(const Point2i& p_pos)
{
	CellOp op;
	op.idx = node->get_cell(p_pos.x,p_pos.y);
	if (op.idx!=TileMap::INVALID_CELL) {
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

	if (b == rotate_0) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(false);
		transp->set_pressed(false);
	}
	else if (b == rotate_90) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(false);
		transp->set_pressed(true);
	}
	else if (b == rotate_180) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(true);
		transp->set_pressed(false);
	}
	else if (b == rotate_270) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(true);
		transp->set_pressed(true);
	}

	flip_h=mirror_x->is_pressed();
	flip_v=mirror_y->is_pressed();
	transpose=transp->is_pressed();

	rotate_0->set_pressed(!flip_h && !flip_v && !transpose);
	rotate_90->set_pressed(flip_h && !flip_v && transpose);
	rotate_180->set_pressed(flip_h && flip_v && !transpose);
	rotate_270->set_pressed(!flip_h && flip_v && transpose);
}

TileMapEditor::TileMapEditor(EditorNode *p_editor) {

	node=NULL;
	canvas_item_editor=NULL;
	editor=p_editor;
	undo_redo=editor->get_undo_redo();

	tool=TOOL_NONE;
	selection_active=false;
	mouse_over=false;

	flip_h=false;
	flip_v=false;
	transpose=false;

	search_box = memnew( LineEdit );
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_box->connect("text_entered", this, "_text_entered");
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("input_event", this, "_sbox_input");
	add_child(search_box);

	int mw = EDITOR_DEF("tile_map/palette_min_width", 80);

	// Add tile palette
	palette = memnew( ItemList );
	palette->set_v_size_flags(SIZE_EXPAND_FILL);
	palette->set_custom_minimum_size(Size2(mw,0));
	add_child(palette);

	// Add menu items
	toolbar = memnew( HBoxContainer );
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(toolbar);

	options = memnew( MenuButton );
	options->set_text("Tile Map");
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("TileMap", "EditorIcons"));

	PopupMenu *p = options->get_popup();

	p->add_item("Pick Tile", OPTION_PICK_TILE);
	p->add_separator();
	p->add_item("Select", OPTION_SELECT);
	p->add_item("Duplicate Selection", OPTION_DUPLICATE);
	p->add_separator();
	p->add_item("Erase Selection", OPTION_ERASE_SELECTION);

	p->connect("item_pressed", this, "_menu_option");

	toolbar->add_child(options);

	toolbar->add_child( memnew( VSeparator ) );

	transp = memnew( ToolButton );
	transp->set_toggle_mode(true);
	transp->set_tooltip("Transpose");
	transp->set_focus_mode(FOCUS_NONE);
	transp->connect("pressed", this, "_update_transform_buttons", make_binds(transp));
	toolbar->add_child(transp);
	mirror_x = memnew( ToolButton );
	mirror_x->set_toggle_mode(true);
	mirror_x->set_tooltip("Mirror X (A)");
	mirror_x->set_focus_mode(FOCUS_NONE);
	mirror_x->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_x));
	toolbar->add_child(mirror_x);
	mirror_y = memnew( ToolButton );
	mirror_y->set_toggle_mode(true);
	mirror_y->set_tooltip("Mirror Y (S)");
	mirror_y->set_focus_mode(FOCUS_NONE);
	mirror_y->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_y));
	toolbar->add_child(mirror_y);
	toolbar->add_child( memnew( VSeparator ) );
	rotate_0 = memnew( ToolButton );
	rotate_0->set_toggle_mode(true);
	rotate_0->set_tooltip("Rotate 0 degrees");
	rotate_0->set_focus_mode(FOCUS_NONE);
	rotate_0->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_0));
	toolbar->add_child(rotate_0);
	rotate_90 = memnew( ToolButton );
	rotate_90->set_toggle_mode(true);
	rotate_90->set_tooltip("Rotate 90 degrees");
	rotate_90->set_focus_mode(FOCUS_NONE);
	rotate_90->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_90));
	toolbar->add_child(rotate_90);
	rotate_180 = memnew( ToolButton );
	rotate_180->set_toggle_mode(true);
	rotate_180->set_tooltip("Rotate 180 degrees");
	rotate_180->set_focus_mode(FOCUS_NONE);
	rotate_180->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_180));
	toolbar->add_child(rotate_180);
	rotate_270 = memnew( ToolButton );
	rotate_270->set_toggle_mode(true);
	rotate_270->set_tooltip("Rotate 270 degrees");
	rotate_270->set_focus_mode(FOCUS_NONE);
	rotate_270->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_270));
	toolbar->add_child(rotate_270);
	toolbar->hide();

	rotate_0->set_pressed(true);
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void TileMapEditorPlugin::edit(Object *p_object) {

	tile_map_editor->edit(p_object->cast_to<Node>());
}

bool TileMapEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("TileMap");
}

void TileMapEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		tile_map_editor->show();
		tile_map_editor->get_toolbar()->show();
	} else {

		tile_map_editor->hide();
		tile_map_editor->get_toolbar()->hide();
		tile_map_editor->edit(NULL);
	}
}

TileMapEditorPlugin::TileMapEditorPlugin(EditorNode *p_node) {

	tile_map_editor = memnew( TileMapEditor(p_node) );
	add_control_to_container(CONTAINER_CANVAS_EDITOR_SIDE, tile_map_editor);
	tile_map_editor->hide();
}

TileMapEditorPlugin::~TileMapEditorPlugin()
{
}

