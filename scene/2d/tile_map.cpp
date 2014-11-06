/*************************************************************************/
/*  tile_map.cpp                                                         */
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
#include "tile_map.h"
#include "io/marshalls.h"
#include "servers/physics_2d_server.h"
void TileMap::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			pending_update=true;
			_update_dirty_quadrants();
			RID space = get_world_2d()->get_space();
			_update_quadrant_transform();
			_update_quadrant_space(space);


		} break;
		case NOTIFICATION_EXIT_TREE: {

			_update_quadrant_space(RID());

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			//move stuff
			_update_quadrant_transform();

		} break;
	}
}

void TileMap::_update_quadrant_space(const RID& p_space) {

	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

		Quadrant &q=E->get();
		Physics2DServer::get_singleton()->body_set_space(q.static_body,p_space);
	}
}

void TileMap::_update_quadrant_transform() {

	if (!is_inside_tree())
		return;

	Matrix32 global_transform = get_global_transform();

	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

		Quadrant &q=E->get();
		Matrix32 xform;
		xform.set_origin( q.pos );
		xform = global_transform * xform;
		Physics2DServer::get_singleton()->body_set_state(q.static_body,Physics2DServer::BODY_STATE_TRANSFORM,xform);
	}
}

void TileMap::set_tileset(const Ref<TileSet>& p_tileset) {

	if (tile_set.is_valid())
		tile_set->disconnect("changed",this,"_recreate_quadrants");

	_clear_quadrants();
	tile_set=p_tileset;

	if (tile_set.is_valid())
		tile_set->connect("changed",this,"_recreate_quadrants");
	else
		clear();

	_recreate_quadrants();
	emit_signal("settings_changed");

}

Ref<TileSet> TileMap::get_tileset() const {

	return tile_set;
}

void TileMap::set_cell_size(Size2 p_size) {

	ERR_FAIL_COND(p_size.x<1 || p_size.y<1);

	_clear_quadrants();
	cell_size=p_size;
	_recreate_quadrants();
	emit_signal("settings_changed");


}
Size2 TileMap::get_cell_size() const {

	return cell_size;
}
void TileMap::set_quadrant_size(int p_size) {

	ERR_FAIL_COND(p_size<1);

	_clear_quadrants();
	quadrant_size=p_size;
	_recreate_quadrants();
	emit_signal("settings_changed");

}
int TileMap::get_quadrant_size() const {

	return quadrant_size;
}

void TileMap::set_center_x(bool p_enable) {

	center_x=p_enable;
	_recreate_quadrants();
	emit_signal("settings_changed");


}
bool TileMap::get_center_x() const {

	return center_x;
}
void TileMap::set_center_y(bool p_enable) {

	center_y=p_enable;
	_recreate_quadrants();
	emit_signal("settings_changed");

}
bool TileMap::get_center_y() const {

	return center_y;
}

void TileMap::_update_dirty_quadrants() {

	if (!pending_update)
		return;
	if (!is_inside_tree())
		return;
	if (!tile_set.is_valid())
		return;

	VisualServer *vs = VisualServer::get_singleton();
	Physics2DServer *ps = Physics2DServer::get_singleton();
	Vector2 tofs = get_cell_draw_offset();

	while (dirty_quadrant_list.first()) {

		Quadrant &q = *dirty_quadrant_list.first()->self();

		vs->canvas_item_clear(q.canvas_item);
		ps->body_clear_shapes(q.static_body);
		int shape_idx=0;

		for(int i=0;i<q.cells.size();i++) {

			Map<PosKey,Cell>::Element *E=tile_map.find( q.cells[i] );
			Cell &c=E->get();
			//moment of truth
			if (!tile_set->has_tile(c.id))
				continue;
			Ref<Texture> tex = tile_set->tile_get_texture(c.id);
			Vector2 tile_ofs = tile_set->tile_get_texture_offset(c.id);

			Vector2 offset = _map_to_world(E->key().x, E->key().y) - q.pos + tofs;

			if (!tex.is_valid())
				continue;


			Rect2 r = tile_set->tile_get_region(c.id);
			Size2 s = tex->get_size();

			if (r==Rect2())
				s = tex->get_size();
			else {
				s = r.size;
				r.pos.x+=fp_adjust;
				r.pos.y+=fp_adjust;
				r.size.x-=fp_adjust*2.0;
				r.size.y-=fp_adjust*2.0;
			}

			Rect2 rect;
			rect.pos=offset.floor();
			rect.size=s;

			rect.size.x+=fp_adjust;
			rect.size.y+=fp_adjust;

			if (c.flip_h)
				rect.size.x=-rect.size.x;
			if (c.flip_v)
				rect.size.y=-rect.size.y;


			rect.pos+=tile_ofs;
			if (r==Rect2()) {

				tex->draw_rect(q.canvas_item,rect);
			} else {

				tex->draw_rect_region(q.canvas_item,rect,r);
			}

			Vector< Ref<Shape2D> > shapes = tile_set->tile_get_shapes(c.id);


			for(int i=0;i<shapes.size();i++) {

				Ref<Shape2D> shape = shapes[i];
				if (shape.is_valid()) {

					Vector2 shape_ofs = tile_set->tile_get_shape_offset(c.id);
					Matrix32 xform;
					xform.set_origin(offset.floor());
					if (c.flip_h) {
						xform.elements[0]=-xform.elements[0];
						xform.elements[2].x+=s.x-shape_ofs.x;
					} else {

						xform.elements[2].x+=shape_ofs.x;
					}
					if (c.flip_v) {
						xform.elements[1]=-xform.elements[1];
						xform.elements[2].y+=s.y-shape_ofs.y;
					} else {

						xform.elements[2].y+=shape_ofs.y;
					}


					ps->body_add_shape(q.static_body,shape->get_rid(),xform);
					ps->body_set_shape_metadata(q.static_body,shape_idx++,Vector2(E->key().x,E->key().y));

				}
			}
		}

		dirty_quadrant_list.remove( dirty_quadrant_list.first() );
	}



	pending_update=false;

	if (quadrant_order_dirty) {

		for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

			Quadrant &q=E->get();
			if (q.canvas_item.is_valid()) {
				VS::get_singleton()->canvas_item_raise(q.canvas_item);
			}

		}

		quadrant_order_dirty=false;
	}

	_recompute_rect_cache();

}

void TileMap::_recompute_rect_cache() {


#ifdef DEBUG_ENABLED

	if (!rect_cache_dirty)
		return;

	Rect2 r_total;
	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {


		Rect2 r;
		r.pos=_map_to_world(E->key().x*quadrant_size, E->key().y*quadrant_size);
		r.expand_to( _map_to_world(E->key().x*quadrant_size+quadrant_size, E->key().y*quadrant_size) );
		r.expand_to( _map_to_world(E->key().x*quadrant_size+quadrant_size, E->key().y*quadrant_size+quadrant_size) );
		r.expand_to( _map_to_world(E->key().x*quadrant_size, E->key().y*quadrant_size+quadrant_size) );
		if (E==quadrant_map.front())
			r_total=r;
		else
			r_total=r_total.merge(r);

	}

	if (r_total==Rect2()) {
		rect_cache=Rect2(-10,-10,20,20);
	} else {
		rect_cache=r_total.grow(MAX(cell_size.x,cell_size.y)*quadrant_size);
	}

	item_rect_changed();

	rect_cache_dirty=false;
#endif


}

Map<TileMap::PosKey,TileMap::Quadrant>::Element *TileMap::_create_quadrant(const PosKey& p_qk) {

	Matrix32 xform;
	//xform.set_origin(Point2(p_qk.x,p_qk.y)*cell_size*quadrant_size);
	Quadrant q;
	q.pos = _map_to_world(p_qk.x*quadrant_size,p_qk.y*quadrant_size);
	xform.set_origin( q.pos );
	q.canvas_item = VisualServer::get_singleton()->canvas_item_create();
	VisualServer::get_singleton()->canvas_item_set_parent( q.canvas_item, get_canvas_item() );
	VisualServer::get_singleton()->canvas_item_set_transform( q.canvas_item, xform );
	q.static_body=Physics2DServer::get_singleton()->body_create(Physics2DServer::BODY_MODE_STATIC);
	Physics2DServer::get_singleton()->body_attach_object_instance_ID(q.static_body,get_instance_ID());
	Physics2DServer::get_singleton()->body_set_layer_mask(q.static_body,collision_layer);
	Physics2DServer::get_singleton()->body_set_param(q.static_body,Physics2DServer::BODY_PARAM_FRICTION,friction);
	Physics2DServer::get_singleton()->body_set_param(q.static_body,Physics2DServer::BODY_PARAM_BOUNCE,bounce);

	if (is_inside_tree()) {
		xform = get_global_transform() * xform;
		RID space = get_world_2d()->get_space();
		Physics2DServer::get_singleton()->body_set_space(q.static_body,space);
	}

	Physics2DServer::get_singleton()->body_set_state(q.static_body,Physics2DServer::BODY_STATE_TRANSFORM,xform);

	rect_cache_dirty=true;
	quadrant_order_dirty=true;
	return quadrant_map.insert(p_qk,q);
}

void TileMap::_erase_quadrant(Map<PosKey,Quadrant>::Element *Q) {

	Quadrant &q=Q->get();
	Physics2DServer::get_singleton()->free(q.static_body);
	VisualServer::get_singleton()->free(q.canvas_item);
	if (q.dirty_list.in_list())
		dirty_quadrant_list.remove(&q.dirty_list);

	quadrant_map.erase(Q);
	rect_cache_dirty=true;
}

void TileMap::_make_quadrant_dirty(Map<PosKey,Quadrant>::Element *Q) {

	Quadrant &q=Q->get();
	if (!q.dirty_list.in_list())
		dirty_quadrant_list.add(&q.dirty_list);

	if (pending_update)
		return;
	pending_update=true;
	if (!is_inside_tree())
		return;
	call_deferred("_update_dirty_quadrants");
}


void TileMap::set_cell(int p_x,int p_y,int p_tile,bool p_flip_x,bool p_flip_y) {

	PosKey pk(p_x,p_y);

	Map<PosKey,Cell>::Element *E=tile_map.find(pk);
	if (!E && p_tile==INVALID_CELL)
		return; //nothing to do

	PosKey qk(p_x/quadrant_size,p_y/quadrant_size);
	if (p_tile==INVALID_CELL) {
		//erase existing
		tile_map.erase(pk);
		Map<PosKey,Quadrant>::Element *Q = quadrant_map.find(qk);
		ERR_FAIL_COND(!Q);
		Quadrant &q=Q->get();
		q.cells.erase(pk);
		if (q.cells.size()==0)
			_erase_quadrant(Q);
		else
			_make_quadrant_dirty(Q);

		return;
	}

	Map<PosKey,Quadrant>::Element *Q = quadrant_map.find(qk);

	if (!E) {
		E=tile_map.insert(pk,Cell());
		if (!Q) {
			Q=_create_quadrant(qk);
		}
		Quadrant &q=Q->get();
		q.cells.insert(pk);
	} else {
		ERR_FAIL_COND(!Q); // quadrant should exist...

		if (E->get().id==p_tile && E->get().flip_h==p_flip_x && E->get().flip_v==p_flip_y)
			return; //nothing changed

	}


	Cell &c = E->get();

	c.id=p_tile;
	c.flip_h=p_flip_x;
	c.flip_v=p_flip_y;

	_make_quadrant_dirty(Q);

}

int TileMap::get_cell(int p_x,int p_y) const {

	PosKey pk(p_x,p_y);

	const Map<PosKey,Cell>::Element *E=tile_map.find(pk);

	if (!E)
		return INVALID_CELL;

	return E->get().id;

}
bool TileMap::is_cell_x_flipped(int p_x,int p_y) const {

	PosKey pk(p_x,p_y);

	const Map<PosKey,Cell>::Element *E=tile_map.find(pk);

	if (!E)
		return false;

	return E->get().flip_h;
}
bool TileMap::is_cell_y_flipped(int p_x,int p_y) const {

	PosKey pk(p_x,p_y);

	const Map<PosKey,Cell>::Element *E=tile_map.find(pk);

	if (!E)
		return false;

	return E->get().flip_v;
}


void TileMap::_recreate_quadrants() {

	_clear_quadrants();

	for (Map<PosKey,Cell>::Element *E=tile_map.front();E;E=E->next()) {

		PosKey qk(E->key().x/quadrant_size,E->key().y/quadrant_size);

		Map<PosKey,Quadrant>::Element *Q=quadrant_map.find(qk);
		if (!Q) {
			Q=_create_quadrant(qk);
			dirty_quadrant_list.add(&Q->get().dirty_list);
		}

		Q->get().cells.insert(E->key());
		_make_quadrant_dirty(Q);
	}



}

void TileMap::_clear_quadrants() {

	while (quadrant_map.size()) {
		_erase_quadrant( quadrant_map.front() );
	}
}

void TileMap::clear() {

	_clear_quadrants();
	tile_map.clear();
}

void TileMap::_set_tile_data(const DVector<int>& p_data) {

	int c=p_data.size();
	DVector<int>::Read r = p_data.read();


	for(int i=0;i<c;i+=2) {

		const uint8_t *ptr=(const uint8_t*)&r[i];
		uint8_t local[8];
		for(int j=0;j<8;j++)
			local[j]=ptr[j];

#ifdef BIG_ENDIAN_ENABLED


		SWAP(local[0],local[3]);
		SWAP(local[1],local[2]);
		SWAP(local[4],local[7]);
		SWAP(local[5],local[6]);
#endif

		int16_t x = decode_uint16(&local[0]);
		int16_t y = decode_uint16(&local[2]);
		uint32_t v = decode_uint32(&local[4]);
		bool flip_h = v&(1<<29);
		bool flip_v = v&(1<<30);
		v&=(1<<29)-1;

//		if (x<-20 || y <-20 || x>4000 || y>4000)
//			continue;
		set_cell(x,y,v,flip_h,flip_v);

	}

}

DVector<int> TileMap::_get_tile_data() const {

	DVector<int> data;
	data.resize(tile_map.size()*2);
	DVector<int>::Write w = data.write();

	int idx=0;
	for(const Map<PosKey,Cell>::Element *E=tile_map.front();E;E=E->next()) {

		uint8_t *ptr = (uint8_t*)&w[idx];
		encode_uint16(E->key().x,&ptr[0]);
		encode_uint16(E->key().y,&ptr[2]);
		uint32_t val = E->get().id;
		if (E->get().flip_h)
			val|=(1<<29);
		if (E->get().flip_v)
			val|=(1<<30);

		encode_uint32(val,&ptr[4]);
		idx+=2;
	}


	w = DVector<int>::Write();

	return data;

}

Rect2 TileMap::get_item_rect() const {

	const_cast<TileMap*>(this)->_update_dirty_quadrants();
	return rect_cache;
}

void TileMap::set_collision_layer_mask(uint32_t p_layer) {

	collision_layer=p_layer;
	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

		Quadrant &q=E->get();
		Physics2DServer::get_singleton()->body_set_layer_mask(q.static_body,collision_layer);
	}
}

void TileMap::set_collision_friction(float p_friction) {

	friction=p_friction;
	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

		Quadrant &q=E->get();
		Physics2DServer::get_singleton()->body_set_param(q.static_body,Physics2DServer::BODY_PARAM_FRICTION,p_friction);
	}

}

float TileMap::get_collision_friction() const{

	return friction;
}

void TileMap::set_collision_bounce(float p_bounce){

	bounce=p_bounce;
	for (Map<PosKey,Quadrant>::Element *E=quadrant_map.front();E;E=E->next()) {

		Quadrant &q=E->get();
		Physics2DServer::get_singleton()->body_set_param(q.static_body,Physics2DServer::BODY_PARAM_BOUNCE,p_bounce);
	}

}
float TileMap::get_collision_bounce() const{

	return bounce;
}


uint32_t TileMap::get_collision_layer_mask() const {

	return collision_layer;
}

void TileMap::set_mode(Mode p_mode) {

	_clear_quadrants();
	mode=p_mode;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

TileMap::Mode TileMap::get_mode() const {
	return mode;
}

void TileMap::set_half_offset(HalfOffset p_half_offset) {

	_clear_quadrants();
	half_offset=p_half_offset;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

Vector2 TileMap::get_cell_draw_offset() const {

	switch(mode) {

		case MODE_SQUARE: {

			return Vector2();
		} break;
		case MODE_ISOMETRIC: {

			return Vector2(-cell_size.x*0.5,0);

		} break;
		case MODE_CUSTOM: {

			Vector2 min;
			min.x = MIN(custom_transform[0].x,min.x);
			min.y = MIN(custom_transform[0].y,min.y);
			min.x = MIN(custom_transform[1].x,min.x);
			min.y = MIN(custom_transform[1].y,min.y);
			return min;
		} break;
	}

	return Vector2();

}

TileMap::HalfOffset TileMap::get_half_offset() const {
	return half_offset;
}

Matrix32 TileMap::get_cell_transform() const {

	switch(mode) {

		case MODE_SQUARE: {

			Matrix32 m;
			m[0]*=cell_size.x;
			m[1]*=cell_size.y;
			return m;
		} break;
		case MODE_ISOMETRIC: {

			//isometric only makes sense when y is positive in both x and y vectors, otherwise
			//the drawing of tiles will overlap
			Matrix32 m;
			m[0]=Vector2(cell_size.x*0.5,cell_size.y*0.5);
			m[1]=Vector2(-cell_size.x*0.5,cell_size.y*0.5);
			return m;

		} break;
		case MODE_CUSTOM: {

			return custom_transform;
		} break;
	}

	return Matrix32();
}

void TileMap::set_custom_transform(const Matrix32& p_xform) {

	_clear_quadrants();
	custom_transform=p_xform;
	_recreate_quadrants();
	emit_signal("settings_changed");

}

Matrix32 TileMap::get_custom_transform() const{

	return custom_transform;
}

Vector2 TileMap::_map_to_world(int x,int y,bool p_ignore_ofs) const {

	Vector2 ret = get_cell_transform().xform(Vector2(x,y));
	if (!p_ignore_ofs) {
		switch(half_offset) {

			case HALF_OFFSET_X: {
				if (ABS(y)&1) {

					ret+=get_cell_transform()[0]*0.5;
				}
			} break;
			case HALF_OFFSET_Y: {
				if (ABS(x)&1) {
					ret+=get_cell_transform()[1]*0.5;
				}
			} break;
			default: {}
		}
	}
	return ret;
}
Vector2 TileMap::map_to_world(const Vector2& p_pos,bool p_ignore_ofs) const {

	return _map_to_world(p_pos.x,p_pos.y,p_ignore_ofs);
}
Vector2 TileMap::world_to_map(const Vector2& p_pos) const{

	Vector2 ret = get_cell_transform().affine_inverse().xform(p_pos);


	switch(half_offset) {

		case HALF_OFFSET_X: {
			if (int(ret.y)&1) {

				ret.x-=0.5;
			}
		} break;
		case HALF_OFFSET_Y: {
			if (int(ret.x)&1) {
				ret.y-=0.5;
			}
		} break;
		default: {}
	}

	return ret.floor();
}


void TileMap::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_tileset","tileset:TileSet"),&TileMap::set_tileset);
	ObjectTypeDB::bind_method(_MD("get_tileset:TileSet"),&TileMap::get_tileset);

	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&TileMap::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&TileMap::get_mode);

	ObjectTypeDB::bind_method(_MD("set_half_offset","half_offset"),&TileMap::set_half_offset);
	ObjectTypeDB::bind_method(_MD("get_half_offset"),&TileMap::get_half_offset);

	ObjectTypeDB::bind_method(_MD("set_custom_transform","custom_transform"),&TileMap::set_custom_transform);
	ObjectTypeDB::bind_method(_MD("get_custom_transform"),&TileMap::get_custom_transform);

	ObjectTypeDB::bind_method(_MD("set_cell_size","size"),&TileMap::set_cell_size);
	ObjectTypeDB::bind_method(_MD("get_cell_size"),&TileMap::get_cell_size);

	ObjectTypeDB::bind_method(_MD("_set_old_cell_size","size"),&TileMap::_set_old_cell_size);
	ObjectTypeDB::bind_method(_MD("_get_old_cell_size"),&TileMap::_get_old_cell_size);

	ObjectTypeDB::bind_method(_MD("set_quadrant_size","size"),&TileMap::set_quadrant_size);
	ObjectTypeDB::bind_method(_MD("get_quadrant_size"),&TileMap::get_quadrant_size);

	ObjectTypeDB::bind_method(_MD("set_center_x","enable"),&TileMap::set_center_x);
	ObjectTypeDB::bind_method(_MD("get_center_x"),&TileMap::get_center_x);

	ObjectTypeDB::bind_method(_MD("set_center_y","enable"),&TileMap::set_center_y);
	ObjectTypeDB::bind_method(_MD("get_center_y"),&TileMap::get_center_y);

	ObjectTypeDB::bind_method(_MD("set_collision_layer_mask","mask"),&TileMap::set_collision_layer_mask);
	ObjectTypeDB::bind_method(_MD("get_collision_layer_mask"),&TileMap::get_collision_layer_mask);

	ObjectTypeDB::bind_method(_MD("set_collision_friction","value"),&TileMap::set_collision_friction);
	ObjectTypeDB::bind_method(_MD("get_collision_friction"),&TileMap::get_collision_friction);

	ObjectTypeDB::bind_method(_MD("set_collision_bounce","value"),&TileMap::set_collision_bounce);
	ObjectTypeDB::bind_method(_MD("get_collision_bounce"),&TileMap::get_collision_bounce);

	ObjectTypeDB::bind_method(_MD("set_cell","x","y","tile","flip_x","flip_y"),&TileMap::set_cell,DEFVAL(false),DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_cell","x","y"),&TileMap::get_cell);
	ObjectTypeDB::bind_method(_MD("is_cell_x_flipped","x","y"),&TileMap::is_cell_x_flipped);
	ObjectTypeDB::bind_method(_MD("is_cell_y_flipped","x","y"),&TileMap::is_cell_y_flipped);

	ObjectTypeDB::bind_method(_MD("clear"),&TileMap::clear);

	ObjectTypeDB::bind_method(_MD("map_to_world","mappos","ignore_half_ofs"),&TileMap::map_to_world,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("world_to_map","worldpos"),&TileMap::world_to_map);

	ObjectTypeDB::bind_method(_MD("_clear_quadrants"),&TileMap::_clear_quadrants);
	ObjectTypeDB::bind_method(_MD("_recreate_quadrants"),&TileMap::_recreate_quadrants);
	ObjectTypeDB::bind_method(_MD("_update_dirty_quadrants"),&TileMap::_update_dirty_quadrants);

	ObjectTypeDB::bind_method(_MD("_set_tile_data"),&TileMap::_set_tile_data);
	ObjectTypeDB::bind_method(_MD("_get_tile_data"),&TileMap::_get_tile_data);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"mode",PROPERTY_HINT_ENUM,"Square,Isometric,Custom"),_SCS("set_mode"),_SCS("get_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"tile_set",PROPERTY_HINT_RESOURCE_TYPE,"TileSet"),_SCS("set_tileset"),_SCS("get_tileset"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"cell_size",PROPERTY_HINT_RANGE,"1,8192,1",0),_SCS("_set_old_cell_size"),_SCS("_get_old_cell_size"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"cell/size",PROPERTY_HINT_RANGE,"1,8192,1"),_SCS("set_cell_size"),_SCS("get_cell_size"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"cell/quadrant_size",PROPERTY_HINT_RANGE,"1,128,1"),_SCS("set_quadrant_size"),_SCS("get_quadrant_size"));
	ADD_PROPERTY( PropertyInfo(Variant::MATRIX32,"cell/custom_transform"),_SCS("set_custom_transform"),_SCS("get_custom_transform"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"cell/half_offset",PROPERTY_HINT_ENUM,"Offset X,Offset Y,Disabled"),_SCS("set_half_offset"),_SCS("get_half_offset"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"collision/friction",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_collision_friction"),_SCS("get_collision_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"collision/bounce",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_collision_bounce"),_SCS("get_collision_bounce"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"collision/layers",PROPERTY_HINT_ALL_FLAGS),_SCS("set_collision_layer_mask"),_SCS("get_collision_layer_mask"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"tile_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_tile_data"),_SCS("_get_tile_data"));

	ADD_SIGNAL(MethodInfo("settings_changed"));

	BIND_CONSTANT( INVALID_CELL );
	BIND_CONSTANT( MODE_SQUARE );
	BIND_CONSTANT( MODE_ISOMETRIC );
	BIND_CONSTANT( MODE_CUSTOM );
	BIND_CONSTANT( HALF_OFFSET_X );
	BIND_CONSTANT( HALF_OFFSET_Y );
	BIND_CONSTANT( HALF_OFFSET_DISABLED );

}

TileMap::TileMap() {



	rect_cache_dirty=true;
	pending_update=false;
	quadrant_order_dirty=false;
	quadrant_size=16;
	cell_size=Size2(64,64);
	center_x=false;
	center_y=false;
	collision_layer=1;
	friction=1;
	bounce=0;
	mode=MODE_SQUARE;
	half_offset=HALF_OFFSET_DISABLED;

	fp_adjust=0.01;
	fp_adjust=0.01;
}

TileMap::~TileMap() {

	clear();
}
