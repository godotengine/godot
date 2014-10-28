/*************************************************************************/
/*  remote_blend_transform_2d.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014      Guy Rabiller.                                 */
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
#include "remote_blend_transform_2d.h"
#include "scene/scene_string_names.h"

void RemoteBlendTransform2D::_update_cache() {

	cache = 0;
	
	if (!is_inside_scene())
		return;
	
	if (has_node(remote_node)) {
		Node *node = get_node(remote_node);
		if (!node || this==node || node->is_a_parent_of(this) || this->is_a_parent_of(node)) {
			return;
		}

		cache = node->get_instance_ID();

		if( ready ) {
  		Node2D* n  = node->cast_to<Node2D>();
  		remote_mat = n->get_global_transform();
		}
	}
}

void RemoteBlendTransform2D::_update_remote() {

	if( !is_inside_scene()    ) return;
	if( !cache                ) return;

	Object *obj = ObjectDB::get_instance(cache);
	if( !obj                  ) return;

	Node2D *n = obj->cast_to<Node2D>();
	if( !n                    ) return;
	if( !n->is_inside_scene() ) return;

  if( blend_factor >= 1.0f ) {
    n->set_global_transform(get_global_transform());
  } else
  if( blend_factor <= 0.0f ) {
    n->set_global_transform(remote_mat);
  } else {
    Matrix32 me_mat = get_global_transform();
    Matrix32 it_mat( remote_mat.get_rotation()*(1.0-blend_factor)+me_mat.get_rotation()*(blend_factor), Vector2::linear_interpolate( remote_mat.elements[2], me_mat.elements[2], blend_factor ) );
    it_mat.scale_basis( Vector2::linear_interpolate( remote_mat.get_scale(), me_mat.get_scale(), blend_factor ) );
    n->set_global_transform(it_mat);
  }
}

void RemoteBlendTransform2D::_restore_remote() {

  if( !ready                ) return;
	if( !is_inside_scene()    ) return;
	if( !cache                ) return;

	Object *obj = ObjectDB::get_instance(cache);
	if( !obj                  ) return;

	Node2D *n = obj->cast_to<Node2D>();
	if( !n                    ) return;
	if( !n->is_inside_scene() ) return;

	n->set_global_transform(remote_mat);
}

void RemoteBlendTransform2D::_notification(int p_what) {
	switch(p_what) {
		case NOTIFICATION_READY:             { _update_cache();   ready = true;  } break;
		case NOTIFICATION_TRANSFORM_CHANGED: { _update_remote();                 } break;
		case NOTIFICATION_EXIT_SCENE:        { _restore_remote(); ready = false; } break;
	}
}

void RemoteBlendTransform2D::set_remote_node(const NodePath& p_remote_node) {

  _restore_remote();
  
	remote_node = p_remote_node;

  _update_cache();
  
  _update_remote();
}

NodePath RemoteBlendTransform2D::get_remote_node() const{

	return remote_node;
}

void RemoteBlendTransform2D::set_blend_factor( float factor ) {

  blend_factor = factor;
  
  _update_remote();
}

float RemoteBlendTransform2D::get_blend_factor() {

  return blend_factor;
}

void RemoteBlendTransform2D::set_remote_transform( const Matrix32& m_remote ) {

  remote_mat = m_remote;
}

Matrix32 RemoteBlendTransform2D::get_remote_transform() const {

  return remote_mat;
}

void RemoteBlendTransform2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_remote_node","path"),&RemoteBlendTransform2D::set_remote_node);
	ObjectTypeDB::bind_method(_MD("get_remote_node"),&RemoteBlendTransform2D::get_remote_node);
	
	ObjectTypeDB::bind_method(_MD("set_blend_factor","blend"),&RemoteBlendTransform2D::set_blend_factor);
	ObjectTypeDB::bind_method(_MD("get_blend_factor"),&RemoteBlendTransform2D::get_blend_factor);
	
	ObjectTypeDB::bind_method(_MD("set_remote_transform","remote_transform"),&RemoteBlendTransform2D::set_remote_transform);
	ObjectTypeDB::bind_method(_MD("get_remote_transform"),&RemoteBlendTransform2D::get_remote_transform);

	ADD_PROPERTY( PropertyInfo(Variant::NODE_PATH,"remote_path"),_SCS("set_remote_node"),_SCS("get_remote_node"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"blend_factor",PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_blend_factor"),_SCS("get_blend_factor") );
	
	ADD_PROPERTY( PropertyInfo( Variant::MATRIX32, "remote_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR ), _SCS("set_remote_transform"),_SCS("get_remote_transform") );
}

RemoteBlendTransform2D::RemoteBlendTransform2D() {

	cache        = 0;
	blend_factor = 1.0f;
	ready        = false;

}
