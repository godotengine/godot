/*************************************************************************/
/*  remote_blend_transform_2d.h                                          */
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
#include "scene/2d/node_2d.h"

class RemoteBlendTransform2D : public Node2D {

	OBJ_TYPE(RemoteBlendTransform2D,Node2D);

	NodePath remote_node;
	Matrix32 remote_mat;
	ObjectID cache;
	float    blend_factor;
	bool     ready;

	void _update_remote();
	void _restore_remote();
	void _update_cache();
	void _node_exited_scene();
	
protected:

	static void _bind_methods();
	void _notification(int p_what);
	
public:

	void set_remote_node(const NodePath& p_remote_node);
	NodePath get_remote_node() const;
	
	void set_blend_factor( float factor );
	float get_blend_factor();
	
	void set_remote_transform( const Matrix32& m_remote );
	Matrix32 get_remote_transform() const;

	RemoteBlendTransform2D();
};
