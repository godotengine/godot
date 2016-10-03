/*************************************************************************/
/*  light.h                                                              */
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
#ifndef LIGHT_H
#define LIGHT_H


#include "scene/3d/visual_instance.h"
#include "scene/resources/texture.h"
#include "servers/visual_server.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Light : public VisualInstance {

	OBJ_TYPE( Light, VisualInstance );
	OBJ_CATEGORY("3D Light Nodes");

public:



private:

	VS::LightType type;
	bool editor_only;
	void _update_visibility();
// bind helpers

protected:

	RID light;

	virtual bool _can_gizmo_scale() const;
	
	static void _bind_methods();
	void _notification(int p_what);


	Light(VisualServer::LightType p_type);
public:

	VS::LightType get_light_type() const { return type; }

	void set_editor_only(bool p_editor_only);
	bool is_editor_only() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Light();
	~Light();

};



class DirectionalLight : public Light {

	OBJ_TYPE( DirectionalLight, Light );

public:


private:


protected:
	static void _bind_methods();
public:


	DirectionalLight();
};


class OmniLight : public Light {

	OBJ_TYPE( OmniLight, Light );
protected:
	static void _bind_methods();

public:


	OmniLight() : Light( VisualServer::LIGHT_OMNI ) { }
};

class SpotLight : public Light {

	OBJ_TYPE( SpotLight, Light );
protected:
	static void _bind_methods();
public:


	SpotLight() : Light( VisualServer::LIGHT_SPOT ) {}
};


#endif
