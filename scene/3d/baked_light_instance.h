/*************************************************************************/
/*  baked_light_instance.h                                               */
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
#ifndef BAKED_LIGHT_INSTANCE_H
#define BAKED_LIGHT_INSTANCE_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/baked_light.h"

class BakedLightBaker;


class BakedLightInstance : public VisualInstance {
	OBJ_TYPE(BakedLightInstance,VisualInstance);

	Ref<BakedLight> baked_light;


protected:

	static void _bind_methods();
public:


	RID get_baked_light_instance() const;

	void set_baked_light(const Ref<BakedLight>& baked_light);
	Ref<BakedLight> get_baked_light() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	String get_configuration_warning() const;

	BakedLightInstance();
};



class BakedLightSampler : public VisualInstance {
	OBJ_TYPE(BakedLightSampler,VisualInstance);


public:

	enum Param {
		PARAM_RADIUS=VS::BAKED_LIGHT_SAMPLER_RADIUS,
		PARAM_STRENGTH=VS::BAKED_LIGHT_SAMPLER_STRENGTH,
		PARAM_ATTENUATION=VS::BAKED_LIGHT_SAMPLER_ATTENUATION,
		PARAM_DETAIL_RATIO=VS::BAKED_LIGHT_SAMPLER_DETAIL_RATIO,
		PARAM_MAX=VS::BAKED_LIGHT_SAMPLER_MAX
	};



protected:

	RID base;
	float params[PARAM_MAX];
	int resolution;
	static void _bind_methods();
public:

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_param(Param p_param,float p_value);
	float get_param(Param p_param) const;

	void set_resolution(int p_resolution);
	int get_resolution() const;

	BakedLightSampler();
	~BakedLightSampler();
};

VARIANT_ENUM_CAST( BakedLightSampler::Param );


#endif // BAKED_LIGHT_H
