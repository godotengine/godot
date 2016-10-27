/*************************************************************************/
/*  environment.h                                                        */
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
#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "resource.h"
#include "servers/visual_server.h"
#include "scene/resources/texture.h"

class Environment : public Resource {

	OBJ_TYPE(Environment,Resource);
public:

	enum BGMode {

		BG_CLEAR_COLOR,
		BG_COLOR,
		BG_SKYBOX,
		BG_CANVAS,
		BG_KEEP,
		BG_MAX
	};

	enum GlowBlendMode {
		GLOW_BLEND_MODE_ADDITIVE,
		GLOW_BLEND_MODE_SCREEN,
		GLOW_BLEND_MODE_SOFTLIGHT,
		GLOW_BLEND_MODE_DISABLED,
	};

private:
	RID environment;

	BGMode bg_mode;
	Ref<CubeMap> bg_skybox;
	float bg_skybox_scale;
	Color bg_color;
	float bg_energy;
	int bg_canvas_max_layer;
	Color ambient_color;
	float ambient_energy;
	float ambient_skybox_contribution;

protected:

	static void _bind_methods();
	virtual void _validate_property(PropertyInfo& property) const;

public:


	void set_background(BGMode p_bg);
	void set_skybox(const Ref<CubeMap>& p_skybox);
	void set_skybox_scale(float p_scale);
	void set_bg_color(const Color& p_color);
	void set_bg_energy(float p_energy);
	void set_canvas_max_layer(int p_max_layer);
	void set_ambient_light_color(const Color& p_color);
	void set_ambient_light_energy(float p_energy);
	void set_ambient_light_skybox_contribution(float p_energy);

	BGMode get_background() const;
	Ref<CubeMap> get_skybox() const;
	float get_skybox_scale() const;
	Color get_bg_color() const;
	float get_bg_energy() const;
	int get_canvas_max_layer() const;
	Color get_ambient_light_color() const;
	float get_ambient_light_energy() const;
	float get_ambient_light_skybox_contribution() const;


	virtual RID get_rid() const;

	Environment();
	~Environment();
};

VARIANT_ENUM_CAST(Environment::BGMode)
VARIANT_ENUM_CAST(Environment::GlowBlendMode)


#endif // ENVIRONMENT_H
