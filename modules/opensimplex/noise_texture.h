/*************************************************************************/
/*  noise_texture.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef NOISE_TEXTURE_H
#define NOISE_TEXTURE_H

#include "open_simplex_noise.h"

#include "core/image.h"
#include "core/reference.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"

class NoiseTexture : public Texture {
	GDCLASS(NoiseTexture, Texture);

private:
	Ref<Image> data;

	Thread noise_thread;

	bool first_time;
	bool update_queued;
	bool regen_queued;

	RID texture;
	uint32_t flags;

	Ref<OpenSimplexNoise> noise;
	Vector2i size;
	Vector2 noise_offset;
	bool seamless;
	bool as_normalmap;
	float bump_strength;

	void _thread_done(const Ref<Image> &p_image);
	static void _thread_function(void *p_ud);

	void _queue_update();
	Ref<Image> _generate_texture();
	void _update_texture();
	void _set_texture_data(const Ref<Image> &p_image);

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;

public:
	void set_noise(Ref<OpenSimplexNoise> p_noise);
	Ref<OpenSimplexNoise> get_noise();

	void set_width(int p_width);
	void set_height(int p_height);

	void set_noise_offset(Vector2 p_noise_offset);
	Vector2 get_noise_offset() const;

	void set_seamless(bool p_seamless);
	bool get_seamless();

	void set_as_normalmap(bool p_as_normalmap);
	bool is_normalmap();

	void set_bump_strength(float p_bump_strength);
	float get_bump_strength();

	int get_width() const;
	int get_height() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	virtual RID get_rid() const { return texture; }
	virtual bool has_alpha() const { return false; }

	virtual Ref<Image> get_data() const;

	NoiseTexture();
	virtual ~NoiseTexture();
};

#endif // NOISE_TEXTURE_H
