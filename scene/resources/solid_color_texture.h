/**************************************************************************/
/*  solid_color_texture.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "scene/resources/texture.h"

class SolidColorTexture : public Texture2D {
	GDCLASS(SolidColorTexture, Texture2D);

private:
	Color color = Color(1, 1, 1, 1);
	Size2i size = Size2i(64, 64);
	mutable RID texture;
	mutable bool update_pending;

	void _queue_update() const;
	void _update() const;

protected:
	static void _bind_methods();

public:
	void set_color(const Color &p_color);
	Color get_color() const;

	void set_size(const Size2 &p_size);
	virtual Size2 get_size() const override;

	virtual RID get_rid() const override;
	virtual int get_width() const override { return size.x; }
	virtual int get_height() const override { return size.y; }
	virtual bool has_alpha() const override { return true; }

	virtual Ref<Image> get_image() const override;

	SolidColorTexture();
	~SolidColorTexture();
};
