/**************************************************************************/
/*  rendering_effect.h                                                    */
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

#ifndef RENDERING_EFFECT_H
#define RENDERING_EFFECT_H

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"
#include "servers/rendering/storage/render_data.h"

class RenderingEffect : public Resource {
	GDCLASS(RenderingEffect, Resource);

public:
	enum EffectCallbackType {
		EFFECT_CALLBACK_TYPE_PRE_OPAQUE,
		EFFECT_CALLBACK_TYPE_POST_OPAQUE,
		EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT,
		EFFECT_CALLBACK_TYPE_POST_TRANSPARENT,
		EFFECT_CALLBACK_TYPE_MAX
	};

private:
	RID rid;
	EffectCallbackType effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT;

	bool access_resolved_color = false;
	bool access_resolved_depth = false;
	bool needs_motion_vectors = false;

protected:
	static void _bind_methods();

	GDVIRTUAL2(_render_callback, int, const RenderData *)

public:
	virtual RID get_rid() const override { return rid; }

	void set_effect_callback_type(EffectCallbackType p_callback_type);
	EffectCallbackType get_effect_callback_type() const;

	void set_access_resolved_color(bool p_enabled);
	bool get_access_resolved_color() const;

	void set_access_resolved_depth(bool p_enabled);
	bool get_access_resolved_depth() const;

	void set_needs_motion_vectors(bool p_enabled);
	bool get_needs_motion_vectors() const;

	RenderingEffect();
	~RenderingEffect();
};

VARIANT_ENUM_CAST(RenderingEffect::EffectCallbackType)

#endif // RENDERING_EFFECT_H
