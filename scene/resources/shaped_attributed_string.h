/*************************************************************************/
/*  shaped_attributed_string.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SHAPED_ATTRIBUTED_STRING_H
#define SHAPED_ATTRIBUTED_STRING_H

#include "scene/resources/shaped_string.h"

enum TextAttribute {

	TEXT_ATTRIBUTE_FONT, //Ref<Font>
	TEXT_ATTRIBUTE_FONT_FEATURES, //String - 4 letter tag comma separated list
	TEXT_ATTRIBUTE_LANGUAGE, //String - 4 letter tag + css mode
	TEXT_ATTRIBUTE_COLOR, //Color
	TEXT_ATTRIBUTE_OUTLINE_COLOR, //Color
	TEXT_ATTRIBUTE_UNDERLINE_COLOR, //Color
	TEXT_ATTRIBUTE_UNDERLINE_WIDTH, //int
	TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR, //Color
	TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH, //int
	TEXT_ATTRIBUTE_REPLACEMENT_IMAGE, //Ref<Texture>
	TEXT_ATTRIBUTE_REPLACEMENT_RECT, //Size2
	TEXT_ATTRIBUTE_REPLACEMENT_VALIGN, //Enum (VAlign)
	TEXT_ATTRIBUTE_META //Variant (User definde data)
};

/*************************************************************************/
/*  ShapedAttributedString                                               */
/*************************************************************************/

class ShapedAttributedString : public ShapedString {
	GDCLASS(ShapedAttributedString, ShapedString);

protected:
	enum {
		_CLUSTER_TYPE_IMAGE = 11, //Embedded image
		_CLUSTER_TYPE_RECT = 12 //Reserved rect for embedded object
	};

	Vector<Map<TextAttribute, Variant> > visaul_attributes;

	Map<int, Map<TextAttribute, Variant> > attributes;

	virtual bool _compare_attributes(const Map<TextAttribute, Variant> &p_first, const Map<TextAttribute, Variant> &p_second) const;
	virtual void _ensure_break(int p_key);
	virtual void _optimize_attributes();

	virtual Ref<ShapedString> _shape_substring(int32_t p_start, int32_t p_end) const;

	virtual void _generate_justification_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<JustificationOpportunity> &p_ops) const;
	virtual void _generate_break_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<BreakOpportunity> &p_ops) const;

#ifdef USE_TEXT_SHAPING
	virtual void _shape_single_cluster(int32_t p_start, int32_t p_end, hb_direction_t p_run_direction, hb_script_t p_run_script, UChar32 p_codepoint, int p_fallback_index, /*out*/ Cluster &p_cluster) const;
#endif
	virtual void _shape_bidi_script_attrib_run(hb_direction_t p_run_direction, hb_script_t p_run_script, const Map<TextAttribute, Variant> &p_attribs, int32_t p_run_start, int32_t p_run_end, int p_fallback_index);
	virtual void _shape_bidi_script_run(hb_direction_t p_run_direction, hb_script_t p_run_script, int32_t p_run_start, int32_t p_run_end, int p_fallback_index);
	virtual void _shape_rect_run(hb_direction_t p_run_direction, const Size2 &p_size, VAlign p_align, int32_t p_run_start, int32_t p_run_end);
	virtual void _shape_image_run(hb_direction_t p_run_direction, const Ref<Texture> &p_image, VAlign p_align, int32_t p_run_start, int32_t p_run_end);

	static void _bind_methods();

public:
	virtual void replace_text(int32_t p_start, int32_t p_end, const String &p_text);
	virtual void replace_utf8(int32_t p_start, int32_t p_end, const PoolByteArray p_text);
	virtual void replace_utf16(int32_t p_start, int32_t p_end, const PoolByteArray p_text);
	virtual void replace_utf32(int32_t p_start, int32_t p_end, const PoolByteArray p_text);

	virtual void add_attribute(TextAttribute p_attribute, Variant p_value, int p_start, int p_end);
	virtual void remove_attribute(TextAttribute p_attribute, int p_start, int p_end);
	virtual void remove_attributes(int p_start, int p_end);
	virtual void clear_attributes();
	virtual Variant get_attribute(TextAttribute p_attribute, int p_index) const;

	virtual Vector2 draw_cluster(RID p_canvas_item, const Point2 &p_position, int p_index, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const;
	virtual void draw(RID p_canvas_item, const Point2 &p_position, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const;
};

VARIANT_ENUM_CAST(TextAttribute);

#endif
