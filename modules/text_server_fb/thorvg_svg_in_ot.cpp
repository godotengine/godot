/**************************************************************************/
/*  thorvg_svg_in_ot.cpp                                                  */
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

#ifdef GDEXTENSION
// Headers for building as GDExtension plug-in.

#include <godot_cpp/classes/xml_parser.hpp>
#include <godot_cpp/core/mutex_lock.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/templates/vector.hpp>

using namespace godot;

#elif defined(GODOT_MODULE)
// Headers for building as built-in module.

#include "core/error/error_macros.h"
#include "core/io/xml_parser.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"
#include "core/variant/variant.h"

#include "modules/modules_enabled.gen.h" // For svg.
#endif

#ifdef MODULE_SVG_ENABLED
#ifdef MODULE_FREETYPE_ENABLED

#include "thorvg_svg_in_ot.h"

#include "thorvg_bounds_iterator.h"

#include <freetype/otsvg.h>
#include <ft2build.h>

#include <math.h>
#include <stdlib.h>

FT_Error tvg_svg_in_ot_init(FT_Pointer *p_state) {
	*p_state = memnew(TVG_State);

	return FT_Err_Ok;
}

void tvg_svg_in_ot_free(FT_Pointer *p_state) {
	TVG_State *state = *reinterpret_cast<TVG_State **>(p_state);
	memdelete(state);
}

FT_Error tvg_svg_in_ot_preset_slot(FT_GlyphSlot p_slot, FT_Bool p_cache, FT_Pointer *p_state) {
	TVG_State *state = *reinterpret_cast<TVG_State **>(p_state);
	if (!state) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "SVG in OT state not initialized.");
	}
	MutexLock lock(state->mutex);

	FT_SVG_Document document = (FT_SVG_Document)p_slot->other;
	FT_Size_Metrics metrics = document->metrics;

	GL_State &gl_state = state->glyph_map[p_slot->glyph_index];
	if (!gl_state.ready) {
		Ref<XMLParser> parser;
		parser.instantiate();
		parser->_open_buffer((const uint8_t *)document->svg_document, document->svg_document_length);

		float aspect = 1.0f;
		String xml_body;
		while (parser->read() == OK) {
			if (parser->has_attribute("id")) {
				const String &gl_name = parser->get_named_attribute_value("id");
				if (gl_name.begins_with("glyph")) {
					int dot_pos = gl_name.find(".");
					int64_t gl_idx = gl_name.substr(5, (dot_pos > 0) ? dot_pos - 5 : -1).to_int();
					if (p_slot->glyph_index != gl_idx) {
						parser->skip_section();
						continue;
					}
				}
			}
			if (parser->get_node_type() == XMLParser::NODE_ELEMENT && parser->get_node_name() == "svg") {
				if (parser->has_attribute("viewBox")) {
					PackedStringArray vb = parser->get_named_attribute_value("viewBox").split(" ");

					if (vb.size() == 4) {
						aspect = vb[2].to_float() / vb[3].to_float();
					}
				}
				continue;
			}
			if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
				xml_body += vformat("<%s", parser->get_node_name());
				for (int i = 0; i < parser->get_attribute_count(); i++) {
					xml_body += vformat(" %s=\"%s\"", parser->get_attribute_name(i), parser->get_attribute_value(i));
				}

				if (parser->is_empty()) {
					xml_body += "/>";
				} else {
					xml_body += ">";
				}
			} else if (parser->get_node_type() == XMLParser::NODE_TEXT) {
				xml_body += parser->get_node_data();
			} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END) {
				xml_body += vformat("</%s>", parser->get_node_name());
			}
		}
		String temp_xml_str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\">" + xml_body;
		CharString temp_xml = temp_xml_str.utf8();

		std::unique_ptr<tvg::Picture> picture = tvg::Picture::gen();
		tvg::Result result = picture->load(temp_xml.get_data(), temp_xml.length(), "svg+xml", false);
		if (result != tvg::Result::Success) {
			ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to load SVG document (bounds detection).");
		}

		float min_x = INFINITY, min_y = INFINITY, max_x = -INFINITY, max_y = -INFINITY;
		tvg_get_bounds(picture.get(), min_x, min_y, max_x, max_y);

		float new_h = (max_y - min_y);
		float new_w = (max_x - min_x);

		if (new_h * aspect >= new_w) {
			new_w = (new_h * aspect);
		} else {
			new_h = (new_w / aspect);
		}

		String xml_code_str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" + rtos(min_x) + " " + rtos(min_y) + " " + rtos(new_w) + " " + rtos(new_h) + "\">" + xml_body;
		gl_state.xml_code = xml_code_str.utf8();

		picture = tvg::Picture::gen();
		result = picture->load(gl_state.xml_code.get_data(), gl_state.xml_code.length(), "svg+xml", false);
		if (result != tvg::Result::Success) {
			ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to load SVG document (glyph metrics).");
		}

		float x_svg_to_out, y_svg_to_out;
		x_svg_to_out = (float)metrics.x_ppem / new_w;
		y_svg_to_out = (float)metrics.y_ppem / new_h;

		gl_state.m.e11 = (double)document->transform.xx / (1 << 16) * x_svg_to_out;
		gl_state.m.e12 = -(double)document->transform.xy / (1 << 16) * x_svg_to_out;
		gl_state.m.e21 = -(double)document->transform.yx / (1 << 16) * y_svg_to_out;
		gl_state.m.e22 = (double)document->transform.yy / (1 << 16) * y_svg_to_out;
		gl_state.m.e13 = (double)document->delta.x / 64 * new_w / metrics.x_ppem;
		gl_state.m.e23 = -(double)document->delta.y / 64 * new_h / metrics.y_ppem;
		gl_state.m.e31 = 0;
		gl_state.m.e32 = 0;
		gl_state.m.e33 = 1;

		result = picture->transform(gl_state.m);
		if (result != tvg::Result::Success) {
			ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to apply transform to SVG document.");
		}

		result = picture->bounds(&gl_state.x, &gl_state.y, &gl_state.w, &gl_state.h, true);
		if (result != tvg::Result::Success) {
			ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to get SVG bounds.");
		}

		gl_state.bmp_y = gl_state.h + metrics.descender / 64.f;
		gl_state.bmp_x = 0;

		gl_state.ready = true;
	}

	p_slot->bitmap_left = (FT_Int)gl_state.bmp_x;
	p_slot->bitmap_top = (FT_Int)gl_state.bmp_y;

	float tmp = ceil(gl_state.h);
	p_slot->bitmap.rows = (unsigned int)tmp;
	tmp = ceil(gl_state.w);
	p_slot->bitmap.width = (unsigned int)tmp;
	p_slot->bitmap.pitch = (int)p_slot->bitmap.width * 4;
	p_slot->bitmap.pixel_mode = FT_PIXEL_MODE_BGRA;

	float metrics_width, metrics_height;
	float horiBearingX, horiBearingY;
	float vertBearingX, vertBearingY;

	metrics_width = (float)gl_state.w;
	metrics_height = (float)gl_state.h;
	horiBearingX = (float)gl_state.x;
	horiBearingY = (float)-gl_state.y;
	vertBearingX = p_slot->metrics.horiBearingX / 64.0f - p_slot->metrics.horiAdvance / 64.0f / 2;
	vertBearingY = (p_slot->metrics.vertAdvance / 64.0f - p_slot->metrics.height / 64.0f) / 2;

	tmp = roundf(metrics_width * 64);
	p_slot->metrics.width = (FT_Pos)tmp;
	tmp = roundf(metrics_height * 64);
	p_slot->metrics.height = (FT_Pos)tmp;

	p_slot->metrics.horiBearingX = (FT_Pos)(horiBearingX * 64);
	p_slot->metrics.horiBearingY = (FT_Pos)(horiBearingY * 64);
	p_slot->metrics.vertBearingX = (FT_Pos)(vertBearingX * 64);
	p_slot->metrics.vertBearingY = (FT_Pos)(vertBearingY * 64);

	if (p_slot->metrics.vertAdvance == 0) {
		p_slot->metrics.vertAdvance = (FT_Pos)(metrics_height * 1.2f * 64);
	}

	return FT_Err_Ok;
}

FT_Error tvg_svg_in_ot_render(FT_GlyphSlot p_slot, FT_Pointer *p_state) {
	TVG_State *state = *reinterpret_cast<TVG_State **>(p_state);
	if (!state) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "SVG in OT state not initialized.");
	}
	MutexLock lock(state->mutex);

	if (!state->glyph_map.has(p_slot->glyph_index)) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "SVG glyph not loaded.");
	}

	GL_State &gl_state = state->glyph_map[p_slot->glyph_index];
	ERR_FAIL_COND_V_MSG(!gl_state.ready, FT_Err_Invalid_SVG_Document, "SVG glyph not ready.");

	std::unique_ptr<tvg::Picture> picture = tvg::Picture::gen();
	tvg::Result res = picture->load(gl_state.xml_code.get_data(), gl_state.xml_code.length(), "svg+xml", false);
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to load SVG document (glyph rendering).");
	}
	res = picture->transform(gl_state.m);
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_SVG_Document, "Failed to apply transform to SVG document.");
	}

	std::unique_ptr<tvg::SwCanvas> sw_canvas = tvg::SwCanvas::gen();
	res = sw_canvas->target((uint32_t *)p_slot->bitmap.buffer, (int)p_slot->bitmap.width, (int)p_slot->bitmap.width, (int)p_slot->bitmap.rows, tvg::SwCanvas::ARGB8888S);
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_Outline, "Failed to create SVG canvas.");
	}
	res = sw_canvas->push(std::move(picture));
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_Outline, "Failed to set SVG canvas source.");
	}
	res = sw_canvas->draw();
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_Outline, "Failed to draw to SVG canvas.");
	}
	res = sw_canvas->sync();
	if (res != tvg::Result::Success) {
		ERR_FAIL_V_MSG(FT_Err_Invalid_Outline, "Failed to sync SVG canvas.");
	}

	state->glyph_map.erase(p_slot->glyph_index);

	p_slot->bitmap.pixel_mode = FT_PIXEL_MODE_BGRA;
	p_slot->bitmap.num_grays = 256;
	p_slot->format = FT_GLYPH_FORMAT_BITMAP;

	return FT_Err_Ok;
}

SVG_RendererHooks tvg_svg_in_ot_hooks = {
	(SVG_Lib_Init_Func)tvg_svg_in_ot_init,
	(SVG_Lib_Free_Func)tvg_svg_in_ot_free,
	(SVG_Lib_Render_Func)tvg_svg_in_ot_render,
	(SVG_Lib_Preset_Slot_Func)tvg_svg_in_ot_preset_slot,
};

SVG_RendererHooks *get_tvg_svg_in_ot_hooks() {
	return &tvg_svg_in_ot_hooks;
}

#endif // MODULE_FREETYPE_ENABLED
#endif // MODULE_SVG_ENABLED
