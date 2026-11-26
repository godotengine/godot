/**************************************************************************/
/*  hb_draw_func_wrap.cpp                                                 */
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

#include "hb_draw_func_wrap.h"

#if defined(__GNUC__) && !defined(__clang__)
#define GODOT_GCC_PRAGMA(m_content) _Pragma(#m_content)
#define GODOT_GCC_WARNING_PUSH GODOT_GCC_PRAGMA(GCC diagnostic push)
#define GODOT_GCC_WARNING_IGNORE(m_warning) GODOT_GCC_PRAGMA(GCC diagnostic ignored m_warning)
#define GODOT_GCC_WARNING_POP GODOT_GCC_PRAGMA(GCC diagnostic pop)
#else
#define GODOT_GCC_PRAGMA(m_content)
#define GODOT_GCC_WARNING_PUSH
#define GODOT_GCC_WARNING_IGNORE(m_warning)
#define GODOT_GCC_WARNING_POP
#endif

#undef DEBUG_ENABLED

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wctor-dtor-privacy")
GODOT_GCC_WARNING_IGNORE("-Wduplicated-branches")
#include <hb-draw.hh>
GODOT_GCC_WARNING_POP

struct HBContext {
	hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
	hb_draw_funcs_t *draw_func = nullptr;
	hb_gpu_draw_t *draw = nullptr;
	bool path_opened = false;
};

static int hb_move_to(const FT_Vector *to, void *user) {
	HBContext *context = static_cast<HBContext *>(user);
	if (context->path_opened) {
		context->draw_func->emit_close_path(context->draw, context->st);
	}
	context->draw_func->emit_move_to(context->draw, context->st, to->x, to->y);
	context->path_opened = true;
	return 0;
}

static int hb_line_to(const FT_Vector *to, void *user) {
	HBContext *context = static_cast<HBContext *>(user);
	context->draw_func->emit_line_to(context->draw, context->st, to->x, to->y);
	return 0;
}

static int hb_conic_to(const FT_Vector *control, const FT_Vector *to, void *user) {
	HBContext *context = static_cast<HBContext *>(user);
	context->draw_func->emit_quadratic_to(context->draw, context->st, control->x, control->y, to->x, to->y);
	return 0;
}

static int hb_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user) {
	HBContext *context = static_cast<HBContext *>(user);
	context->draw_func->emit_cubic_to(context->draw, context->st, control1->x, control1->y, control2->x, control2->y, to->x, to->y);
	return 0;
}

bool hb_gpu_draw_glyph_outline_with_ft_stroker_or_fail(hb_gpu_draw_t *p_draw, FT_Outline *p_outline) {
	HBContext context = { HB_DRAW_STATE_DEFAULT, hb_gpu_draw_get_funcs(p_draw), p_draw };
	FT_Outline_Funcs hb_functions;
	hb_functions.move_to = &hb_move_to;
	hb_functions.line_to = &hb_line_to;
	hb_functions.conic_to = &hb_conic_to;
	hb_functions.cubic_to = &hb_cubic_to;
	hb_functions.shift = 0;
	hb_functions.delta = 0;

	int error = FT_Outline_Decompose(p_outline, &hb_functions, &context);
	if (context.path_opened) {
		context.draw_func->emit_close_path(context.draw, context.st);
	}

	return error == 0;
}
