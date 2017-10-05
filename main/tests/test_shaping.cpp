/*************************************************************************/
/*  test_shaping.cpp                                                     */
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

#include "test_shaping.h"

#include "core/math/math_funcs.h"
#include "core/os/file_access.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "scene/resources/dynamic_font.h"
#include "scene/resources/shaped_attributed_string.h"
#include "servers/visual_server.h"

namespace TestShaping {

class TestMainLoop : public MainLoop {

	ShapedAttributedString as1;

	RID canvas;
	RID vp;
	RID item;

	//Fonts
	Ref<DynamicFont> default_font;

	bool quit;

public:
	virtual void input_event(const Ref<InputEvent> &p_event) {

		if (p_event->is_pressed())
			quit = true;
	}

	virtual void init() {
		print_line("INITIALIZING SHAPING TEST");

		//Load fonts
		bool fonts_ok = true;
		String font_path;
		List<String> cmdline = OS::get_singleton()->get_cmdline_args();
		for (List<String>::Element *E = cmdline.front(); E; E = E->next()) {
			if (E->get() == "--datapath") {
				if (E->next()) {
					font_path = E->next()->get();
					break;
				}
			}
		}

		if (FileAccess::exists(font_path + "NotoSans-Regular.ttf")) {
			default_font.instance();
			default_font->set_size(30);
			Ref<DynamicFontData> default_font_data;
			default_font_data.instance();
			default_font_data->set_font_path(font_path + "NotoSans-Regular.ttf");
			default_font->set_font_data(default_font_data);
		} else {
			print_line("  \"NotoSans-Regular.ttf\" not found");
			fonts_ok = false;
		}

		if (!fonts_ok) {
			print_line("Usage: \"godot --test shaping --datapath {PATH}\"");
			quit = true;
			return;
		}

		//Load ICU data

		//Init canvas
		VisualServer *vs = VisualServer::get_singleton();
		vp = vs->viewport_create();
		canvas = vs->canvas_create();
		Size2i screen_size = OS::get_singleton()->get_window_size();
		vs->viewport_attach_canvas(vp, canvas);
		vs->viewport_set_size(vp, screen_size.x, screen_size.y);
		vs->viewport_attach_to_screen(vp, Rect2(Vector2(), screen_size));
		vs->viewport_set_active(vp, true);

		item = vs->canvas_item_create();
		vs->canvas_item_set_parent(item, canvas);

		//Build and draw test strings
		as1.set_base_font(default_font);
		as1.set_text("Test Attributes 1.0 = XXXX");
		as1.add_attribute(TEXT_ATTRIBUTE_COLOR, Color(0.5, 1, 0.5, 0.7), 0, 4);
		as1.add_attribute(TEXT_ATTRIBUTE_COLOR, Color(1, 0.5, 1, 1), 16, 17);
		as1.draw(item, Point2(20, 120), Color(1, 1, 1), false);

		vs->canvas_item_add_line(item, Point2(20, 120), Point2(20 + as1.get_width(), 120), Color(0, 1, 0, 0.2), 2);
		vs->canvas_item_add_line(item, Point2(20, 120 - as1.get_ascent()), Point2(20 + as1.get_width(), 120 - as1.get_ascent()), Color(1, 0, 0, 0.2), 2);
		vs->canvas_item_add_line(item, Point2(20, 120 + as1.get_descent()), Point2(20 + as1.get_width(), 120 + as1.get_descent()), Color(1, 0, 1, 0.2), 2);

		quit = false;
	}

	virtual bool iteration(float p_time) {

		return quit;
	}

	virtual bool idle(float p_time) {

		return quit;
	}

	virtual void finish() {

		VisualServer::get_singleton()->free(item);
		VisualServer::get_singleton()->free(canvas);
		VisualServer::get_singleton()->free(vp);
	}
};

MainLoop *test() {

	return memnew(TestMainLoop);
}
} // namespace TestShaping
