/*************************************************************************/
/*  carousel_button.h                                                    */
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

#include "scene/gui/button.h"
#include "scene/gui/texture_button.h"

class CarouselButton : public Button {
	GDCLASS(CarouselButton, Button);

private:
	MouseButton current_mouse_button = MouseButton::NONE;
	int selected = 0;
	bool wraparound = true;
	bool fit_to_longest_item = true;
	struct CarouselButtonItem {
		String text;
		Ref<Texture2D> icon;
	};
	Vector<CarouselButtonItem> items;
	bool update_selected();
	DrawMode left_draw_mode = DrawMode::DRAW_NORMAL;
	DrawMode right_draw_mode = DrawMode::DRAW_NORMAL;
	bool arrow_hovered(bool p_arrow);
	bool arrow_pressed(bool p_arrow);
	bool arrow_disabled(bool p_arrow);
	void set_arrow_pressed(bool p_arrow, bool p_pressed);
	void set_arrow_hovered(bool p_arrow, bool p_hovered);
	void set_arrow_disabled(bool p_arrow, bool p_disabled);
	Size2 get_right_arrow_size() const;
	Size2 get_left_arrow_size() const;
	bool is_over_arrow(bool p_arrow, Vector2 p_pos);
	Ref<Texture2D> get_arrow_texture(bool p_arrow);
	void update_internal_margin();
	Size2 get_largest_size() const;
	Color get_arrow_modulate(bool p_arrow) const;

protected:
	Size2 get_minimum_size() const override;
	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &property) const override;
	virtual void pressed() override;
	static void _bind_methods();
	void _on_left_pressed();
	void _on_right_pressed();
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void add_item(const String &p_item_name, int p_idx = -1);
	void remove_item(int p_idx);
	void set_item_text(int idx, const String &p_text);
	String get_item_text(int idx) const;
	void set_item_icon(int idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_item_icon(int idx) const;
	int get_item_count();
	void set_item_count(int p_count);
	void select(int p_idx);
	int get_selected();
	void set_wraparound(bool p_wraparound);
	bool get_wraparound();
	void set_fit_to_longest_item(bool p_fit_to_longest_item);
	bool get_fit_to_longest_item();
	CarouselButton();
};
