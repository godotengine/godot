/**************************************************************************/
/*  foldable_container.h                                                  */
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

#ifndef FOLDABLE_CONTAINER_H
#define FOLDABLE_CONTAINER_H

#include "scene/gui/container.h"
#include "scene/property_list_helper.h"
#include "scene/resources/text_line.h"

class FoldableGroup;

class FoldableContainer : public Container {
	GDCLASS(FoldableContainer, Container)

public:
	enum TitlePosition {
		POSITION_TOP,
		POSITION_BOTTOM,
		POSITION_MAX
	};

private:
	struct Button {
		Ref<Texture2D> icon = nullptr;
		String tooltip;
		Variant metadata;
		Rect2 rect;

		int id = -1;
		bool visible = true;
		bool disabled = false;
		bool auto_hide = false;
		bool toggle_mode = false;
		bool toggled_on = false;
	};

	static inline PropertyListHelper base_property_helper;
	PropertyListHelper property_helper;

	Vector<FoldableContainer::Button> buttons;
	Ref<FoldableGroup> foldable_group;
	bool changing_group = false;
	int _hovered = -1;
	int _pressed = -1;
	bool folded = false;
	String text;
	Ref<TextLine> text_buf;
	String language;
	TextDirection text_direction = TEXT_DIRECTION_AUTO;
	HorizontalAlignment text_alignment = HORIZONTAL_ALIGNMENT_LEFT;
	TextServer::OverrunBehavior overrun_behavior = TextServer::OVERRUN_NO_TRIMMING;
	TitlePosition title_position = POSITION_TOP;

	bool is_hovering = false;
	int title_panel_height = 0;

	struct ThemeCache {
		Ref<StyleBox> title_style;
		Ref<StyleBox> title_hover_style;
		Ref<StyleBox> title_collapsed_style;
		Ref<StyleBox> title_collapsed_hover_style;
		Ref<StyleBox> panel_style;
		Ref<StyleBox> focus_style;

		Ref<StyleBox> button_normal_style;
		Ref<StyleBox> button_hovered_style;
		Ref<StyleBox> button_pressed_style;
		Ref<StyleBox> button_disabled_style;

		Color button_icon_normal;
		Color button_icon_hovered;
		Color button_icon_pressed;
		Color button_icon_disabled;

		Color title_font_color;
		Color title_hovered_font_color;
		Color title_collapsed_font_color;
		Color title_font_outline_color;

		Ref<Font> title_font;
		int title_font_size = 0;
		int title_font_outline_size = 0;

		Ref<Texture2D> expanded_arrow;
		Ref<Texture2D> expanded_arrow_mirrored;
		Ref<Texture2D> folded_arrow;
		Ref<Texture2D> folded_arrow_mirrored;

		int h_separation = 0;
	} theme_cache;

	Ref<StyleBox> _get_title_style() const;
	Ref<Texture2D> _get_title_icon() const;
	Size2 _get_title_panel_min_size() const;
	void _shape();
	HorizontalAlignment _get_actual_alignment() const;
	void _update_group();

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual String get_tooltip(const Point2 &p_pos) const override;
	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const { return property_helper.property_get_value(p_name, r_ret); }
	void _get_property_list(List<PropertyInfo> *p_list) const { property_helper.get_property_list(p_list); }
	bool _property_can_revert(const StringName &p_name) const { return property_helper.property_can_revert(p_name); }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return property_helper.property_get_revert(p_name, r_property); }
	static void _bind_methods();

public:
	void fold();
	void expand();

	void set_folded(bool p_folded);
	bool is_folded() const;

	void set_expanded(bool p_expanded);
	bool is_expanded() const;

	void set_foldable_group(const Ref<FoldableGroup> &p_group);
	Ref<FoldableGroup> get_foldable_group() const;

	void set_text(const String &p_text);
	String get_text() const;

	void set_text_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_text_alignment() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;

	void set_language(const String &p_language);
	String get_language() const;

	void set_title_position(TitlePosition p_title_position);
	TitlePosition get_title_position() const;

	void add_button(const Ref<Texture2D> &p_icon = nullptr, int p_position = -1, int p_id = -1);
	void remove_button(int p_index);

	void set_button_count(int p_count);
	int get_button_count() const;

	Rect2 get_button_rect(int p_index) const;
	void clear();

	void set_button_id(int p_index, int p_id);
	int get_button_id(int p_index) const;

	int move_button(int p_from, int p_to);
	int get_button_index(int p_id) const;

	void set_button_toggle_mode(int p_index, bool p_mode);
	bool get_button_toggle_mode(int p_index) const;

	void set_button_toggled(int p_index, bool p_toggled_on);
	bool is_button_toggled(int p_index) const;

	void set_button_icon(int p_index, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_button_icon(int p_index) const;

	void set_button_tooltip(int p_index, String p_tooltip);
	String get_button_tooltip(int p_index) const;

	void set_button_disabled(int p_index, bool p_disabled);
	bool is_button_disabled(int p_index) const;

	void set_button_auto_hide(int p_index, bool p_auto_hide);
	bool is_button_auto_hide(int p_index) const;

	void set_button_visible(int p_index, bool p_visible);
	bool is_button_visible(int p_index) const;

	void set_button_metadata(int p_index, Variant p_metadata);
	Variant get_button_metadata(int p_index) const;

	int get_button_at_position(const Point2 &p_pos) const;

	virtual Size2 get_minimum_size() const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	FoldableContainer(const String &p_text = String());
	~FoldableContainer();
};

VARIANT_ENUM_CAST(FoldableContainer::TitlePosition);

class FoldableGroup : public Resource {
	GDCLASS(FoldableGroup, Resource);
	friend class FoldableContainer;
	HashSet<FoldableContainer *> containers;
	bool allow_folding_all = false;
	bool updating_group = false;

protected:
	static void _bind_methods();

public:
	FoldableContainer *get_expanded_container();
	void get_containers(List<FoldableContainer *> *r_containers);
	TypedArray<FoldableContainer> _get_containers();
	void set_allow_folding_all(bool p_enabled);
	bool is_allow_folding_all();
	FoldableGroup();
};

#endif // FOLDABLE_CONTAINER_H
