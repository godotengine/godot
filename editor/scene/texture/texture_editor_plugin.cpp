/**************************************************************************/
/*  texture_editor_plugin.cpp                                             */
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

#include "texture_editor_plugin.h"

#include "editor/editor_string_names.h"
#include "editor/scene/texture/color_channel_selector.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/animated_texture.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/compressed_texture.h"
#include "scene/resources/dpi_texture.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/portable_compressed_texture.h"

constexpr const char *texture_2d_shader_code = R"(
shader_type canvas_item;
render_mode blend_mix;

instance uniform vec4 u_channel_factors = vec4(1.0);
instance uniform float u_zoom = 1.0;
instance uniform vec2 u_pan = vec2(0.0);

vec4 filter_preview_colors(vec4 input_color, vec4 factors) {
	// Filter RGB.
	vec4 output_color = input_color * vec4(factors.rgb, input_color.a);

	// Remove transparency when alpha is not enabled.
	output_color.a = mix(1.0, output_color.a, factors.a);

	// Switch to opaque grayscale when visualizing only one channel.
	float csum = factors.r + factors.g + factors.b + factors.a;
	float single = clamp(2.0 - csum, 0.0, 1.0);
	for (int i = 0; i < 4; i++) {
		float c = input_color[i];
		output_color = mix(output_color, vec4(c, c, c, 1.0), factors[i] * single);
	}

	return output_color;
}

void fragment() {
	vec2 zoom_uv = (UV - 0.5) / u_zoom + 0.5 + u_pan;
	COLOR = filter_preview_colors(texture(TEXTURE, zoom_uv), u_channel_factors);
}
)";

void TexturePreview::init_shaders() {
	texture_material.instantiate();

	Ref<Shader> texture_shader;
	texture_shader.instantiate();
	texture_shader->set_code(texture_2d_shader_code);

	texture_material->set_shader(texture_shader);
}

void TexturePreview::finish_shaders() {
	texture_material.unref();
}

TextureRect *TexturePreview::get_texture_display() {
	return texture_display;
}

void TexturePreview::_texture_display_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::WHEEL_UP) {
			if (mb->is_pressed()) {
				on_zoom_in_pressed();
			}
		} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
			if (mb->is_pressed()) {
				on_zoom_out_pressed();
			}
		}

		if (mb->get_button_index() == MouseButton::LEFT && zoom_level != 1.0) {
			if (mb->is_pressed()) {
				panning = true;
				drag_start = pan + mb->get_position();
			} else {
				panning = false;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		// Mouse movement
		if (panning) {
			pan = drag_start - mm->get_position();
			_update_pan();
		}
	}
}

void TexturePreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (!is_inside_tree()) {
				// TODO: This is a workaround because `NOTIFICATION_THEME_CHANGED`
				// is getting called for some reason when the `TexturePreview` is
				// getting destroyed, which causes `get_theme_font()` to return `nullptr`.
				// See https://github.com/godotengine/godot/issues/50743.
				break;
			}

			if (metadata_label) {
				Ref<Font> metadata_label_font = get_theme_font(SNAME("expression"), EditorStringName(EditorFonts));
				metadata_label->add_theme_font_override(SceneStringName(font), metadata_label_font);
			}

			bg_rect->set_color(get_theme_color(SNAME("dark_color_2"), EditorStringName(Editor)));
			checkerboard->set_texture(get_editor_theme_icon(SNAME("Checkerboard")));
			theme_cache.outline_color = get_theme_color(SNAME("extra_border_color_1"), EditorStringName(Editor));

			zoom_out_button->set_button_icon(get_editor_theme_icon(SNAME("ZoomLess")));
			zoom_reset_button->set_button_icon(get_editor_theme_icon(SNAME("ZoomReset")));
			zoom_in_button->set_button_icon(get_editor_theme_icon(SNAME("ZoomMore")));
			if (popout_button) {
				popout_button->set_button_icon(get_editor_theme_icon(SNAME("DistractionFree")));
			}
		} break;
	}
}

Control::CursorShape TexturePreview::get_cursor_shape(const Point2 &p_pos) const {
	if (!Math::is_equal_approx(zoom_level, 1) && texture_display->get_rect().has_point(p_pos)) {
		return CursorShape::CURSOR_MOVE;
	}
	return CursorShape::CURSOR_ARROW;
}

void TexturePreview::_draw_outline() {
	const float outline_width = Math::round(EDSCALE);
	const Rect2 outline_rect = Rect2(Vector2(), outline_overlay->get_size()).grow(outline_width * 0.5);
	outline_overlay->draw_rect(outline_rect, theme_cache.outline_color, false, outline_width);
}

void TexturePreview::_update_texture_display_ratio() {
	if (texture_display->get_texture().is_valid()) {
		centering_container->set_ratio(texture_display->get_texture()->get_size().aspect());
	}
}

static Image::Format get_texture_2d_format(const Ref<Texture2D> &p_texture) {
	const Ref<ImageTexture> image_texture = p_texture;
	if (image_texture.is_valid()) {
		return image_texture->get_format();
	}

	const Ref<CompressedTexture2D> compressed_texture = p_texture;
	if (compressed_texture.is_valid()) {
		return compressed_texture->get_format();
	}

	const Ref<PortableCompressedTexture2D> portable_compressed_texture = p_texture;
	if (portable_compressed_texture.is_valid()) {
		return portable_compressed_texture->get_format();
	}

	// AtlasTexture?

	// Unknown
	return Image::FORMAT_MAX;
}

static int get_texture_mipmaps_count(const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND_V(p_texture.is_null(), -1);

	// We are having to download the image only to get its mipmaps count. It would be nice if we didn't have to.
	Ref<Image> image;
	Ref<AtlasTexture> at = p_texture;
	if (at.is_valid()) {
		// The AtlasTexture tries to obtain the region from the atlas as an image,
		// which will fail if it is a compressed format.
		Ref<Texture2D> atlas = at->get_atlas();
		if (atlas.is_valid()) {
			image = atlas->get_image();
		}
	} else {
		image = p_texture->get_image();
	}

	if (image.is_valid()) {
		return image->get_mipmap_count();
	}
	return -1;
}

void TexturePreview::_update_metadata_label_text() {
	const Ref<Texture2D> texture = texture_display->get_texture();
	ERR_FAIL_COND(texture.is_null());

	const Image::Format format = get_texture_2d_format(texture.ptr());

	const String format_name = format != Image::FORMAT_MAX ? Image::get_format_name(format) : texture->get_class();

	const Vector2i resolution = texture->get_size();
	const int mipmaps = get_texture_mipmaps_count(texture);

	if (format != Image::FORMAT_MAX) {
		// Avoid signed integer overflow that could occur with huge texture sizes by casting everything to uint64_t.
		uint64_t memory = uint64_t(resolution.x) * uint64_t(resolution.y) * uint64_t(Image::get_format_pixel_size(format));
		// Handle VRAM-compressed formats that are stored with 4 bpp.
		memory >>= Image::get_format_pixel_rshift(format);

		float mipmaps_multiplier = 1.0;
		float mipmap_increase = 0.25;
		for (int i = 0; i < mipmaps; i++) {
			// Each mip adds 25% memory usage of the previous one.
			// With a complete mipmap chain, memory usage increases by ~33%.
			mipmaps_multiplier += mipmap_increase;
			mipmap_increase *= 0.25;
		}
		memory *= mipmaps_multiplier;

		if (mipmaps >= 1) {
			metadata_label->set_text(
					vformat(String::utf8("%d×%d %s\n") + TTR("%s Mipmaps") + "\n" + TTR("Memory: %s"),
							texture->get_width(),
							texture->get_height(),
							format_name,
							mipmaps,
							String::humanize_size(memory)));
		} else {
			// "No Mipmaps" is easier to distinguish than "0 Mipmaps",
			// especially since 0, 6, and 8 look quite close with the default code font.
			metadata_label->set_text(
					vformat(String::utf8("%d×%d %s\n") + TTR("No Mipmaps") + "\n" + TTR("Memory: %s"),
							texture->get_width(),
							texture->get_height(),
							format_name,
							String::humanize_size(memory)));
		}
	} else {
		metadata_label->set_text(
				vformat(String::utf8("%d×%d %s"),
						texture->get_width(),
						texture->get_height(),
						format_name));
	}
}

void TexturePreview::_update_pan() {
	float zoom_clamp = (0.5 - 0.5 / zoom_level) * texture_display->get_size().height;
	pan.x = CLAMP(pan.x, -zoom_clamp, zoom_clamp);
	pan.y = CLAMP(pan.y, -zoom_clamp, zoom_clamp);
	texture_display->set_instance_shader_parameter("u_pan", pan / texture_display->get_size().height);
}

void TexturePreview::on_selected_channels_changed() {
	texture_display->set_instance_shader_parameter("u_channel_factors", channel_selector->get_selected_channel_factors());
}

void TexturePreview::on_popout_pressed() {
	AcceptDialog *popout_dialog = memnew(AcceptDialog);
	popout_dialog->set_title("Texture Preview");

	Vector2 screen_size = get_tree()->get_root()->get_size();
	Vector2 popout_position = Vector2();
	Vector2 popout_size = MAX(Vector2(400, 300), screen_size * 0.5);
	if (EditorSettings::get_singleton()->has_setting("interface/editor/thumnail_window_size")) {
		popout_size = EditorSettings::get_singleton()->get_setting("interface/editor/thumnail_window_size");
	}
	if (EditorSettings::get_singleton()->has_setting("interface/editor/thumnail_window_position")) {
		popout_position = EditorSettings::get_singleton()->get_setting("interface/editor/thumnail_window_position");
	}

	add_child(popout_dialog);
	TexturePreview *texture_preview_copy = memnew(TexturePreview(texture_display->get_texture(), true, true));
	popout_dialog->add_child(texture_preview_copy);

	popout_dialog->connect("canceled", callable_mp(this, &TexturePreview::on_popout_closed).bind(popout_dialog));
	popout_dialog->connect("confirmed", callable_mp(this, &TexturePreview::on_popout_closed).bind(popout_dialog));
	popout_dialog->popup_centered(popout_size);
	if (popout_position != Vector2()) {
		popout_dialog->set_position(popout_position);
	}
}

void TexturePreview::on_popout_closed(AcceptDialog *p_dialog) {
	if (p_dialog) {
		EditorSettings::get_singleton()->set_setting("interface/editor/thumnail_window_size", p_dialog->get_size());
		EditorSettings::get_singleton()->set_setting("interface/editor/thumnail_window_position", p_dialog->get_position());
		EditorSettings::get_singleton()->save();
		p_dialog->queue_free();
	}
}

void TexturePreview::on_zoom_out_pressed() {
	zoom_level = CLAMP(zoom_level - 0.25, 1, 8);
	_update_pan();
	texture_display->set_instance_shader_parameter("u_zoom", zoom_level);
}

void TexturePreview::on_zoom_reset_pressed() {
	zoom_level = 1;
	_update_pan();
	texture_display->set_instance_shader_parameter("u_zoom", zoom_level);
}

void TexturePreview::on_zoom_in_pressed() {
	zoom_level = CLAMP(zoom_level + 0.25, 1, 8);
	texture_display->set_instance_shader_parameter("u_zoom", zoom_level);
}

TexturePreview::TexturePreview(Ref<Texture2D> p_texture, bool p_show_metadata, bool p_popout) {
	set_custom_minimum_size(Size2(0.0, 256.0) * EDSCALE);

	bg_rect = memnew(ColorRect);

	add_child(bg_rect);

	margin_container = memnew(MarginContainer);
	const float outline_width = Math::round(EDSCALE);
	margin_container->add_theme_constant_override("margin_right", outline_width);
	margin_container->add_theme_constant_override("margin_top", outline_width);
	margin_container->add_theme_constant_override("margin_left", outline_width);
	margin_container->add_theme_constant_override("margin_bottom", outline_width);
	add_child(margin_container);

	centering_container = memnew(AspectRatioContainer);
	margin_container->add_child(centering_container);

	checkerboard = memnew(TextureRect);
	checkerboard->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	checkerboard->set_stretch_mode(TextureRect::STRETCH_TILE);
	checkerboard->set_texture_repeat(CanvasItem::TEXTURE_REPEAT_ENABLED);
	centering_container->add_child(checkerboard);

	texture_display = memnew(TextureRect);
	texture_display->set_texture_filter(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	texture_display->set_texture(p_texture);
	texture_display->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	texture_display->set_material(texture_material);
	texture_display->set_instance_shader_parameter("u_channel_factors", Vector4(1, 1, 1, 1));
	texture_display->connect("gui_input", callable_mp(this, &TexturePreview::_texture_display_gui_input));
	centering_container->add_child(texture_display);

	// Creating a separate control so it is not affected by the filtering shader.
	outline_overlay = memnew(Control);
	outline_overlay->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	centering_container->add_child(outline_overlay);

	outline_overlay->connect(SceneStringName(draw), callable_mp(this, &TexturePreview::_draw_outline));

	if (p_texture.is_valid()) {
		_update_texture_display_ratio();
		p_texture->connect_changed(callable_mp(this, &TexturePreview::_update_texture_display_ratio));
	}

	// Null can be passed by `Camera3DPreview` (which immediately after sets a texture anyways).
	const Image::Format format = p_texture.is_valid() ? get_texture_2d_format(p_texture.ptr()) : Image::FORMAT_MAX;
	const uint32_t components_mask = format != Image::FORMAT_MAX ? Image::get_format_component_mask(format) : 0xf;

	// Add color channel selector at the bottom left if more than 1 channel is available.
	if (p_show_metadata && !is_power_of_2(components_mask)) {
		channel_selector = memnew(ColorChannelSelector);
		channel_selector->connect("selected_channels_changed", callable_mp(this, &TexturePreview::on_selected_channels_changed));
		channel_selector->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
		channel_selector->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
		channel_selector->set_available_channels_mask(components_mask);
		add_child(channel_selector);
	}

	// Add right upper buttons
	right_upper_corner_container = memnew(HBoxContainer);
	right_upper_corner_container->set_h_size_flags(Control::SIZE_SHRINK_END);
	right_upper_corner_container->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	add_child(right_upper_corner_container);
	zoom_out_button = memnew(Button);
	zoom_out_button->connect("pressed", callable_mp(this, &TexturePreview::on_zoom_out_pressed));
	zoom_out_button->set_flat(true);
	zoom_reset_button = memnew(Button);
	zoom_reset_button->connect("pressed", callable_mp(this, &TexturePreview::on_zoom_reset_pressed));
	zoom_reset_button->set_flat(true);
	zoom_in_button = memnew(Button);
	zoom_in_button->connect("pressed", callable_mp(this, &TexturePreview::on_zoom_in_pressed));
	zoom_in_button->set_flat(true);
	right_upper_corner_container->add_child(zoom_out_button);
	right_upper_corner_container->add_child(zoom_reset_button);
	right_upper_corner_container->add_child(zoom_in_button);

	// Add a button to the upper right corner that opens a popup window with a copy of the preview.
	if (!p_popout) {
		popout_button = memnew(Button);
		popout_button->connect("pressed", callable_mp(this, &TexturePreview::on_popout_pressed));
		popout_button->set_tooltip_text(TTRC("Open the preview in a separate window."));
		popout_button->set_flat(true);
		popout_button->set_h_size_flags(Control::SIZE_SHRINK_END);
		popout_button->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
		right_upper_corner_container->add_child(popout_button);
	}

	if (p_show_metadata) {
		metadata_label = memnew(Label);
		metadata_label->set_focus_mode(FOCUS_ACCESSIBILITY);

		if (p_texture.is_valid()) {
			_update_metadata_label_text();
			p_texture->connect_changed(callable_mp(this, &TexturePreview::_update_metadata_label_text));
		}

		// It's okay that these colors are static since the grid color is static too.
		metadata_label->add_theme_color_override(SceneStringName(font_color), Color(1, 1, 1));
		metadata_label->add_theme_color_override("font_shadow_color", Color(0, 0, 0));

		metadata_label->add_theme_font_size_override(SceneStringName(font_size), 14 * EDSCALE);
		metadata_label->add_theme_color_override("font_outline_color", Color(0, 0, 0));
		metadata_label->add_theme_constant_override("outline_size", 8 * EDSCALE);

		metadata_label->set_h_size_flags(Control::SIZE_SHRINK_END);
		metadata_label->set_v_size_flags(Control::SIZE_SHRINK_END);

		add_child(metadata_label);
	}
}

bool EditorInspectorPluginTexture::can_handle(Object *p_object) {
	return Object::cast_to<ImageTexture>(p_object) != nullptr || Object::cast_to<AtlasTexture>(p_object) != nullptr || Object::cast_to<CompressedTexture2D>(p_object) != nullptr || Object::cast_to<PortableCompressedTexture2D>(p_object) != nullptr || Object::cast_to<AnimatedTexture>(p_object) != nullptr || Object::cast_to<DPITexture>(p_object) != nullptr || Object::cast_to<Image>(p_object) != nullptr;
}

void EditorInspectorPluginTexture::parse_begin(Object *p_object) {
	Ref<Texture> texture(Object::cast_to<Texture>(p_object));
	if (texture.is_null()) {
		Ref<Image> image(Object::cast_to<Image>(p_object));
		texture = ImageTexture::create_from_image(image);

		ERR_FAIL_COND_MSG(texture.is_null(), "Failed to create the texture from an invalid image.");
	}

	add_custom_control(memnew(TexturePreview(texture, true, false)));
}

TextureEditorPlugin::TextureEditorPlugin() {
	Ref<EditorInspectorPluginTexture> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
