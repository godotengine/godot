/**************************************************************************/
/*  lightmap_gi_editor_plugin.cpp                                         */
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

#include "lightmap_gi_editor_plugin.h"

#include "core/input/input_event.h"
#include "core/io/resource_loader.h"
#include "core/math/transform_2d.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/string/node_path.h"
#include "core/string/translation_server.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/scene/3d/mesh_instance_3d_editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_viewport.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/lightmap_gi.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_button.h"
#include "scene/gui/control.h"
#include "scene/gui/item_list.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/texture.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

#include "modules/modules_enabled.gen.h" // For lightmapper_rd.

void LightmapGIBakeEditor::_bake_select_file(const String &p_file) {
	if (lightmap) {
		LightmapGI::BakeError err = LightmapGI::BAKE_ERROR_OK;
		const uint64_t time_started = OS::get_singleton()->get_ticks_msec();
		if (get_tree()->get_edited_scene_root()) {
			Ref<LightmapGIData> lightmapGIData = lightmap->get_light_data();

			if (lightmapGIData.is_valid()) {
				String path = lightmapGIData->get_path();
				if (!path.is_resource_file()) {
					int srpos = path.find("::");
					if (srpos != -1) {
						String base = path.substr(0, srpos);
						if (ResourceLoader::get_resource_type(base) == "PackedScene") {
							if (!get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->get_scene_file_path() != base) {
								err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
							}
						} else {
							if (FileAccess::exists(base + ".import")) {
								err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
							}
						}
					}
				} else {
					if (FileAccess::exists(path + ".import")) {
						err = LightmapGI::BAKE_ERROR_FOREIGN_DATA;
					}
				}
			}

			if (err == LightmapGI::BAKE_ERROR_OK) {
				if (get_tree()->get_edited_scene_root() == lightmap) {
					err = lightmap->bake(lightmap, p_file, bake_func_step);
				} else {
					err = lightmap->bake(lightmap->get_parent(), p_file, bake_func_step);
				}
			}
		} else {
			err = LightmapGI::BAKE_ERROR_NO_SCENE_ROOT;
		}

		bake_func_end(time_started);

		switch (err) {
			case LightmapGI::BAKE_ERROR_NO_SAVE_PATH: {
				String scene_path = lightmap->get_scene_file_path();
				if (scene_path.is_empty() && lightmap->get_owner()) {
					scene_path = lightmap->get_owner()->get_scene_file_path();
				}
				if (scene_path.is_empty()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for lightmap images.\nSave your scene and try again."));
					break;
				}
				scene_path = scene_path.get_basename() + ".lmbake";

				file_dialog->set_current_path(scene_path);
				file_dialog->popup_file_dialog();
			} break;
			case LightmapGI::BAKE_ERROR_NO_MESHES: {
				EditorNode::get_singleton()->show_warning(
						TTR("No meshes with lightmapping support to bake. Make sure they contain UV2 data and their Global Illumination property is set to Static.") +
						String::utf8("\n\n•  ") + TTR("To import a scene with lightmapping support, set Meshes > Light Baking to Static Lightmaps in the Import dock.") +
						String::utf8("\n•  ") + TTR("To enable lightmapping support on a primitive mesh, edit the PrimitiveMesh resource in the inspector and check Add UV2.") +
						String::utf8("\n•  ") + TTR("To enable lightmapping support on a CSG mesh, select the root CSG node and choose CSG > Bake Mesh Instance at the top of the 3D editor viewport.\nSelect the generated MeshInstance3D node and choose Mesh > Unwrap UV2 for Lightmap/AO at the top of the 3D editor viewport."));
			} break;
			case LightmapGI::BAKE_ERROR_CANT_CREATE_IMAGE: {
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images. Make sure the lightmap destination path is writable."));
			} break;
			case LightmapGI::BAKE_ERROR_NO_SCENE_ROOT: {
				EditorNode::get_singleton()->show_warning(TTR("No editor scene root found."));
			} break;
			case LightmapGI::BAKE_ERROR_FOREIGN_DATA: {
				EditorNode::get_singleton()->show_warning(TTR("Lightmap data is not local to the scene."));
			} break;
			case LightmapGI::BAKE_ERROR_TEXTURE_SIZE_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Maximum texture size is too small for the lightmap images.\nWhile this can be fixed by increasing the maximum texture size, it is recommended you split the scene into more objects instead."));
			} break;
			case LightmapGI::BAKE_ERROR_LIGHTMAP_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images. Make sure all meshes to bake have the Lightmap Size Hint property set high enough, and the LightmapGI's Texel Scale value is not too low."));
			} break;
			case LightmapGI::BAKE_ERROR_ATLAS_TOO_SMALL: {
				EditorNode::get_singleton()->show_warning(TTR("Failed fitting a lightmap image into an atlas. This should never happen and should be reported."));
			} break;
			default: {
			} break;
		}
	}
}

void LightmapGIBakeEditor::_bake() {
	_bake_select_file("");
}

EditorProgress *LightmapGIBakeEditor::tmp_progress = nullptr;

bool LightmapGIBakeEditor::bake_func_step(float p_progress, const String &p_description, void *, bool p_refresh) {
	if (!tmp_progress) {
		tmp_progress = memnew(EditorProgress("bake_lightmaps", TTR("Bake Lightmaps"), 1000, true));
		ERR_FAIL_NULL_V(tmp_progress, false);
	}
	return tmp_progress->step(p_description, p_progress * 1000, p_refresh);
}

void LightmapGIBakeEditor::bake_func_end(uint64_t p_time_started) {
	if (tmp_progress != nullptr) {
		memdelete(tmp_progress);
		tmp_progress = nullptr;
	}

	const int time_taken = OS::get_singleton()->get_ticks_msec() - p_time_started;
	print_line(vformat("Done baking lightmaps in %02d:%02d:%02d.%02d.", time_taken / 3'600'000, (time_taken % 3'600'000) / 60'000, (time_taken % 60'000) / 1000, (time_taken % 1000) / 10));
	// Request attention in case the user was doing something else.
	// Baking lightmaps is likely the editor task that can take the most time,
	// so only request the attention for baking lightmaps.
	DisplayServer::get_singleton()->window_request_attention();
}

void LightmapGIBakeEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			bake->set_button_icon(get_editor_theme_icon(SNAME("Bake")));
		} break;
	}
}

void LightmapGIBakeEditor::_bind_methods() {
	ClassDB::bind_method("_bake", &LightmapGIBakeEditor::_bake);
}

void LightmapGIBakeEditor::edit(LightmapGI *p_lightmap) {
	lightmap = p_lightmap;
	if (lightmap) {
		bake->show();
	} else {
		bake->hide();
	}
}

LightmapGIBakeEditor::LightmapGIBakeEditor() {
	bake = memnew(Button);
	bake->set_theme_type_variation(SceneStringName(FlatButton));
	bake->set_text(TTR("Bake Lightmaps"));

#ifdef MODULE_LIGHTMAPPER_RD_ENABLED
	// Disable lightmap baking if not supported on the current GPU.
	if (!DisplayServer::get_singleton()->can_create_rendering_device()) {
		bake->set_disabled(true);
		bake->set_tooltip_text(vformat(TTR("Lightmap baking is not supported on this GPU (%s)."), RenderingServer::get_singleton()->get_video_adapter_name()));
	}
#else
	// Disable lightmap baking if the module is disabled at compile-time.
	bake->set_disabled(true);
#if defined(ANDROID_ENABLED) || defined(APPLE_EMBEDDED_ENABLED)
	bake->set_tooltip_text(vformat(TTR("Lightmaps cannot be baked on %s."), OS::get_singleton()->get_name()));
#else
	bake->set_tooltip_text(TTR("Lightmaps cannot be baked, as the `lightmapper_rd` module was disabled at compile-time."));
#endif
#endif // MODULE_LIGHTMAPPER_RD_ENABLED

	bake->hide();
	bake->connect(SceneStringName(pressed), Callable(this, "_bake"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(bake);
	lightmap = nullptr;

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->add_filter("*.lmbake", TTR("LightMap Bake"));
	file_dialog->set_title(TTR("Select lightmap bake file:"));
	file_dialog->connect("file_selected", callable_mp(this, &LightmapGIBakeEditor::_bake_select_file));
	bake->add_child(file_dialog);
}

///////////////////////////////////////////

void LightmapGITextureEditor::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("editors/panning")) {
				panner->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/sub_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
				panner->setup_warped_panning(this, EDITOR_GET("editors/panning/warped_mouse_panning"));
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			panner->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/sub_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
			panner->setup_warped_panning(this, EDITOR_GET("editors/panning/warped_mouse_panning"));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			texture_preview->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("TextureRegionPreviewBG"), EditorStringName(EditorStyles)));
			texture_overlay->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("TextureRegionPreviewFG"), EditorStringName(EditorStyles)));

			zoom_out->set_button_icon(get_editor_theme_icon(SNAME("ZoomLess")));
			zoom_in->set_button_icon(get_editor_theme_icon(SNAME("ZoomMore")));
			zoom_to_fit->set_button_icon(get_editor_theme_icon(SNAME("DistractionFree")));
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			// This happens when the user leaves the Editor and returns,
			// they could have changed the textures, so the cache is cleared.
			// cache_map.clear();
			// _edit_region();
		} break;
	}
}

bool LightmapGITextureEditor::_update_current_texture(int p_layer, bool p_shadowmask) {
	p_layer = CLAMP(p_layer, 0, total_layers - 1);

	if (current_layer == p_layer && show_shadowmask == p_shadowmask) {
		return false;
	}

	current_layer = p_layer;
	show_shadowmask = p_shadowmask;

	if (lightmap_data.is_valid()) {
		TypedArray<TextureLayered> textures = show_shadowmask ? lightmap_data->get_shadowmask_textures() : lightmap_data->get_lightmap_textures();

		int subtotal = 0;
		for (Ref<TextureLayered> lightmap_texture : textures) {
			if (lightmap_texture.is_null()) {
				continue;
			}
			if (lightmap_texture->get_layers() < current_layer) {
				subtotal += lightmap_texture->get_layers();
				continue;
			}
			current_texture = ImageTexture::create_from_image(lightmap_texture->get_layer_data(current_layer - subtotal));
			return true;
		}
	}

	current_texture = Ref<Texture2D>();
	return true;
}

Transform2D LightmapGITextureEditor::_get_offset_transform() const {
	Transform2D mtx;
	mtx.columns[2] = -draw_ofs * draw_zoom;
	mtx.scale_basis(Vector2(draw_zoom, draw_zoom));

	return mtx;
}

void LightmapGITextureEditor::_texture_preview_draw() {
	if (current_texture.is_null()) {
		return;
	}

	Transform2D mtx = _get_offset_transform();

	texture_preview->draw_set_transform_matrix(mtx);

	texture_preview->draw_rect(Rect2(Point2(), current_texture->get_size()), Color(0.5, 0.5, 0.5, 0.5), false);
	texture_preview->draw_texture(current_texture, Point2());

	texture_preview->draw_set_transform_matrix(Transform2D());
}

void LightmapGITextureEditor::_texture_overlay_draw() {
	if (current_texture.is_null()) {
		return;
	}

	Rect2 scroll_rect(Point2(), current_texture->get_size());
	const Size2 scroll_margin = texture_overlay->get_size() / draw_zoom;
	scroll_rect.position -= scroll_margin;
	scroll_rect.size += scroll_margin * 2;

	updating_scroll = true;

	hscroll->set_min(scroll_rect.position.x);
	hscroll->set_max(scroll_rect.position.x + scroll_rect.size.x);
	if (Math::abs(scroll_rect.position.x - (scroll_rect.position.x + scroll_rect.size.x)) <= scroll_margin.x) {
		hscroll->hide();
	} else {
		hscroll->show();
		hscroll->set_page(scroll_margin.x);
		hscroll->set_value(draw_ofs.x);
	}

	vscroll->set_min(scroll_rect.position.y);
	vscroll->set_max(scroll_rect.position.y + scroll_rect.size.y);
	if (Math::abs(scroll_rect.position.y - (scroll_rect.position.y + scroll_rect.size.y)) <= scroll_margin.y) {
		vscroll->hide();
		draw_ofs.y = scroll_rect.position.y;
	} else {
		vscroll->show();
		vscroll->set_page(scroll_margin.y);
		vscroll->set_value(draw_ofs.y);
	}

	Size2 hmin = hscroll->get_combined_minimum_size();
	Size2 vmin = vscroll->get_combined_minimum_size();

	// Avoid scrollbar overlapping.
	hscroll->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, vscroll->is_visible() ? -vmin.width : 0);
	vscroll->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, hscroll->is_visible() ? -hmin.height : 0);

	updating_scroll = false;

	if (request_center && hscroll->get_min() < 0) {
		hscroll->set_value((hscroll->get_min() + hscroll->get_max() - hscroll->get_page()) / 2);
		vscroll->set_value((vscroll->get_min() + vscroll->get_max() - vscroll->get_page()) / 2);
		// This ensures that the view is updated correctly.
		callable_mp(this, &LightmapGITextureEditor::_pan_callback).call_deferred(Vector2(1, 0), Ref<InputEvent>());
		callable_mp(this, &LightmapGITextureEditor::_scroll_changed).call_deferred(0.0);
		request_center = false;
	}

	Transform2D mtx = _get_offset_transform();

	for (UserEntry entry : users_in_list) {
		if (entry.slice != current_layer) {
			continue;
		}

		int user_idx = entry.user_idx;
		Rect2 uv_rect = _get_user_uv_rect(user_idx);
		if (show_uv && (selected_user.user_idx == user_idx || selected_user.user_idx == -1)) {
			Node3D *node = Object::cast_to<Node3D>(lightmap->get_node_or_null(lightmap_data->get_user_path(user_idx)));
			if (node) {
				Ref<Mesh> mesh;
				MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(node);
				if (mi) {
					mesh = mi->get_mesh();
				} else if (node->has_method("get_bake_meshes")) {
					// get_bake_meshes() returns a flat array interleaving Mesh/Transform3D values.
					int idx = lightmap_data->get_user_sub_instance(user_idx) * 2;
					if (idx > -1) {
						Array bake_meshes = node->call("get_bake_meshes");
						mesh = bake_meshes.get(idx);
					}
				}

				if (mesh.is_valid()) {
					Color uv_color = get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.5);
					PackedVector2Array uv_lines;
					Error result = MeshInstance3DEditor::get_uv_lines(mesh, 1, &uv_lines);
					if (result == Error::OK) {
						Transform2D uv_mtx = mtx * Transform2D(0, uv_rect.size, 0, uv_rect.position);
						texture_overlay->draw_set_transform_matrix(uv_mtx);
						texture_overlay->draw_multiline(uv_lines, uv_color);
					}
				}
			}
		}
		texture_overlay->draw_set_transform_matrix(mtx);

		Color color = Color(0.5, 0.5, 0.5, 0.5);
		if (selected_user.user_idx == user_idx) {
			color = Color(1.0, 0.0, 1.0);
		}
		texture_overlay->draw_rect(uv_rect, color, false);
	}
	texture_overlay->draw_set_transform_matrix(Transform2D());
}

void LightmapGITextureEditor::_texture_overlay_input(const Ref<InputEvent> &p_input) {
	if (panner->gui_input(p_input, texture_overlay->get_global_rect())) {
		return;
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_input;
	if (magnify_gesture.is_valid()) {
		_zoom_on_position(draw_zoom * magnify_gesture->get_factor(), magnify_gesture->get_position());
	}

	Ref<InputEventPanGesture> pan_gesture = p_input;
	if (pan_gesture.is_valid()) {
		hscroll->set_value(hscroll->get_value() + hscroll->get_page() * pan_gesture->get_delta().x / 8);
		vscroll->set_value(vscroll->get_value() + vscroll->get_page() * pan_gesture->get_delta().y / 8);
	}

	Ref<InputEventMouseButton> mb = p_input;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed() && !panner->is_panning()) {
				Transform2D mtx = _get_offset_transform();
				Vector2 point = mtx.affine_inverse().xform(mb->get_position());
				for (UserEntry user : users_in_list) {
					Rect2 uv_rect = _get_user_uv_rect(user.user_idx);
					if (uv_rect.has_point(point)) {
						_select_user(user, mb->is_double_click() ? FOCUS : SELECT);
						break;
					}
				}
			}
		}
	}
}

void LightmapGITextureEditor::_pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event) {
	p_scroll_vec /= draw_zoom;
	hscroll->set_value(hscroll->get_value() - p_scroll_vec.x);
	vscroll->set_value(vscroll->get_value() - p_scroll_vec.y);
}

void LightmapGITextureEditor::_zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event) {
	_zoom_on_position(draw_zoom * p_zoom_factor, p_origin);
}

void LightmapGITextureEditor::_zoom_in() {
	_zoom_on_position(draw_zoom * 1.5, texture_overlay->get_size() / 2.0);
}

void LightmapGITextureEditor::_zoom_reset() {
	_zoom_on_position(1.0, texture_overlay->get_size() / 2.0);
}

void LightmapGITextureEditor::_zoom_out() {
	_zoom_on_position(draw_zoom / 1.5, texture_overlay->get_size() / 2.0);
}

float LightmapGITextureEditor::_get_zoom_to_fit(const Point2 p_size) {
	const float margin_percentage = 0.1f;
	const float max_margin = 16.0f;
	const Size2 margin = (margin_percentage * texture_overlay->get_size()).minf(max_margin);
	const Size2 display_area_size = texture_overlay->get_size() - margin;
	const Vector2 ratio = display_area_size / p_size;
	return MIN(ratio.x, ratio.y);
}

void LightmapGITextureEditor::_zoom_to_fit() {
	float texture_fit_zoom = _get_zoom_to_fit(current_texture->get_size());

	request_center = true;

	_zoom_on_position(texture_fit_zoom, texture_overlay->get_size() / 2.0);
}

void LightmapGITextureEditor::_zoom_on_position(float p_zoom, Point2 p_position) {
	if (p_zoom < min_draw_zoom || p_zoom > max_draw_zoom) {
		return;
	}

	float prev_zoom = draw_zoom;
	draw_zoom = p_zoom;
	Point2 ofs = p_position;
	ofs = ofs / prev_zoom - ofs / draw_zoom;
	draw_ofs = (draw_ofs + ofs).round();

	texture_preview->queue_redraw();
	texture_overlay->queue_redraw();

	_update_zoom_label();
}

void LightmapGITextureEditor::_update_zoom_label() {
	String zoom_text;
	// The zoom level displayed is relative to the editor scale
	// (like in most image editors). Its lower bound is clamped to 1 as some people
	// lower the editor scale to increase the available real estate,
	// even if their display doesn't have a particularly low DPI.
	TranslationServer *translation_server = TranslationServer::get_singleton();
	String locale = translation_server->get_tool_locale();
	if (draw_zoom >= 10) {
		zoom_text = translation_server->format_number(rtos(Math::round((draw_zoom / MAX(1, EDSCALE)) * 100)), locale);
	} else {
		// 2 decimal places if the zoom is below 10%, 1 decimal place if it's below 1000%.
		zoom_text = translation_server->format_number(rtos(Math::snapped((draw_zoom / MAX(1, EDSCALE)) * 100, (draw_zoom >= 0.1) ? 0.1 : 0.01)), locale);
	}
	zoom_text += " " + translation_server->get_percent_sign(locale);
	zoom_reset->set_text(zoom_text);
}

void LightmapGITextureEditor::_scroll_changed(float) {
	if (updating_scroll) {
		return;
	}

	draw_ofs.x = hscroll->get_value();
	draw_ofs.y = vscroll->get_value();

	texture_preview->queue_redraw();
	texture_overlay->queue_redraw();
}

void LightmapGITextureEditor::_set_layer(int p_layer) {
	_update_current_texture(p_layer, show_shadowmask);

	_update_list();

	UserEntry entry;
	entry.slice = current_layer;
	_select_user(entry, NONE);
}

void LightmapGITextureEditor::_show_shadowmask_pressed() {
	_update_current_texture(current_layer, cb_show_shadowmask->is_pressed());

	texture_preview->queue_redraw();
	texture_overlay->queue_redraw();
}

void LightmapGITextureEditor::_show_uv_pressed() {
	show_uv = cb_show_uv->is_pressed();

	texture_preview->queue_redraw();
	texture_overlay->queue_redraw();
}

void LightmapGITextureEditor::_update_list() {
	user_list->clear();
	users_in_list.clear();

	if (lightmap_data.is_valid()) {
		for (int user_idx = 0; user_idx < lightmap_data->get_user_count(); ++user_idx) {
			if (lightmap_data->get_user_lightmap_slice_index(user_idx) == current_layer) {
				UserEntry entry;
				entry.slice = current_layer;
				entry.user_idx = user_idx;
				entry.subinstance = lightmap_data->get_user_sub_instance(user_idx);
				users_in_list.push_back(entry);
				user_list->add_item(vformat(TTR("Island %s"), user_idx));
			}
		}
	}
}

void LightmapGITextureEditor::_select_user(UserEntry p_user, SelectMode p_select_mode) {
	selected_user = p_user;

	if (selected_user.user_idx < 0) {
		// Only changed layer
		lbl_islands_title->set_text(TTR("Islands in layer:"));
		_update_current_texture(current_layer, show_shadowmask);
		sb_layer->set_value_no_signal(current_layer);
	} else {
		//
		for (int i = 0; i < users_in_list.size(); ++i) {
			UserEntry entry = users_in_list[i];
			if (entry.user_idx == selected_user.user_idx) {
				if (p_select_mode != NONE) {
					NodePath path = lightmap_data->get_user_path(selected_user.user_idx);
					Node *node = lightmap->get_node_or_null(path);
					if (selected_node != node) {
						Node *root = get_tree()->get_edited_scene_root();
						if (node == root || node->get_owner() == root || root->is_editable_instance(node->get_owner())) {
							// The node is visible in the scene tree dock so select it directly
						} else {
							// The node is part of an instanced scene and not editable so select the instanced scene itself
							while (node && node->get_owner() != root) {
								node = node->get_parent();
							}
						}
						if (node) {
							internal_selection = true;
							// Can't use EditorNode::get_singleton()->edit_node(node) because it triggers LightmapGITextureEditor::edit() twice:
							// once directly and once by the scene tree dock selection_changed signal
							EditorSelection *editor_selection = EditorNode::get_singleton()->get_editor_selection();
							editor_selection->clear();
							editor_selection->add_node(node);
						}
					}
					if (p_select_mode == FOCUS) {
						EditorData &editor_data = EditorNode::get_editor_data();
						Node3DEditorPlugin *editor = Object::cast_to<Node3DEditorPlugin>(editor_data.get_editor_by_name("3D"));
						editor->get_spatial_editor()->get_editor_viewport(0)->focus_selection();
					}
				}
				_update_current_texture(entry.slice, show_shadowmask);
				sb_layer->set_value_no_signal(current_layer);
				user_list->select(i);
				user_list->ensure_current_is_visible();
				break;
			}
		}
	}

	texture_preview->queue_redraw();
	texture_overlay->queue_redraw();

	_zoom_to_fit();
}

Rect2 LightmapGITextureEditor::_get_user_uv_rect(int p_user) {
	if (lightmap_data.is_null()) {
		return Rect2();
	}
	Rect2 uv_rect = lightmap_data->get_user_lightmap_uv_scale(p_user);
	uv_rect.position *= current_texture->get_size();
	uv_rect.size *= current_texture->get_size();

	return uv_rect;
}

void LightmapGITextureEditor::_on_user_list_item_selected(int p_index) {
	UserEntry entry = users_in_list.get(p_index);
	_select_user(entry, SELECT);
}

void LightmapGITextureEditor::_on_user_list_item_activated(int p_index) {
	UserEntry entry = users_in_list.get(p_index);
	_select_user(entry, FOCUS);
}

void LightmapGITextureEditor::edit(LightmapGI *p_lightmap, Node3D *p_selected_node, int user_idx) {
	if (internal_selection) {
		clear_internal_selection();
		return;
	}

	if (lightmap != p_lightmap) {
		lightmap = p_lightmap;
		current_layer = -1;
		selected_user.slice = current_layer;
		selected_user.user_idx = -1;
		selected_user.subinstance = -1;
	}

	selected_node = p_selected_node;

	if (lightmap) {
		lightmap_data = p_lightmap->get_light_data();
		if (lightmap_data.is_valid()) {
			texture_preview->show();
			texture_overlay->show();
			int total = 0;
			for (Ref<TextureLayered> tex : lightmap_data->get_lightmap_textures()) {
				total += tex->get_layers();
			}

			total_layers = total;
			sb_layer->set_max(total_layers - 1);

			if (total_layers > 1) {
				sb_layer->set_editable(true);
			} else {
				sb_layer->set_editable(false);
			}

			int subinstance = -1;
			if (user_idx < 0) {
				_update_current_texture(current_layer, show_shadowmask);
				_update_list();
			} else {
				if (selected_user.subinstance > -1) {
					// it was part of a subinstance
					NodePath current_selected = lightmap_data->get_user_path(selected_user.user_idx);
					NodePath to_be_selected = lightmap_data->get_user_path(user_idx);
					if (current_selected == to_be_selected) {
						user_idx = selected_user.user_idx;
					}
				}
				subinstance = lightmap_data->get_user_sub_instance(user_idx);
				int layer = lightmap_data->get_user_lightmap_slice_index(user_idx);
				_update_current_texture(layer, show_shadowmask);
				_update_list();
			}

			UserEntry user;
			user.user_idx = user_idx;
			user.slice = current_layer;
			user.subinstance = subinstance;
			_select_user(user, NONE);

			open();
			callable_mp(this, &LightmapGITextureEditor::_zoom_to_fit).call_deferred();
			return;
		}
	}

	current_layer = -1;
	selected_user.slice = current_layer;
	selected_user.user_idx = -1;
	selected_user.subinstance = -1;
	users_in_list.clear();
	current_texture = Ref<Texture2D>();
	sb_layer->set_editable(false);
	texture_preview->hide();
	texture_overlay->hide();
	close();
}

LightmapGITextureEditor::LightmapGITextureEditor() {
	set_name(TTRC("Lightmap Texture Viewer"));
	set_icon_name("LightmapGIData");
	set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_VERTICAL | EditorDock::DOCK_LAYOUT_FLOATING);
	set_global(false);
	set_closable(false);
	set_transient(true);

	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &LightmapGITextureEditor::_pan_callback), callable_mp(this, &LightmapGITextureEditor::_zoom_callback));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *hb_tools = memnew(HBoxContainer);
	vb->add_child(hb_tools);

	cb_show_shadowmask = memnew(CheckButton);
	hb_tools->add_child(cb_show_shadowmask);
	cb_show_shadowmask->set_text(TTR("Shadowmask Texture"));
	cb_show_shadowmask->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_show_shadowmask_pressed));

	cb_show_uv = memnew(CheckButton);
	hb_tools->add_child(cb_show_uv);
	cb_show_uv->set_text(TTR("Show UV"));
	cb_show_uv->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_show_uv_pressed));

	// Default the zoom to match the editor scale, but don't dezoom on editor scales below 100% to prevent pixel art from looking bad.
	draw_zoom = MAX(1.0f, EDSCALE);
	max_draw_zoom = 128.0f * MAX(1.0f, EDSCALE);
	min_draw_zoom = 0.01f * MAX(1.0f, EDSCALE);

	BoxContainer *hb_preview = memnew(BoxContainer);
	hb_preview->set_vertical(false);
	hb_preview->set_h_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	hb_preview->set_v_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	vb->add_child(hb_preview);

	texture_preview = memnew(PanelContainer);
	hb_preview->add_child(texture_preview);
	texture_preview->set_h_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	texture_preview->set_v_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	texture_preview->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	texture_preview->set_clip_contents(true);
	texture_preview->connect(SceneStringName(draw), callable_mp(this, &LightmapGITextureEditor::_texture_preview_draw));

	texture_overlay = memnew(Control);
	texture_preview->add_child(texture_overlay);
	texture_overlay->set_focus_mode(Control::FOCUS_CLICK);
	texture_overlay->connect(SceneStringName(draw), callable_mp(this, &LightmapGITextureEditor::_texture_overlay_draw));
	texture_overlay->connect(SceneStringName(gui_input), callable_mp(this, &LightmapGITextureEditor::_texture_overlay_input));
	texture_overlay->connect(SceneStringName(focus_exited), callable_mp(panner.ptr(), &ViewPanner::release_pan_key));
	texture_overlay->hide();

	HBoxContainer *zoom_hb = memnew(HBoxContainer);
	texture_overlay->add_child(zoom_hb);
	zoom_hb->set_begin(Point2(5, 5));

	zoom_out = memnew(Button);
	zoom_out->set_theme_type_variation(SceneStringName(FlatButton));
	zoom_out->set_focus_mode(FOCUS_ACCESSIBILITY);
	zoom_out->set_tooltip_text(TTRC("Zoom Out"));
	zoom_out->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_zoom_out));
	zoom_hb->add_child(zoom_out);

	zoom_reset = memnew(Button);
	zoom_reset->set_theme_type_variation(SceneStringName(FlatButton));
	zoom_reset->set_focus_mode(FOCUS_ACCESSIBILITY);
	zoom_reset->set_tooltip_text(TTR("Zoom Reset"));
	zoom_reset->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_zoom_reset));
	zoom_hb->add_child(zoom_reset);

	zoom_in = memnew(Button);
	zoom_in->set_theme_type_variation(SceneStringName(FlatButton));
	zoom_in->set_focus_mode(FOCUS_ACCESSIBILITY);
	zoom_in->set_tooltip_text(TTR("Zoom In"));
	zoom_in->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_zoom_in));
	zoom_hb->add_child(zoom_in);

	zoom_to_fit = memnew(Button);
	zoom_to_fit->set_theme_type_variation(SceneStringName(FlatButton));
	zoom_to_fit->set_focus_mode(FOCUS_ACCESSIBILITY);
	zoom_to_fit->set_tooltip_text(TTR("Zoom To Fit"));
	zoom_to_fit->connect(SceneStringName(pressed), callable_mp(this, &LightmapGITextureEditor::_zoom_to_fit));
	zoom_hb->add_child(zoom_to_fit);

	vscroll = memnew(VScrollBar);
	vscroll->set_anchors_and_offsets_preset(Control::PRESET_RIGHT_WIDE);
	vscroll->set_step(0.001);
	vscroll->connect(SceneStringName(value_changed), callable_mp(this, &LightmapGITextureEditor::_scroll_changed));
	texture_overlay->add_child(vscroll);

	hscroll = memnew(HScrollBar);
	hscroll->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_WIDE);
	hscroll->set_step(0.001);
	hscroll->connect(SceneStringName(value_changed), callable_mp(this, &LightmapGITextureEditor::_scroll_changed));
	texture_overlay->add_child(hscroll);

	VBoxContainer *vb_list = memnew(VBoxContainer);
	hb_preview->add_child(vb_list);

	HBoxContainer *hb_layer = memnew(HBoxContainer);
	vb_list->add_child(hb_layer);
	hb_layer->add_child(memnew(Label(TTR("Layer:"))));

	sb_layer = memnew(SpinBox);
	hb_layer->add_child(sb_layer);
	sb_layer->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	sb_layer->connect(SceneStringName(value_changed), callable_mp(this, &LightmapGITextureEditor::_set_layer));
	sb_layer->set_accessibility_name(TTRC("Layer"));
	sb_layer->set_editable(false);
	sb_layer->set_min(0);
	sb_layer->set_max(0);
	sb_layer->set_step(1.0);

	lbl_islands_title = memnew(Label(TTR("Islands in layer:")));
	vb_list->add_child(lbl_islands_title);

	user_list = memnew(ItemList);
	vb_list->add_child(user_list);
	user_list->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	user_list->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	// user_list->set_select_mode(ItemList::SELECT_TOGGLE);
	user_list->connect(SceneStringName(item_selected), callable_mp(this, &LightmapGITextureEditor::_on_user_list_item_selected));
	user_list->connect("item_activated", callable_mp(this, &LightmapGITextureEditor::_on_user_list_item_activated));
}

///////////////////////////////////////////////

void LightmapGIEditorPlugin::edit(Object *p_object) {
	LightmapGI *lightmap_gi = Object::cast_to<LightmapGI>(p_object);
	if (lightmap_gi) {
		bake_editor->edit(lightmap_gi);
		texture_editor->edit(lightmap_gi);
		return;
	}

	bake_editor->edit(nullptr);

	if (texture_editor->is_internal_selection()) {
		// The node selection was made internally so no need to check everything again
		texture_editor->clear_internal_selection();
		return;
	}

	// Not a LightmapGI search if it's part of one.
	Node3D *selected_node = Object::cast_to<Node3D>(p_object);

	Node *root = EditorNode::get_singleton()->get_edited_scene();

	if (root && selected_node) {
		TypedArray<Node> lightmap_nodes = root->find_children("*", "LightmapGI");

		bool found = false;
		for (int i = 0; i < lightmap_nodes.size(); ++i) {
			if (found) {
				break;
			}
			LightmapGI *lightmap = Object::cast_to<LightmapGI>(lightmap_nodes[i]);
			if (lightmap == nullptr) {
				continue;
			}
			Ref<LightmapGIData> lightmap_data = lightmap->get_light_data();

			if (lightmap_data.is_null()) {
				continue;
			}

			// TODO get a better filter

			for (int user_idx = 0; user_idx < lightmap_data->get_user_count(); ++user_idx) {
				Node *candidate = lightmap->get_node_or_null(lightmap_data->get_user_path(user_idx));
				if (candidate == nullptr) {
					continue;
				}
				if (candidate == selected_node) {
					texture_editor->edit(lightmap, selected_node, user_idx);
					found = true;
					break;
				} else if (selected_node->is_instance()) {
					TypedArray<Node3D> children = selected_node->find_children("*", "Node3D");
					for (int j = 0; j < children.size(); ++j) {
						if (candidate == children[j]) {
							texture_editor->edit(lightmap, selected_node, user_idx);
							found = true;
							break;
						}
					}
				}
				if (found) {
					break;
				}
			}
		}

		if (found) {
			return;
		}
	}

	texture_editor->edit(nullptr);
}

bool LightmapGIEditorPlugin::handles(Object *p_object) const {
	Node *node = Object::cast_to<Node>(p_object);

	if (node == nullptr) {
		return false;
	}

	if (Object::cast_to<LightmapGI>(node)) {
		return true;
	}

	if (Object::cast_to<MeshInstance3D>(node)) {
		return true;
	}

	if (Object::cast_to<Node3D>(node) && node->has_method("get_bake_meshes")) {
		// Possible GridMap or user created node.
		return true;
	}

	if (node->is_instance()) {
		// Possible scene instance.
		return true;
	}

	return false;
}

LightmapGIEditorPlugin::LightmapGIEditorPlugin() {
	bake_editor = memnew(LightmapGIBakeEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(bake_editor);
	bake_editor->bake->hide();

	texture_editor = memnew(LightmapGITextureEditor);
	add_dock(texture_editor);
}
