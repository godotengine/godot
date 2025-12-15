/**************************************************************************/
/*  editor_preview_plugins.cpp                                            */
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

#include "editor_preview_plugins.h"

#include "core/config/project_settings.h"
#include "core/io/image.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "main/main.h"
#include "modules/gridmap/grid_map.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/line_2d.h"
#include "scene/2d/mesh_instance_2d.h"
#include "scene/2d/multimesh_instance_2d.h"
#include "scene/2d/physics/touch_screen_button.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/tile_map_layer.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/option_button.h"
#include "scene/main/window.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/font.h"
#include "scene/resources/gradient_texture.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/world_2d.h"
#include "servers/audio/audio_stream.h"

void post_process_preview(Ref<Image> p_image) {
	if (p_image->get_format() != Image::FORMAT_RGBA8) {
		p_image->convert(Image::FORMAT_RGBA8);
	}

	const int w = p_image->get_width();
	const int h = p_image->get_height();

	const int r = MIN(w, h) / 32;
	const int r2 = r * r;
	Color transparent = Color(0, 0, 0, 0);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < r; j++) {
			int dx = i - r;
			int dy = j - r;
			if (dx * dx + dy * dy > r2) {
				p_image->set_pixel(i, j, transparent);
				p_image->set_pixel(w - 1 - i, j, transparent);
				p_image->set_pixel(w - 1 - i, h - 1 - j, transparent);
				p_image->set_pixel(i, h - 1 - j, transparent);
			} else {
				break;
			}
		}
	}
}

bool EditorTexturePreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Texture");
}

bool EditorTexturePreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

Ref<Texture2D> EditorTexturePreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Image> img;

	Ref<AtlasTexture> tex_atlas = p_from;
	Ref<Texture3D> tex_3d = p_from;
	Ref<TextureLayered> tex_lyr = p_from;

	if (tex_atlas.is_valid()) {
		Ref<Texture2D> tex = tex_atlas->get_atlas();
		if (tex.is_null()) {
			return Ref<Texture2D>();
		}

		Ref<Image> atlas = tex->get_image();
		if (atlas.is_null()) {
			return Ref<Texture2D>();
		}

		if (atlas->is_compressed()) {
			atlas = atlas->duplicate();
			if (atlas->decompress() != OK) {
				return Ref<Texture2D>();
			}
		}

		if (!tex_atlas->get_region().has_area()) {
			return Ref<Texture2D>();
		}

		img = atlas->get_region(tex_atlas->get_region());

	} else if (tex_3d.is_valid()) {
		if (tex_3d->get_depth() == 0) {
			return Ref<Texture2D>();
		}

		Vector<Ref<Image>> data = tex_3d->get_data();
		if (data.size() != tex_3d->get_depth()) {
			return Ref<Texture2D>();
		}

		// Use the middle slice for the thumbnail.
		const int mid_depth = (tex_3d->get_depth() - 1) / 2;
		if (!data.is_empty() && data[mid_depth].is_valid()) {
			img = data[mid_depth]->duplicate();
		}

	} else if (tex_lyr.is_valid()) {
		if (tex_lyr->get_layers() == 0) {
			return Ref<Texture2D>();
		}

		// Use the middle slice for the thumbnail.
		const int mid_layer = (tex_lyr->get_layers() - 1) / 2;

		Ref<Image> data = tex_lyr->get_layer_data(mid_layer);
		if (data.is_valid()) {
			img = data->duplicate();
		}

	} else {
		Ref<Texture2D> tex = p_from;
		if (tex.is_valid()) {
			img = tex->get_image();
			if (img.is_valid()) {
				img = img->duplicate();
			}
		}
	}

	if (img.is_null() || img->is_empty()) {
		return Ref<Texture2D>();
	}

	p_metadata["dimensions"] = img->get_size();

	img->clear_mipmaps();

	if (img->is_compressed()) {
		if (img->decompress() != OK) {
			return Ref<Texture2D>();
		}
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	Vector2i new_size_i = Vector2i(new_size).maxi(1);
	img->resize(new_size_i.x, new_size_i.y, Image::INTERPOLATE_CUBIC);
	post_process_preview(img);

	return ImageTexture::create_from_image(img);
}

////////////////////////////////////////////////////////////////////////////

bool EditorImagePreviewPlugin::handles(const String &p_type) const {
	return p_type == "Image";
}

Ref<Texture2D> EditorImagePreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Image> img = p_from;

	if (img.is_null() || img->is_empty()) {
		return Ref<Texture2D>();
	}

	img = img->duplicate();
	img->clear_mipmaps();

	if (img->is_compressed()) {
		if (img->decompress() != OK) {
			return Ref<Texture2D>();
		}
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);
	post_process_preview(img);

	return ImageTexture::create_from_image(img);
}

bool EditorImagePreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

////////////////////////////////////////////////////////////////////////////

bool EditorBitmapPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "BitMap");
}

Ref<Texture2D> EditorBitmapPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<BitMap> bm = p_from;

	if (bm->get_size() == Size2()) {
		return Ref<Texture2D>();
	}

	Vector<uint8_t> data;

	data.resize(bm->get_size().width * bm->get_size().height);

	{
		uint8_t *w = data.ptrw();

		for (int i = 0; i < bm->get_size().width; i++) {
			for (int j = 0; j < bm->get_size().height; j++) {
				if (bm->get_bit(i, j)) {
					w[j * (int)bm->get_size().width + i] = 255;
				} else {
					w[j * (int)bm->get_size().width + i] = 0;
				}
			}
		}
	}

	Ref<Image> img = Image::create_from_data(bm->get_size().width, bm->get_size().height, false, Image::FORMAT_L8, data);

	if (img->is_compressed()) {
		if (img->decompress() != OK) {
			return Ref<Texture2D>();
		}
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);
	post_process_preview(img);

	return ImageTexture::create_from_image(img);
}

bool EditorBitmapPreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

///////////////////////////////////////////////////////////////////////////

void EditorPackedScenePreviewPlugin::abort() {
	draw_requester.abort();

	// Raise abort flag
	aborted.set();
}

bool EditorPackedScenePreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "PackedScene");
}

Ref<Texture2D> EditorPackedScenePreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	return generate_from_path(p_from->get_path(), p_size, p_metadata);
}

Ref<Texture2D> EditorPackedScenePreviewPlugin::generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const {
	ERR_FAIL_COND_V_MSG(!Engine::get_singleton()->is_editor_hint(), Ref<Texture2D>(), "This function can only be called from the editor.");
	ERR_FAIL_NULL_V_MSG(EditorNode::get_singleton(), Ref<Texture2D>(), "EditorNode doesn't exist.");

	if (!EditorResourcePreview::get_singleton()->is_threaded()) {
		ERR_PRINT_ONCE("Scene preview can only be generated on a separate thread.");
		return Ref<Texture2D>();
	}

	// Lower the abort flag only if it's not raised
	if (aborted.is_set()) {
		return Ref<Texture2D>();
	}

	// Scene file size too large to handle in background (use dummy thumbnail to avoid retrying without file change)
	// Try find and load the image `EditorNode::_save_scene_with_preview` created for us
	if (_get_scene_file_size(p_path) > uint64_t(EDITOR_GET("docks/filesystem/thumbnail_file_size_threshold")) * 1024 * 1024) { // MB
		String temp_path = EditorPaths::get_singleton()->get_cache_dir();
		String cache_base = ProjectSettings::get_singleton()->globalize_path(p_path).md5_text();
		cache_base = temp_path.path_join("resthumb-" + cache_base);

		String path = cache_base + ".png";

		if (!FileAccess::exists(path)) {
			return _create_dummy_thumbnail();
		}

		Ref<Image> img = Image::load_from_file(path);

		if (img.is_valid()) {
			// Found image, use it.
			post_process_preview(img);
			return ImageTexture::create_from_image(img);
		}
		return _create_dummy_thumbnail();
	}

	if (aborted.is_set()) {
		print_verbose("Thumbnail creation aborted.");
		return Ref<Texture2D>();
	}

	// Load scene
	Ref<PackedScene> pack = ResourceLoader::load(p_path, "PackedScene", ResourceFormatLoader::CACHE_MODE_IGNORE_DEEP);

	if (pack.is_null()) {
		ERR_PRINT(vformat(R"(Failed to generate scene thumbnail for "%s" : Invalid scene file.)", p_path));
		return _create_dummy_thumbnail();
	}

	if (aborted.is_set()) {
		print_verbose("Thumbnail creation aborted.");
		return Ref<Texture2D>();
	}

	// Count node types before instantiating it
	bool root_is_viewport = ClassDB::is_parent_class(pack->get_state()->get_node_type(0), SNAME("Viewport"));
	int count_2d = 0;
	int count_3d = 0;
	int count_light_3d = 0;
	_count_node_types(pack, count_2d, count_3d, count_light_3d);
	node_lookup_tables.clear(); // No longer needed

	if (aborted.is_set()) {
		print_verbose("Thumbnail creation aborted.");
		return Ref<Texture2D>();
	}

	// Prohibit Viewport class as root when generating thumbnails (Causes rendering issues)
	if (root_is_viewport) {
		return _create_dummy_thumbnail();
	}

	if (count_3d > 0) { // Is 3d scene
		// Call scene instantiation at idle time
		scene_construct_done.clear();
		callable_mp((EditorPackedScenePreviewPlugin *)this, &EditorPackedScenePreviewPlugin::_construct_scene_3d).call_deferred(pack, p_size, count_light_3d);

		// Wait for scene construct
		while (!scene_construct_done.is_set()) {
			_wait_frame();
		}

		if (aborted.is_set() || vp_3d_rid.is_null()) {
			if (p_vp_3d != nullptr) {
				callable_mp((Node *)p_vp_3d, &Node::queue_free).call_deferred();
			}
			print_verbose("Thumbnail creation aborted.");
			return Ref<Texture2D>();
		}

		RenderingServer *rs = RS::get_singleton();

		_wait_frame();
		_wait_frame(); // Wait twice, as the first wait might occur in frame post draw.
		rs->viewport_set_update_mode(vp_3d_rid, RS::ViewportUpdateMode::VIEWPORT_UPDATE_ONCE);
		_wait_frame(); // HACK - one more frame for CPUParticles to render correctly, don't know why

		if (aborted.is_set()) {
			if (p_vp_3d != nullptr) {
				callable_mp((Node *)p_vp_3d, &Node::queue_free).call_deferred();
			}
			print_verbose("Thumbnail creation aborted.");
			return Ref<Texture2D>();
		}

		// Retrieving via RenderingServer as the scene is now inside tree, we can't get texture directly on a thread (will be blocked by thread locks)
		Ref<Image> capture_img = rs->texture_2d_get(rs->viewport_get_texture(vp_3d_rid));

		// Viewport texture is somehow invalid, abort the process
		if (capture_img.is_null()) {
			if (p_vp_3d != nullptr) {
				callable_mp((Node *)p_vp_3d, &Node::queue_free).call_deferred();
			}
			WARN_PRINT(vformat(R"(Failed to create thumbnail for scene: "%s".)", p_path));
			return Ref<Texture2D>();
		}

		// Retrieve thumbnail
		Ref<ImageTexture> thumbnail = ImageTexture::create_from_image(capture_img);

		// Clean up
		if (p_vp_3d == nullptr || vp_3d_rid.is_null()) {
			ERR_PRINT("FIXME: Preview scene is freed unexpectedly before cleanup");
		} else {
			callable_mp((Node *)p_vp_3d, &Node::queue_free).call_deferred();
			vp_3d_rid = RID();
			p_vp_3d = nullptr;
			print_verbose(vformat(R"(Cleaned up preview scene for: "%s".)", p_path));
		}

		return thumbnail;
	}

	if (count_2d > 0) { // Is 2d scene
		// If anyone want to rewrite this part to call RenderingServer directly, note that at the time of writing (2025-09-24),
		// there's a limitation where CanvasItem cannot be rendered outside of the tree.
		// See CanvasItem::queue_redraw() and RenderingServer::draw()

		// Call scene instantiation at idle time
		scene_construct_done.clear();
		callable_mp((EditorPackedScenePreviewPlugin *)this, &EditorPackedScenePreviewPlugin::_construct_scene_2d).call_deferred(pack, p_size);

		// Wait for scene construct
		while (!scene_construct_done.is_set()) {
			_wait_frame();
		}

		if (aborted.is_set() || vp_2d_rid.is_null() || vp_gui_rid.is_null()) {
			if (p_vp_2d != nullptr) {
				callable_mp((Node *)p_vp_2d, &Node::queue_free).call_deferred();
			}
			if (p_vp_gui != nullptr) {
				callable_mp((Node *)p_vp_gui, &Node::queue_free).call_deferred();
			}
			print_verbose("Thumbnail creation aborted.");
			return Ref<Texture2D>();
		}

		_wait_frame();
		_wait_frame(); // Wait twice, as the first wait might occur in frame post draw.

		if (aborted.is_set()) {
			if (p_vp_2d != nullptr) {
				callable_mp((Node *)p_vp_2d, &Node::queue_free).call_deferred();
			}
			if (p_vp_gui != nullptr) {
				callable_mp((Node *)p_vp_gui, &Node::queue_free).call_deferred();
			}
			print_verbose("Thumbnail creation aborted.");
			return Ref<Texture2D>();
		}

		// Retrieving via RenderingServer as the scene is now inside tree, we can't get texture directly on a thread (will be blocked by thread locks)
		RenderingServer *rs = RS::get_singleton();
		Ref<Image> capture_2d_img = rs->texture_2d_get(rs->viewport_get_texture(vp_2d_rid));
		Ref<Image> capture_gui_img = rs->texture_2d_get(rs->viewport_get_texture(vp_gui_rid));

		// Viewport texture is somehow invalid, abort the process
		if (capture_2d_img.is_null() || capture_gui_img.is_null()) {
			if (p_vp_2d != nullptr) {
				callable_mp((Node *)p_vp_2d, &Node::queue_free).call_deferred();
			}
			if (p_vp_gui != nullptr) {
				callable_mp((Node *)p_vp_gui, &Node::queue_free).call_deferred();
			}
			WARN_PRINT(vformat(R"(Failed to create thumbnail for scene: "%s".)", p_path));
			return Ref<Texture2D>();
		}

		// Retrieve 2D scene capture
		Ref<ImageTexture> capture_2d = ImageTexture::create_from_image(capture_2d_img);
		capture_2d->get_image()->resize(p_size.x, p_size.y);
		capture_2d->get_image()->convert(Image::Format::FORMAT_RGBA8); // ALPHA channel is required for image blending

		// Retrieve GUI capture
		Ref<ImageTexture> capture_gui = ImageTexture::create_from_image(capture_gui_img);
		capture_gui->get_image()->resize(p_size.x, p_size.y);

		// Mix 2D, GUI thumbnail images into one
		Ref<ImageTexture> thumbnail;
		thumbnail.instantiate();
		Ref<Image> thumbnail_image = Image::create_empty(p_size.x, p_size.y, false, Image::Format::FORMAT_RGBA8); // blend_rect needs ALPHA channel to work
		thumbnail_image->blit_rect(capture_2d->get_image(), capture_2d->get_image()->get_used_rect(), Point2i(0, 0));
		thumbnail_image->blend_rect(capture_gui->get_image(), capture_gui->get_image()->get_used_rect(), Point2i(0, 0));
		thumbnail->set_image(thumbnail_image);

		// Clean up
		if (p_vp_2d == nullptr || vp_2d_rid.is_null() || p_vp_gui == nullptr || vp_gui_rid.is_null()) {
			ERR_PRINT("FIXME: Preview scene is freed unexpectedly before cleanup");
		} else {
			callable_mp((Node *)p_vp_2d, &Node::queue_free).call_deferred();
			callable_mp((Node *)p_vp_gui, &Node::queue_free).call_deferred();
			vp_2d_rid = RID();
			vp_gui_rid = RID();
			p_vp_2d = nullptr;
			p_vp_gui = nullptr;
			print_verbose(vformat(R"(Cleaned up preview scene for: "%s".)", p_path));
		}

		return thumbnail;
	}

	// Is scene without any visuals (No Node2D, Node3D, Control found)
	return _create_dummy_thumbnail();
}

void EditorPackedScenePreviewPlugin::_setup_scene_3d(Node *p_node) const {
	// Do not account any SubViewport at preview scene, as it would not render correctly
	if (Object::cast_to<SubViewport>(p_node) && p_node->get_parent()) {
		p_node->get_parent()->remove_child(p_node);
		callable_mp(p_node, &Node::queue_free).call_deferred();
		return;
	}

	// Don't let window to popup
	Window *window = Object::cast_to<Window>(p_node);
	if (window) {
		window->set_visible(false);
	}

	// Make sure no Node2D, Control node is visible (Might occupy large proportion of the thumbnail)
	Node2D *n2d = Object::cast_to<Node2D>(p_node);
	if (n2d) {
		n2d->set_visible(false);
	}

	Control *ctrl = Object::cast_to<Control>(p_node);
	if (ctrl) {
		ctrl->set_visible(false);
	}

	// Disable skinning for skeleton meshes (animations and skeleton scaling might disturb aabb calculation)
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh && mesh->is_visible_in_tree()) {
		mesh->set_skeleton_path(NodePath());
	}

	CPUParticles3D *cpu_particles = Object::cast_to<CPUParticles3D>(p_node);
	if (cpu_particles && cpu_particles->is_visible_in_tree()) {
		cpu_particles->set_seed(0);
		cpu_particles->set_pre_process_time(cpu_particles->get_lifetime() * 0.5);
		cpu_particles->set_use_local_coordinates(true); // HACK - Now constructs scene outside of tree, using global coords will cause error, might result in visual artifacts, but is the best solution now
		cpu_particles->restart(true); // Keep seed to make simulation persistent
	}

	GPUParticles3D *gpu_particles = Object::cast_to<GPUParticles3D>(p_node);
	if (gpu_particles && gpu_particles->is_visible_in_tree()) {
		// Convert to CPUParticles (As GPUParticles can't be rendered correctly in thumbnails, don't know why)
		CPUParticles3D *gtc_particles = memnew(CPUParticles3D); // GPU to CPU particles
		gtc_particles->convert_from_particles(gpu_particles);
		gpu_particles->add_child(gtc_particles); // So the created CPUParticles instance will be freed later we call queue_free() on the preview scene

		// Setup the converted CPUParticles
		gtc_particles->set_seed(0);
		gtc_particles->set_pre_process_time(gtc_particles->get_lifetime() * 0.5);
		gtc_particles->set_use_local_coordinates(true);
		gtc_particles->restart(true);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_setup_scene_3d(p_node->get_child(i));
	}
}

// This should be called at idle time to ensure thread-safe
void EditorPackedScenePreviewPlugin::_construct_scene_3d(Ref<PackedScene> p_pack, const Size2 &p_size, int p_light_count) const {
	bool _scene_setup_success = _setup_packed_scene(p_pack);
	if (!_scene_setup_success) {
		scene_construct_done.set();
		ERR_PRINT(vformat(R"(Failed to generate scene thumbnail for "%s" : error in setting up preview scene, thus not safe to create thumbnail image.)", p_pack->get_path()));
		return;
	}

	Node *p_scene = p_pack->instantiate();
	if (p_scene == nullptr) {
		scene_construct_done.set();
		ERR_PRINT(vformat(R"(Scene "%s" failed to instantiate.)", p_pack->get_path()));
		return;
	}

	SubViewport *sub_viewport = memnew(SubViewport);
	sub_viewport->set_update_mode(SubViewport::UpdateMode::UPDATE_DISABLED);

	sub_viewport->set_size(p_size.round());
	sub_viewport->set_transparent_background(false);
	sub_viewport->set_disable_3d(false);

	if (p_size.x < 2048 && p_size.y < 2048) { // Universal baseline for textures in Godot 4 is 4K
		sub_viewport->set_scaling_3d_scale(2.0); // Supersampling x2
	}

	sub_viewport->set_msaa_3d(supported_msaa_method);

	Ref<Environment> environment;
	Color default_clear_color = GLOBAL_GET("rendering/environment/defaults/default_clear_color");
	environment.instantiate();
	environment->set_background(Environment::BGMode::BG_CLEAR_COLOR);
	environment->set_bg_color(default_clear_color);
	Ref<World3D> world_3d;
	sub_viewport->set_world_3d(world_3d);
	sub_viewport->set_use_own_world_3d(true);

	// Add scene to viewport
	_setup_scene_3d(p_scene);
	sub_viewport->add_child(p_scene);

	// Setup preview light
	DirectionalLight3D *light_1 = nullptr;
	DirectionalLight3D *light_2 = nullptr;

	if (p_light_count == 0) {
		light_1 = memnew(DirectionalLight3D);
		light_2 = memnew(DirectionalLight3D);
		sub_viewport->add_child(light_1);
		sub_viewport->add_child(light_2);
		light_1->set_color(Color(1.0, 1.0, 1.0, 1.0));
		light_2->set_color(Color(0.7, 0.7, 0.7, 1.0));
		light_1->set_transform(Transform3D(Basis().rotated(Vector3(0, 1, 0), -Math::PI / 6), Vector3(0.0, 0.0, 0.0)));
		light_2->set_transform(Transform3D(Basis().rotated(Vector3(1, 0, 0), -Math::PI / 6), Vector3(0.0, 0.0, 0.0)));
	}

	// Calculate scene AABB
	AABB scene_aabb;
	_calculate_scene_aabb(p_scene, scene_aabb);
	float bound_sphere_radius = (scene_aabb.get_end() - scene_aabb.get_position()).length() / 2.0f;
	if (bound_sphere_radius <= 0.0f) {
		// The scene has zero volume, so just it give a literal
		bound_sphere_radius = 1.0f;
	}

	// Setup preview camera
	const float cam_fov = 30.0;
	const float cam_distance = bound_sphere_radius / Math::tan(Math::deg_to_rad(cam_fov) / 2.0f);
	const float cam_near = cam_distance * 0.01;
	const float cam_far = cam_distance * 2.0;

	Camera3D *camera_3d = memnew(Camera3D);
	sub_viewport->add_child(camera_3d);
	camera_3d->set_perspective(cam_fov, cam_near, cam_far);

	Transform3D thumbnail_cam_trans_3d;
	thumbnail_cam_trans_3d.set_origin(scene_aabb.get_center() + Vector3(1.0f, 0.25f, 1.0f).normalized() * cam_distance);
	thumbnail_cam_trans_3d.set_look_at(thumbnail_cam_trans_3d.origin, scene_aabb.get_center());

	// // Set camera to orthogonal if distance exceeds camera default far (large scene)
	//if (thumbnail_cam_trans_3d.origin.length() > camera_3d->get_far()) {
	//	real_t distance = thumbnail_cam_trans_3d.origin.length();
	//	camera_3d->set_orthogonal(distance / 2.0, distance / 1000.0, distance); // Approximately contains the whole scene.
	//}

	camera_3d->set_transform(thumbnail_cam_trans_3d);

	// Attach the preview viewport to MainTree
	EditorNode::get_singleton()->add_child(sub_viewport);
	camera_3d->set_current(true);
	sub_viewport->set_update_mode(SubViewport::UpdateMode::UPDATE_ONCE);

	// Assign GUI viewport
	vp_3d_rid = sub_viewport->get_viewport_rid();
	p_vp_3d = sub_viewport;
	scene_construct_done.set();
}

void EditorPackedScenePreviewPlugin::_calculate_scene_aabb(Node *p_node, AABB &r_aabb) const {
	GeometryInstance3D *g3d = Object::cast_to<GeometryInstance3D>(p_node);
	if (g3d && g3d->is_visible_in_tree()) { // Use this because VisualInstance3D may have derived classes that are non-graphical (probes, volumes)
		AABB node_aabb = _get_global_transform_3d(g3d).xform(g3d->get_aabb());
		r_aabb.merge_with(node_aabb);
	}

	CPUParticles3D *cpu_particles = Object::cast_to<CPUParticles3D>(p_node);
	if (cpu_particles && cpu_particles->is_visible_in_tree()) { // CPUParticles3D does not calculate particle bounds, so do it here
		// Account the furthest position where particles can go
		Vector3 particle_destination = _get_global_transform_3d(cpu_particles).origin;
		particle_destination += cpu_particles->get_direction() * cpu_particles->get_param_max(CPUParticles3D::PARAM_INITIAL_LINEAR_VELOCITY);
		r_aabb.expand_to(particle_destination * 0.5);
		r_aabb.expand_to(particle_destination * -0.5);
	}

	GPUParticles3D *gpu_particles = Object::cast_to<GPUParticles3D>(p_node);
	if (gpu_particles && gpu_particles->is_visible_in_tree()) {
		r_aabb.merge_with(_get_global_transform_3d(gpu_particles).xform(gpu_particles->get_visibility_aabb()));
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_calculate_scene_aabb(p_node->get_child(i), r_aabb);
	}
}

Transform3D EditorPackedScenePreviewPlugin::_get_global_transform_3d(Node3D *p_n3d) const {
	// Designed to work even if node is outside the tree (is_inside_tree() != true)
	Transform3D global_transform;
	Array parents;

	Node *p_loop_node = p_n3d;
	while (p_loop_node != nullptr) {
		if (Object::cast_to<Node3D>(p_loop_node)) {
			parents.append(p_loop_node);
		}
		p_loop_node = p_loop_node->get_parent();
	}

	parents.reverse();

	for (int i = 0; i < parents.size(); i++) {
		Node3D *p_parent = Object::cast_to<Node3D>(parents[i]);
		if (i == 0) {
			global_transform = p_parent->get_transform();
			continue;
		}
		global_transform *= p_parent->get_transform();
	}

	return global_transform;
}

void EditorPackedScenePreviewPlugin::_setup_scene_2d(Node *p_node) const {
	// Don't let window to popup
	Window *window = Object::cast_to<Window>(p_node);
	if (window) {
		window->set_visible(false);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_setup_scene_2d(p_node->get_child(i));
	}
}

// Constructs 2d scene under EditorNode, this should be called at idle time for thread safety
void EditorPackedScenePreviewPlugin::_construct_scene_2d(Ref<PackedScene> p_pack, const Size2 &p_size) const {
	bool _scene_setup_success = _setup_packed_scene(p_pack);
	if (!_scene_setup_success) {
		scene_construct_done.set();
		ERR_PRINT(vformat(R"(Failed to generate scene thumbnail for "%s" : error in setting up preview scene, thus not safe to create thumbnail image.)", p_pack->get_path()));
		return;
	}

	Node *p_scene = p_pack->instantiate();
	if (p_scene == nullptr) {
		scene_construct_done.set();
		ERR_PRINT(vformat(R"(Scene "%s" failed to instantiate.)", p_pack->get_path()));
		return;
	}

	int texture_filter = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_filter");
	int texture_repeat = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_repeat");

	SubViewport *sub_viewport = memnew(SubViewport);
	sub_viewport->set_update_mode(SubViewport::UpdateMode::UPDATE_DISABLED);
	sub_viewport->set_disable_3d(true);
	sub_viewport->set_transparent_background(false);
	sub_viewport->set_msaa_2d(supported_msaa_method);

	sub_viewport->set_default_canvas_item_texture_filter(Viewport::DefaultCanvasItemTextureFilter(texture_filter));
	sub_viewport->set_default_canvas_item_texture_repeat(Viewport::DefaultCanvasItemTextureRepeat(texture_repeat));
	Ref<World2D> world;
	world.instantiate();
	sub_viewport->set_world_2d(world);
	sub_viewport->add_child(p_scene);

	_setup_scene_2d(p_scene);
	_hide_gui_in_scene(p_scene);

	// Preview camera
	Camera2D *camera = memnew(Camera2D);
	sub_viewport->add_child(camera);
	camera->set_enabled(true);

	// Attach subviewport (following process needs scene to be in tree)
	EditorNode::get_singleton()->add_child(sub_viewport);
	camera->make_current();

	// Calculate scene rect
	Rect2 scene_rect;
	_calculate_scene_rect(p_scene, scene_rect);
	Vector2 scene_true_center = scene_rect.get_center();

	// Place camera 2D
	camera->set_position(Point2(scene_true_center));

	// Set 2D viewport update
	uint16_t scene_rect_long = MAX(scene_rect.get_size().x, scene_rect.get_size().y);
	if (scene_rect_long == 0) {
		scene_rect_long = MAX(p_size.x, p_size.y); // Prevent 0 size rect (which causes error) and defaults to thumbnail size.
	}
	sub_viewport->set_size(p_size);
	camera->set_zoom(Vector2(p_size.x / float(scene_rect_long), p_size.y / float(scene_rect_long)));
	sub_viewport->set_update_mode(SubViewport::UpdateMode::UPDATE_ONCE);

	// Assign 2D viewport (No GUI)
	vp_2d_rid = sub_viewport->get_viewport_rid();
	p_vp_2d = sub_viewport;

	// Prepare for gui render
	Node *p_scene_gui = p_pack->instantiate();
	_setup_scene_2d(p_scene_gui);
	_hide_node_2d_in_scene(p_scene_gui);

	SubViewport *sub_viewport_gui = memnew(SubViewport);
	sub_viewport_gui->set_size(Size2i(GLOBAL_GET("display/window/size/viewport_width"), GLOBAL_GET("display/window/size/viewport_height")));
	sub_viewport_gui->set_update_mode(SubViewport::UpdateMode::UPDATE_DISABLED);
	sub_viewport_gui->set_transparent_background(true);
	sub_viewport_gui->set_msaa_2d(supported_msaa_method);
	sub_viewport_gui->set_default_canvas_item_texture_filter(Viewport::DefaultCanvasItemTextureFilter(texture_filter));
	sub_viewport_gui->set_default_canvas_item_texture_repeat(Viewport::DefaultCanvasItemTextureRepeat(texture_repeat));
	sub_viewport_gui->set_disable_3d(true);
	sub_viewport_gui->add_child(p_scene_gui);

	// Set GUI viewport update
	EditorNode::get_singleton()->add_child(sub_viewport_gui);
	sub_viewport_gui->set_update_mode(SubViewport::UpdateMode::UPDATE_ONCE);

	// Assign GUI viewport
	vp_gui_rid = sub_viewport_gui->get_viewport_rid();
	p_vp_gui = sub_viewport_gui;
	scene_construct_done.set();
}

void EditorPackedScenePreviewPlugin::_calculate_scene_rect(Node *p_node, Rect2 &r_rect) const {
	// NOTE: There's no universal way to get the exact global rect as a Node2D, so we dig into subclasses one by one

	// NOTE:
	// 1. Sprite2D::position by default is at the **center** of the sprite. (with offset == (0,0) AND centered == true)
	// 2. Rect2::position is at the **up-left** of the rect
	// 3. AABB::position is at the **bottom-left-forward** of the bounding box
	//
	// calculation below is done with these in mind.

	Rect2 n2d_rect; // The rect of the current iterating Node2D

	Sprite2D *sprite = Object::cast_to<Sprite2D>(p_node);
	if (sprite && sprite->is_visible_in_tree()) {
		n2d_rect.size = sprite->get_global_scale() * sprite->get_rect().size;
		n2d_rect.position = sprite->get_global_position() + sprite->get_offset() * sprite->get_global_scale();
		if (sprite->is_centered()) {
			n2d_rect.position -= n2d_rect.size / 2.0f;
		}
	}

	AnimatedSprite2D *anim_sprite = Object::cast_to<AnimatedSprite2D>(p_node);
	if (anim_sprite && anim_sprite->is_visible_in_tree()) {
		if (anim_sprite->get_sprite_frames().is_valid()) {
			Ref<Texture2D> current_frame_tex = anim_sprite->get_sprite_frames()->get_frame_texture(anim_sprite->get_animation(), anim_sprite->get_frame());

			if (current_frame_tex.is_valid()) {
				n2d_rect.size = current_frame_tex->get_size() * anim_sprite->get_global_scale();
				n2d_rect.position = anim_sprite->get_global_position() + anim_sprite->get_offset() * anim_sprite->get_global_scale();
				if (anim_sprite->is_centered()) {
					n2d_rect.position -= n2d_rect.size / 2.0f;
				}
			}
		}
	}

	MeshInstance2D *mesh2d = Object::cast_to<MeshInstance2D>(p_node);
	if (mesh2d && mesh2d->is_visible_in_tree()) {
		// NOTE: Conversion is 1m = 1px (before 2d scale)
		Ref<Mesh> mesh = mesh2d->get_mesh();

		if (mesh.is_valid()) {
			// Discard z axis (depth) and only get length of mesh in x,y axis
			n2d_rect.size.x = (mesh->get_aabb().get_end() - mesh->get_aabb().position).x;
			n2d_rect.size.y = (mesh->get_aabb().get_end() - mesh->get_aabb().position).y;
			n2d_rect.size *= mesh2d->get_global_scale();

			// Account for mesh offset in 3d space when calculating rect2
			n2d_rect.position.x = mesh2d->get_global_position().x + mesh->get_aabb().position.x * mesh2d->get_global_scale().x; // AABB::position is bottom-left
			n2d_rect.position.y = mesh2d->get_global_position().y + mesh->get_aabb().position.y * mesh2d->get_global_scale().y;
		}
	}

	MultiMeshInstance2D *mmesh2d = Object::cast_to<MultiMeshInstance2D>(p_node);
	if (mmesh2d && mmesh2d->is_visible_in_tree()) {
		// Basically the same procedure as MeshInstance2D.
		Ref<MultiMesh> mmesh = mmesh2d->get_multimesh();

		if (mmesh.is_valid()) {
			n2d_rect.size.x = (mmesh->get_aabb().get_end() - mmesh->get_aabb().position).x;
			n2d_rect.size.y = (mmesh->get_aabb().get_end() - mmesh->get_aabb().position).y;
			n2d_rect.size *= mmesh2d->get_global_scale();

			n2d_rect.position.x = mmesh2d->get_global_position().x + mmesh->get_aabb().position.x * mmesh2d->get_global_scale().x;
			n2d_rect.position.y = mmesh2d->get_global_position().y + mmesh->get_aabb().position.y * mmesh2d->get_global_scale().y;
		}
	}

	TileMapLayer *tile_map = Object::cast_to<TileMapLayer>(p_node);
	if (tile_map && tile_map->is_visible_in_tree()) {
		// NOTE: TileMapLayer::get_used_rect() only count cells, not their actual pixel size

		if (tile_map->get_tile_set().is_valid()) {
			Size2 tile_size = tile_map->get_tile_set()->get_tile_size(); // Tile map cell pixel size (x,y).
			Rect2 tile_rect = tile_map->get_used_rect(); // Unit is in cells, not pixels!

			n2d_rect.position = tile_map->get_global_position() + tile_rect.position * tile_size * tile_map->get_global_scale(); // Accounts tilemap offset
			n2d_rect.size = tile_rect.size * tile_size * tile_map->get_global_scale();
		}
	}

	Polygon2D *poly2d = Object::cast_to<Polygon2D>(p_node);
	if (poly2d && poly2d->is_visible_in_tree()) {
		PackedVector2Array polygon = poly2d->get_polygon();

		if (polygon.size() > 2) { // Abort if there's no surface (min = 3 verts)
			// Calculate bounds
			float max_x = polygon[0].x;
			float min_x = polygon[0].x;
			float max_y = polygon[0].y;
			float min_y = polygon[0].y;
			for (int i = 0; i < polygon.size(); i++) {
				if (polygon[i].x > max_x) {
					max_x = polygon[i].x;
				}
				if (polygon[i].x < min_x) {
					min_x = polygon[i].x;
				}

				if (polygon[i].y > max_y) {
					max_y = polygon[i].y;
				}
				if (polygon[i].y < min_y) {
					min_y = polygon[i].y;
				}
			}

			Rect2 poly_rect = Rect2(min_x, min_y, max_x - min_x, max_y - min_y);

			n2d_rect.position = poly2d->get_global_position() + poly2d->get_offset() * poly2d->get_global_scale();
			n2d_rect.position += poly_rect.position * poly2d->get_global_scale();
			n2d_rect.size = poly_rect.size * poly2d->get_global_scale();
		}
	}

	Line2D *line2d = Object::cast_to<Line2D>(p_node);
	if (line2d && line2d->is_visible_in_tree()) {
		// The same procedure as Polygon2D
		PackedVector2Array points = line2d->get_points();

		if (line2d->get_point_count() > 1) { // Abort if there's no line drawn
			// Calculate bounds
			float max_x = points[0].x;
			float min_x = points[0].x;
			float max_y = points[0].y;
			float min_y = points[0].y;
			for (int i = 0; i < points.size(); i++) {
				if (points[i].x > max_x) {
					max_x = points[i].x;
				}
				if (points[i].x < min_x) {
					min_x = points[i].x;
				}

				if (points[i].y > max_y) {
					max_y = points[i].y;
				}
				if (points[i].y < min_y) {
					min_y = points[i].y;
				}
			}

			Rect2 line2d_rect = Rect2(min_x, min_y, max_x - min_x, max_y - min_y);

			n2d_rect.position = line2d->get_global_position();
			n2d_rect.position += line2d_rect.position * line2d->get_global_scale();
			n2d_rect.size = line2d_rect.size * line2d->get_global_scale();
			n2d_rect.size += Size2(line2d->get_width(), line2d->get_width()) / 2.0f; // account for line width
		}
	}

	TouchScreenButton *btn = Object::cast_to<TouchScreenButton>(p_node);
	if (btn && btn->is_visible_in_tree()) {
		Ref<Texture2D> btn_tex = btn->get_texture_normal();

		if (btn_tex.is_valid()) { // Abort if there's no normal texture for this button (won't display anything)
			n2d_rect.position = btn->get_global_position(); // It's not possible to offset image in this node
			n2d_rect.size = btn_tex->get_size() * btn->get_global_scale();
		}
	}

	// Merge the calculated node 2d rect
	if (Math::is_zero_approx(r_rect.get_size().length_squared())) { // Avoid accounting scene origin (0,0) into scene rect.
		r_rect = n2d_rect.abs();
	} else {
		r_rect = r_rect.merge(n2d_rect.abs());
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_calculate_scene_rect(p_node->get_child(i), r_rect);
	}
}

void EditorPackedScenePreviewPlugin::_hide_node_2d_in_scene(Node *p_node) const {
	// NOTE: Irreversible (cannot unhide nodes after this)
	// We cannot simple hide() since it will affect all its children (may contain Control nodes)

	if (p_node->is_class("Node2D")) {
		Node2D *n2d = Object::cast_to<Node2D>(p_node);
		n2d->set_self_modulate(Color(0.0f, 0.0f, 0.0f, 0.0f));
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_hide_node_2d_in_scene(p_node->get_child(i));
	}
}

void EditorPackedScenePreviewPlugin::_hide_gui_in_scene(Node *p_node) const {
	// NOTE: Irreversible (cannot unhide nodes after this)
	// We cannot simply hide() since it will affect all its children (may contain Node2D nodes)

	if (p_node->is_class("Control")) {
		Control *ctrl = Object::cast_to<Control>(p_node);
		ctrl->set_self_modulate(Color(0.0f, 0.0f, 0.0f, 0.0f));
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_hide_gui_in_scene(p_node->get_child(i));
	}
}

// HACK - Wait for main process frame
// Needed for reasons:
// 1. Scene preview requires waiting for RenderingServer to redraw. (main thread)
// 2. The first wait might occur after frame post draw, so even when the frame count increments, preview scene viewport might not be updated yet.
// 3. Because of reason 2. , we need to wait 2 frames for preview to generate correctly
void EditorPackedScenePreviewPlugin::_wait_frame() const {
	const uint64_t max_wait_msecs = 3000;
	const uint64_t prev_msec = OS::get_singleton()->get_ticks_msec();
	const uint64_t prev_frame = Engine::get_singleton()->get_frames_drawn();

	while (Engine::get_singleton()->get_frames_drawn() - prev_frame < 1) {
		// So we don't get stuck in here forever if something goes wrong
		if (OS::get_singleton()->get_ticks_msec() - prev_msec > max_wait_msecs) {
			break;
		}
		if (aborted.is_set()) {
			break;
		}
	}
}

// Given a SceneState and a node index, returns if node is visible in scene
bool EditorPackedScenePreviewPlugin::_is_node_visible(Ref<SceneState> p_state, int p_node_idx) const {
	ERR_FAIL_COND_V_MSG(p_node_idx > p_state->get_node_count() - 1, false, "`p_node_idx` is out of bounds.");

	// Check if self is visible
	bool self_visible = _scene_get_property_value(p_state, p_node_idx, "visible", true);
	if (!self_visible) {
		return false;
	}

	// Check if parents are visible
	NodePath parent_node_path = p_state->get_node_path(p_node_idx);
	HashMap<String, int> map_name_to_idx = _create_node_lookup_table(p_state);

	while (parent_node_path.get_name_count() > 1) {
		parent_node_path = parent_node_path.slice(0, -1); // To next parent

		if (!map_name_to_idx.has(String(parent_node_path))) {
			continue; // Can happen if Node is under a Editable Children, we keep going until reaches its root
		}

		int parent_idx = map_name_to_idx[String(parent_node_path)];
		StringName parent_type = p_state->get_node_type(parent_idx);
		if (ClassDB::is_parent_class(parent_type, "Viewport")) {
			return false; // If hit a viewport derived parent, consider this node is not visible, we don't want it to contribute to thumbnails.
		}

		bool parent_visible = _scene_get_property_value(p_state, parent_idx, "visible", true);
		if (!parent_visible) {
			return false; // Found a non-visible parent
		}
	}

	return true; // self and all parents has not set the `visible` property (all defaults to true)
}

// Given a SceneState, return a table that maps {NodePath : node_index}, as `node_index` represents the node index in SceneState
HashMap<String, int> EditorPackedScenePreviewPlugin::_create_node_lookup_table(Ref<SceneState> p_state) const {
	if (node_lookup_tables.has(p_state->get_path())) {
		return node_lookup_tables[p_state->get_path()];
	}

	HashMap<String, int> map_name_to_idx; // For look up node index by NodePath

	for (int i = 0; i < p_state->get_node_count(); i++) {
		map_name_to_idx[String(p_state->get_node_path(i))] = i;
	}

	node_lookup_tables[p_state->get_path()] = map_name_to_idx;
	return map_name_to_idx;
}

// Designed to work without instantiating the scene (So we can do this on a thread)
void EditorPackedScenePreviewPlugin::_count_node_types(Ref<PackedScene> p_pack, int &r_c2d, int &r_c3d, int &r_clight) const {
	Ref<SceneState> scene_state = p_pack->get_state();
	for (int i = 0; i < scene_state->get_node_count(); i++) {
		// Recursive call when node is a scene instance
		if (scene_state->get_node_instance(i).is_valid()) {
			_count_node_types(scene_state->get_node_instance(i), r_c2d, r_c3d, r_clight);
			continue;
		}

		StringName node_type = scene_state->get_node_type(i);

		if (!_is_node_visible(scene_state, i)) {
			continue; // Do not count nodes that are not visible, or under a viewport.
		}
		if (ClassDB::is_parent_class(node_type, SNAME("Control"))) {
			r_c2d++;
		}
		if (ClassDB::is_parent_class(node_type, SNAME("Node2D"))) {
			r_c2d++;
		}
		if (ClassDB::is_parent_class(node_type, SNAME("Node3D"))) {
			r_c3d++;
		}
		if (ClassDB::is_parent_class(node_type, SNAME("Light3D"))) {
			r_clight++;
		}
	}
}

// Try get node property in scene state as common Variant, if failed, returns `p_default_value`
Variant EditorPackedScenePreviewPlugin::_scene_get_property_value(Ref<SceneState> p_state, int &r_node_idx, const StringName &p_property_name, const Variant &p_default_value) const {
	bool property_found;
	bool node_deferred;
	ERR_FAIL_COND_V_MSG(p_property_name == "", p_default_value, "`p_property_name` is not defined, returning default value.");

	if (r_node_idx > p_state->get_node_count() - 1) {
		ERR_PRINT(vformat("`r_node_idx = %d` is out of bounds (p_state->get_node_count() == %d), returning default value.", r_node_idx, p_state->get_node_count()));
		return p_default_value;
	}

	Variant value = p_state->get_property_value(r_node_idx, p_property_name, property_found, node_deferred);
	if (property_found && value.get_type() != Variant::Type::NIL) {
		return value;
	}
	return p_default_value;
}

// Used for scene file that is valid but not suitable to generate thumbnails (no visuals),
// providing a dummy thumbnail ensures the file will not be read again until it's changed.
Ref<ImageTexture> EditorPackedScenePreviewPlugin::_create_dummy_thumbnail() const {
	Ref<Image> dummy_img = Image::create_empty(2, 2, false, Image::Format::FORMAT_RGBA8);
	const Color default_clear_color = GLOBAL_GET_CACHED(Color, "rendering/environment/defaults/default_clear_color");
	dummy_img->fill(default_clear_color);
	return ImageTexture::create_from_image(dummy_img);
}

// Get the accumulated file size in bytes of all the resources a scene has used
uint64_t EditorPackedScenePreviewPlugin::_get_scene_file_size(const String &p_path) const {
	// Could be gltf files at this point
	uint64_t scene_size = FileAccess::get_size(p_path);

	if (!p_path.ends_with(".tscn") && p_path.ends_with(".scn")) {
		return scene_size;
	}

	// Should only be scene files after this point
	List<String> dependencies;
	ResourceLoader::get_dependencies(p_path, &dependencies);

	for (int i = 0; i < dependencies.size(); i++) {
		PackedStringArray uid_res = dependencies.get(i).split("::::"); // uid::::res
		if (uid_res.size() < 2) {
			continue;
		}
		scene_size += FileAccess::get_size(uid_res[1]);
	}
	return scene_size;
}

// Setup the PackedScene before instantiating it for scene previews
bool EditorPackedScenePreviewPlugin::_setup_packed_scene(Ref<PackedScene> p_pack) const {
	// Refer to SceneState in packed_scene.cpp to see how PackedScene is managed underhood.

	// Sanitize
	Dictionary bundle = p_pack->get_state()->get_bundled_scene();
	ERR_FAIL_COND_V(!bundle.has("names"), false);
	ERR_FAIL_COND_V(!bundle.has("variants"), false);
	ERR_FAIL_COND_V(!bundle.has("node_count"), false);
	ERR_FAIL_COND_V(!bundle.has("nodes"), false);
	ERR_FAIL_COND_V(!bundle.has("conn_count"), false);
	ERR_FAIL_COND_V(!bundle.has("conns"), false);

	const uint8_t supported_version = 3;
	uint8_t current_version = bundle.get("version", 1);

	if (current_version > supported_version) {
		WARN_PRINT_ONCE(vformat("Scene thumbnail creation was built upon PackedScene with version %d, but the version has changed to %d now.", supported_version, current_version));
		// And assume it's safe to continue, there should have no reason to change the main structure of PackedScene
	}

	// Find and remove variants in scene
	const Ref<Script> dummy;
	Array edited_variants = bundle["variants"];

	if (edited_variants.is_empty()) {
		return true; // Scene has no resources at all
	}

	for (int i = 0; i < edited_variants.size(); i++) {
		// Clear script
		if (edited_variants[i].get_type() == Variant::OBJECT) {
			if (Object::cast_to<Script>(edited_variants[i])) {
				edited_variants[i] = dummy;
			}

			if (Object::cast_to<PackedScene>(edited_variants[i])) {
				// Recursively apply to all child scenes
				_setup_packed_scene(edited_variants[i]);
			}
		}

		// Clear all properties that can store a NodePath (will cause error)
		if (edited_variants[i].get_type() == Variant::ARRAY) {
			edited_variants[i] = 0;
		}

		if (edited_variants[i].get_type() == Variant::DICTIONARY) {
			edited_variants[i] = 0;
		}

		if (edited_variants[i].get_type() == Variant::NODE_PATH) {
			edited_variants[i] = 0;
		}
	}

	// Remove signal bindings (As scripts are all removed)
	bundle["conns"] = Array();
	bundle["conn_count"] = Array();

	// Create a new scene state
	bundle["variants"] = edited_variants;
	Ref<SceneState> new_state;
	new_state.instantiate();
	new_state->set_bundled_scene(bundle);
	p_pack->replace_state(new_state);
	return true;
}

EditorPackedScenePreviewPlugin::EditorPackedScenePreviewPlugin() {
	if (RS::get_singleton()->get_current_rendering_method() == "forward_plus" || RS::get_singleton()->get_current_rendering_method() == "mobile") {
		supported_msaa_method = Viewport::MSAA::MSAA_8X;
	} else {
		supported_msaa_method = Viewport::MSAA::MSAA_DISABLED;
	}
}

//////////////////////////////////////////////////////////////////

void EditorMaterialPreviewPlugin::abort() {
	draw_requester.abort();
}

bool EditorMaterialPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Material"); // Any material.
}

bool EditorMaterialPreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

Ref<Texture2D> EditorMaterialPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Material> material = p_from;
	ERR_FAIL_COND_V(material.is_null(), Ref<Texture2D>());

	if (material->get_shader_mode() == Shader::MODE_SPATIAL) {
		RS::get_singleton()->mesh_surface_set_material(sphere, 0, material->get_rid());

		draw_requester.request_and_wait(viewport);
		if (EditorResourcePreview::get_singleton()->is_threaded()) {
			draw_requester.request_and_wait(viewport); // HACK - Prevents incorrect thumbnail assignment when using Forward or Mobile renderer, comment out this line to see the bug.
		}

		Ref<Image> img = RS::get_singleton()->texture_2d_get(viewport_texture);
		RS::get_singleton()->mesh_surface_set_material(sphere, 0, RID());

		ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

		img->convert(Image::FORMAT_RGBA8);
		int thumbnail_size = MAX(p_size.x, p_size.y);
		img->resize(thumbnail_size, thumbnail_size, Image::INTERPOLATE_CUBIC);
		post_process_preview(img);
		return ImageTexture::create_from_image(img);
	}

	return Ref<Texture2D>();
}

EditorMaterialPreviewPlugin::EditorMaterialPreviewPlugin() {
	scenario = RS::get_singleton()->scenario_create();

	viewport = RS::get_singleton()->viewport_create();
	RS::get_singleton()->viewport_set_update_mode(viewport, RS::VIEWPORT_UPDATE_DISABLED);
	RS::get_singleton()->viewport_set_scenario(viewport, scenario);
	RS::get_singleton()->viewport_set_size(viewport, 128, 128);
	RS::get_singleton()->viewport_set_transparent_background(viewport, true);
	RS::get_singleton()->viewport_set_active(viewport, true);
	viewport_texture = RS::get_singleton()->viewport_get_texture(viewport);

	camera = RS::get_singleton()->camera_create();
	RS::get_singleton()->viewport_attach_camera(viewport, camera);
	RS::get_singleton()->camera_set_transform(camera, Transform3D(Basis(), Vector3(0, 0, 3)));
	RS::get_singleton()->camera_set_perspective(camera, 45, 0.1, 10);

	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		camera_attributes = RS::get_singleton()->camera_attributes_create();
		RS::get_singleton()->camera_attributes_set_exposure(camera_attributes, 1.0, 0.000032552); // Matches default CameraAttributesPhysical to work well with default DirectionalLight3Ds.
		RS::get_singleton()->camera_set_camera_attributes(camera, camera_attributes);
	}

	light = RS::get_singleton()->directional_light_create();
	light_instance = RS::get_singleton()->instance_create2(light, scenario);
	RS::get_singleton()->instance_set_transform(light_instance, Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));

	light2 = RS::get_singleton()->directional_light_create();
	RS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	//RS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));

	light_instance2 = RS::get_singleton()->instance_create2(light2, scenario);

	RS::get_singleton()->instance_set_transform(light_instance2, Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));

	sphere = RS::get_singleton()->mesh_create();
	sphere_instance = RS::get_singleton()->instance_create2(sphere, scenario);

	int lats = 32;
	int lons = 32;
	const double lat_step = Math::PI / lats;
	const double lon_step = Math::TAU / lons;
	real_t radius = 1.0;

	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<real_t> tangents;
	Basis tt = Basis(Vector3(0, 1, 0), Math::PI * 0.5);

	for (int i = 1; i <= lats; i++) {
		double lat0 = lat_step * (i - 1) - Math::TAU / 4;
		double z0 = Math::sin(lat0);
		double zr0 = Math::cos(lat0);

		double lat1 = lat_step * i - Math::TAU / 4;
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for (int j = lons; j >= 1; j--) {
			double lng0 = lon_step * (j - 1);
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = lon_step * j;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);

			Vector3 v[4] = {
				Vector3(x1 * zr0, z0, y1 * zr0),
				Vector3(x1 * zr1, z1, y1 * zr1),
				Vector3(x0 * zr1, z1, y0 * zr1),
				Vector3(x0 * zr0, z0, y0 * zr0)
			};

#define ADD_POINT(m_idx)                                                                               \
	normals.push_back(v[m_idx]);                                                                       \
	vertices.push_back(v[m_idx] * radius);                                                             \
	{                                                                                                  \
		Vector2 uv;                                                                                    \
		if (j >= lons / 2) {                                                                           \
			uv = Vector2(Math::atan2(-v[m_idx].x, -v[m_idx].z), Math::atan2(v[m_idx].y, -v[m_idx].z)); \
		} else {                                                                                       \
			uv = Vector2(Math::atan2(v[m_idx].x, v[m_idx].z), Math::atan2(-v[m_idx].y, v[m_idx].z));   \
		}                                                                                              \
		uv /= Math::PI;                                                                                \
		uv *= 4.0;                                                                                     \
		uv = uv * 0.5 + Vector2(0.5, 0.5);                                                             \
		uvs.push_back(uv);                                                                             \
	}                                                                                                  \
	{                                                                                                  \
		Vector3 t = tt.xform(v[m_idx]);                                                                \
		tangents.push_back(t.x);                                                                       \
		tangents.push_back(t.y);                                                                       \
		tangents.push_back(t.z);                                                                       \
		tangents.push_back(1.0);                                                                       \
	}

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}

	Array arr;
	arr.resize(RS::ARRAY_MAX);
	arr[RS::ARRAY_VERTEX] = vertices;
	arr[RS::ARRAY_NORMAL] = normals;
	arr[RS::ARRAY_TANGENT] = tangents;
	arr[RS::ARRAY_TEX_UV] = uvs;
	RS::get_singleton()->mesh_add_surface_from_arrays(sphere, RS::PRIMITIVE_TRIANGLES, arr);
}

EditorMaterialPreviewPlugin::~EditorMaterialPreviewPlugin() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RS::get_singleton()->free_rid(sphere);
	RS::get_singleton()->free_rid(sphere_instance);
	RS::get_singleton()->free_rid(viewport);
	RS::get_singleton()->free_rid(light);
	RS::get_singleton()->free_rid(light_instance);
	RS::get_singleton()->free_rid(light2);
	RS::get_singleton()->free_rid(light_instance2);
	RS::get_singleton()->free_rid(camera);
	RS::get_singleton()->free_rid(camera_attributes);
	RS::get_singleton()->free_rid(scenario);
}

///////////////////////////////////////////////////////////////////////////

bool EditorScriptPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Script");
}

Ref<Texture2D> EditorScriptPreviewPlugin::generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const {
	Error err;
	String code = FileAccess::get_file_as_string(p_path, &err);
	if (err != OK) {
		return Ref<Texture2D>();
	}

	ScriptLanguage *lang = ScriptServer::get_language_for_extension(p_path.get_extension());
	return _generate_from_source_code(lang, code, p_size, p_metadata);
}

Ref<Texture2D> EditorScriptPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Script> scr = p_from;
	if (scr.is_null()) {
		return Ref<Texture2D>();
	}

	String code = scr->get_source_code().strip_edges();
	return _generate_from_source_code(scr->get_language(), code, p_size, p_metadata);
}

Ref<Texture2D> EditorScriptPreviewPlugin::_generate_from_source_code(const ScriptLanguage *p_language, const String &p_source_code, const Size2 &p_size, Dictionary &p_metadata) const {
	if (p_source_code.is_empty()) {
		return Ref<Texture2D>();
	}

	HashSet<String> control_flow_keywords;
	HashSet<String> keywords;

	if (p_language) {
		for (const String &keyword : p_language->get_reserved_words()) {
			if (p_language->is_control_flow_keyword(keyword)) {
				control_flow_keywords.insert(keyword);
			} else {
				keywords.insert(keyword);
			}
		}
	}

	int line = 0;
	int col = 0;
	int thumbnail_size = MAX(p_size.x, p_size.y);
	Ref<Image> img = Image::create_empty(thumbnail_size, thumbnail_size, false, Image::FORMAT_RGBA8);

	Color bg_color = EDITOR_GET("text_editor/theme/highlighting/background_color");
	Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
	Color text_color = EDITOR_GET("text_editor/theme/highlighting/text_color");
	Color symbol_color = EDITOR_GET("text_editor/theme/highlighting/symbol_color");
	Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	Color doc_comment_color = EDITOR_GET("text_editor/theme/highlighting/doc_comment_color");

	if (bg_color.a == 0) {
		bg_color = Color(0, 0, 0, 0);
	}
	bg_color.a = MAX(bg_color.a, 0.2); // Ensure we have some background, regardless of the text editor setting.

	img->fill(bg_color);

	const int x0 = thumbnail_size / 8;
	const int y0 = thumbnail_size / 8;
	const int available_height = thumbnail_size - 2 * y0;
	col = x0;

	bool prev_is_text = false;
	bool in_control_flow_keyword = false;
	bool in_keyword = false;
	bool in_comment = false;
	bool in_doc_comment = false;
	for (int i = 0; i < p_source_code.length(); i++) {
		char32_t c = p_source_code[i];
		if (c > 32) {
			if (col < thumbnail_size) {
				Color color = text_color;

				if (c == '#') {
					if (i < p_source_code.length() - 1 && p_source_code[i + 1] == '#') {
						in_doc_comment = true;
					} else {
						in_comment = true;
					}
				}

				if (in_comment) {
					color = comment_color;
				} else if (in_doc_comment) {
					color = doc_comment_color;
				} else {
					if (is_symbol(c)) {
						// Make symbol a little visible.
						color = symbol_color;
						in_control_flow_keyword = false;
						in_keyword = false;
					} else if (!prev_is_text && is_ascii_identifier_char(c)) {
						int pos = i;

						while (is_ascii_identifier_char(p_source_code[pos])) {
							pos++;
						}
						String word = p_source_code.substr(i, pos - i);
						if (control_flow_keywords.has(word)) {
							in_control_flow_keyword = true;
						} else if (keywords.has(word)) {
							in_keyword = true;
						}

					} else if (!is_ascii_identifier_char(c)) {
						in_keyword = false;
					}

					if (in_control_flow_keyword) {
						color = control_flow_keyword_color;
					} else if (in_keyword) {
						color = keyword_color;
					}
				}
				Color ul = color;
				ul.a *= 0.5;
				img->set_pixel(col, y0 + line * 2, bg_color.blend(ul));
				img->set_pixel(col, y0 + line * 2 + 1, color);

				prev_is_text = is_ascii_identifier_char(c);
			}
			col++;
		} else {
			prev_is_text = false;
			in_control_flow_keyword = false;
			in_keyword = false;

			if (c == '\n') {
				in_comment = false;
				in_doc_comment = false;

				col = x0;
				line++;
				if (line >= available_height / 2) {
					break;
				}
			} else if (c == '\t') {
				col += 3;
			} else {
				col++;
			}
		}
	}
	post_process_preview(img);
	return ImageTexture::create_from_image(img);
}

///////////////////////////////////////////////////////////////////

bool EditorAudioStreamPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "AudioStream");
}

Ref<Texture2D> EditorAudioStreamPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<AudioStream> stream = p_from;
	ERR_FAIL_COND_V(stream.is_null(), Ref<Texture2D>());

	Vector<uint8_t> img;

	int w = p_size.x;
	int h = p_size.y;
	img.resize(w * h * 3);

	uint8_t *imgdata = img.ptrw();
	uint8_t *imgw = imgdata;

	Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
	ERR_FAIL_COND_V(playback.is_null(), Ref<Texture2D>());

	real_t len_s = stream->get_length();
	if (len_s == 0) {
		len_s = 60; //one minute audio if no length specified
	}
	int frame_length = AudioServer::get_singleton()->get_mix_rate() * len_s;

	Vector<AudioFrame> frames;
	frames.resize(frame_length);

	playback->start();
	playback->mix(frames.ptrw(), 1, frames.size());
	playback->stop();

	for (int i = 0; i < w; i++) {
		real_t max = -1000;
		real_t min = 1000;
		int from = uint64_t(i) * frame_length / w;
		int to = (uint64_t(i) + 1) * frame_length / w;
		to = MIN(to, frame_length);
		from = MIN(from, frame_length - 1);
		if (to == from) {
			to = from + 1;
		}

		for (int j = from; j < to; j++) {
			max = MAX(max, frames[j].left);
			max = MAX(max, frames[j].right);

			min = MIN(min, frames[j].left);
			min = MIN(min, frames[j].right);
		}

		int pfrom = CLAMP((min * 0.5 + 0.5) * h / 2, 0, h / 2) + h / 4;
		int pto = CLAMP((max * 0.5 + 0.5) * h / 2, 0, h / 2) + h / 4;

		for (int j = 0; j < h; j++) {
			uint8_t *p = &imgw[(j * w + i) * 3];
			if (j < pfrom || j > pto) {
				p[0] = 100;
				p[1] = 100;
				p[2] = 100;
			} else {
				p[0] = 180;
				p[1] = 180;
				p[2] = 180;
			}
		}
	}

	p_metadata["length"] = stream->get_length();

	//post_process_preview(img);

	Ref<Image> image = Image::create_from_data(w, h, false, Image::FORMAT_RGB8, img);
	return ImageTexture::create_from_image(image);
}

///////////////////////////////////////////////////////////////////////////

void EditorMeshPreviewPlugin::abort() {
	draw_requester.abort();
}

bool EditorMeshPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Mesh"); // Any mesh.
}

Ref<Texture2D> EditorMeshPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Mesh> mesh = p_from;
	ERR_FAIL_COND_V(mesh.is_null(), Ref<Texture2D>());

	RS::get_singleton()->instance_set_base(mesh_instance, mesh->get_rid());

	AABB aabb = mesh->get_aabb();
	Vector3 ofs = aabb.get_center();
	aabb.position -= ofs;
	Transform3D xform;
	xform.basis = Basis().rotated(Vector3(0, 1, 0), -Math::PI * 0.125);
	xform.basis = Basis().rotated(Vector3(1, 0, 0), Math::PI * 0.125) * xform.basis;
	AABB rot_aabb = xform.xform(aabb);
	real_t m = MAX(rot_aabb.size.x, rot_aabb.size.y) * 0.5;
	if (m == 0) {
		return Ref<Texture2D>();
	}
	m = 1.0 / m;
	m *= 0.5;
	xform.basis.scale(Vector3(m, m, m));
	xform.origin = -xform.basis.xform(ofs); //-ofs*m;
	xform.origin.z -= rot_aabb.size.z * 2;
	RS::get_singleton()->instance_set_transform(mesh_instance, xform);

	draw_requester.request_and_wait(viewport);
	if (EditorResourcePreview::get_singleton()->is_threaded()) {
		draw_requester.request_and_wait(viewport); // Wait twice when run on thread, or else won't render correctly. comment this line to see the bug
	}

	Ref<Image> img = RS::get_singleton()->texture_2d_get(viewport_texture);
	ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

	RS::get_singleton()->instance_set_base(mesh_instance, RID());

	img->convert(Image::FORMAT_RGBA8);

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);
	post_process_preview(img);

	return ImageTexture::create_from_image(img);
}

EditorMeshPreviewPlugin::EditorMeshPreviewPlugin() {
	scenario = RS::get_singleton()->scenario_create();

	viewport = RS::get_singleton()->viewport_create();
	RS::get_singleton()->viewport_set_update_mode(viewport, RS::VIEWPORT_UPDATE_DISABLED);
	RS::get_singleton()->viewport_set_scenario(viewport, scenario);
	RS::get_singleton()->viewport_set_size(viewport, 128, 128);
	RS::get_singleton()->viewport_set_transparent_background(viewport, true);
	RS::get_singleton()->viewport_set_active(viewport, true);
	viewport_texture = RS::get_singleton()->viewport_get_texture(viewport);

	camera = RS::get_singleton()->camera_create();
	RS::get_singleton()->viewport_attach_camera(viewport, camera);
	RS::get_singleton()->camera_set_transform(camera, Transform3D(Basis(), Vector3(0, 0, 3)));
	//RS::get_singleton()->camera_set_perspective(camera,45,0.1,10);
	RS::get_singleton()->camera_set_orthogonal(camera, 1.0, 0.01, 1000.0);

	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		camera_attributes = RS::get_singleton()->camera_attributes_create();
		RS::get_singleton()->camera_attributes_set_exposure(camera_attributes, 1.0, 0.000032552); // Matches default CameraAttributesPhysical to work well with default DirectionalLight3Ds.
		RS::get_singleton()->camera_set_camera_attributes(camera, camera_attributes);
	}

	light = RS::get_singleton()->directional_light_create();
	light_instance = RS::get_singleton()->instance_create2(light, scenario);
	RS::get_singleton()->instance_set_transform(light_instance, Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));

	light2 = RS::get_singleton()->directional_light_create();
	RS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	//RS::get_singleton()->light_set_color(light2, RS::LIGHT_COLOR_SPECULAR, Color(0.0, 0.0, 0.0));
	light_instance2 = RS::get_singleton()->instance_create2(light2, scenario);

	RS::get_singleton()->instance_set_transform(light_instance2, Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));

	//sphere = RS::get_singleton()->mesh_create();
	mesh_instance = RS::get_singleton()->instance_create();
	RS::get_singleton()->instance_set_scenario(mesh_instance, scenario);
}

EditorMeshPreviewPlugin::~EditorMeshPreviewPlugin() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	//RS::get_singleton()->free(sphere);
	RS::get_singleton()->free_rid(mesh_instance);
	RS::get_singleton()->free_rid(viewport);
	RS::get_singleton()->free_rid(light);
	RS::get_singleton()->free_rid(light_instance);
	RS::get_singleton()->free_rid(light2);
	RS::get_singleton()->free_rid(light_instance2);
	RS::get_singleton()->free_rid(camera);
	RS::get_singleton()->free_rid(camera_attributes);
	RS::get_singleton()->free_rid(scenario);
}

///////////////////////////////////////////////////////////////////////////

void EditorFontPreviewPlugin::abort() {
	draw_requester.abort();
}

bool EditorFontPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Font");
}

Ref<Texture2D> EditorFontPreviewPlugin::generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Font> sampled_font = ResourceLoader::load(p_path);
	ERR_FAIL_COND_V(sampled_font.is_null(), Ref<Texture2D>());

	String sample;
	static const String sample_base = U"12漢字ԱբΑαАбΑαאבابܐܒހށआআਆઆଆஆఆಆആආกิກິༀကႠა한글ሀᎣᐁᚁᚠᜀᜠᝀᝠកᠠᤁᥐAb😀";
	for (int i = 0; i < sample_base.length(); i++) {
		if (sampled_font->has_char(sample_base[i])) {
			sample += sample_base[i];
		}
	}
	if (sample.is_empty()) {
		sample = sampled_font->get_supported_chars().substr(0, 6);
	}
	Vector2 size = sampled_font->get_string_size(sample, HORIZONTAL_ALIGNMENT_LEFT, -1, 50);

	Vector2 pos;

	pos.x = 64 - size.x / 2;
	pos.y = 80;

	const Color c = GLOBAL_GET("rendering/environment/defaults/default_clear_color");
	const float fg = c.get_luminance() < 0.5 ? 1.0 : 0.0;
	sampled_font->draw_string(canvas_item, pos, sample, HORIZONTAL_ALIGNMENT_LEFT, -1.f, 50, Color(fg, fg, fg));

	draw_requester.request_and_wait(viewport);

	RS::get_singleton()->canvas_item_clear(canvas_item);

	Ref<Image> img = RS::get_singleton()->texture_2d_get(viewport_texture);
	ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

	img->convert(Image::FORMAT_RGBA8);

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);
	post_process_preview(img);

	return ImageTexture::create_from_image(img);
}

Ref<Texture2D> EditorFontPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	String path = p_from->get_path();
	if (!FileAccess::exists(path)) {
		return Ref<Texture2D>();
	}
	return generate_from_path(path, p_size, p_metadata);
}

EditorFontPreviewPlugin::EditorFontPreviewPlugin() {
	viewport = RS::get_singleton()->viewport_create();
	RS::get_singleton()->viewport_set_update_mode(viewport, RS::VIEWPORT_UPDATE_DISABLED);
	RS::get_singleton()->viewport_set_size(viewport, 128, 128);
	RS::get_singleton()->viewport_set_active(viewport, true);
	viewport_texture = RS::get_singleton()->viewport_get_texture(viewport);

	canvas = RS::get_singleton()->canvas_create();
	canvas_item = RS::get_singleton()->canvas_item_create();

	RS::get_singleton()->viewport_attach_canvas(viewport, canvas);
	RS::get_singleton()->canvas_item_set_parent(canvas_item, canvas);
}

EditorFontPreviewPlugin::~EditorFontPreviewPlugin() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RS::get_singleton()->free_rid(canvas_item);
	RS::get_singleton()->free_rid(canvas);
	RS::get_singleton()->free_rid(viewport);
}

////////////////////////////////////////////////////////////////////////////

static const real_t GRADIENT_PREVIEW_TEXTURE_SCALE_FACTOR = 4.0;

bool EditorGradientPreviewPlugin::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Gradient");
}

bool EditorGradientPreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

Ref<Texture2D> EditorGradientPreviewPlugin::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Gradient> gradient = p_from;
	if (gradient.is_valid()) {
		Ref<GradientTexture1D> ptex;
		ptex.instantiate();
		ptex->set_width(p_size.width * GRADIENT_PREVIEW_TEXTURE_SCALE_FACTOR * EDSCALE);
		ptex->set_gradient(gradient);
		return ImageTexture::create_from_image(ptex->get_image());
	}
	return Ref<Texture2D>();
}
