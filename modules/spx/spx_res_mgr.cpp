/**************************************************************************/
/*  spx_platform_mgr.cpp                                                     */
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

#include "spx_res_mgr.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/image_loader.h"
#include "modules/minimp3/audio_stream_mp3.h"
#include "modules/modules_enabled.gen.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/main/window.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/audio_stream_wav.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/sprite_frames.h"
#include "scene/theme/default_theme.h"
#include "scene/theme/theme_db.h"
#include "spx_engine.h"
#include "scene/resources/audio_importer_wav.h"
#include "spx_platform_mgr.h"
#ifdef TOOLS_ENABLED
#include "editor/import/resource_importer_wav.h"
#include "modules/minimp3/resource_importer_mp3.h"
#endif

#ifdef MODULE_SVG_ENABLED
#include "modules/svg/svg_utils.h"
#endif

#define platformMgr SpxEngine::get_singleton()->get_platform()

void SpxResMgr::on_awake() {
	SpxBaseMgr::on_awake();
	is_load_direct = true;
	anim_frames.instantiate();
}

bool SpxResMgr::is_dynamic_anim_mode() const {
	return is_dynamic_anim;
}

String SpxResMgr::_to_engine_path(const String &p_path){
	String path = p_path;
	if (!path.begins_with(platformMgr->_get_persistant_data_dir()) && game_data_root != "res://") {
		if (path.begins_with("../")) {
			path = path.substr(3, -1);
		}
		path = game_data_root + "/" + path;
	}
	return path;
}

Ref<AudioStreamWAV> SpxResMgr::_load_wav(const String &path) {
	Ref<AudioStreamWAV> sample;
	AudioImporterWav::import_asset(sample, path);
	return sample;
}

static Ref<AudioStream> _import_mp3(const String &p_path) {
#ifdef MODULE_MINIMP3_ENABLED
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(f.is_null(), Ref<AudioStreamMP3>());

	uint64_t len = f->get_length();

	Vector<uint8_t> data;
	data.resize(len);
	uint8_t *w = data.ptrw();

	f->get_buffer(w, len);

	Ref<AudioStreamMP3> mp3_stream;
	mp3_stream.instantiate();

	mp3_stream->set_data(data);
	ERR_FAIL_COND_V(!mp3_stream->get_data().size(), Ref<AudioStreamMP3>());
	return mp3_stream;
#else
	Ref<AudioStream> mp3_stream;
	return mp3_stream;
#endif
}

Ref<AudioStream> SpxResMgr::_load_mp3(const String &path) {
	return _import_mp3(path);
}

Ref<AudioStream> SpxResMgr::_load_audio_direct(const String &p_path) {
	String path = _to_engine_path(p_path);
	if (cached_audio.has(path)) {
		return cached_audio[path];
	}
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null()) {
		print_line("Failed to open audio file: " + path);
		return Ref<AudioStreamWAV>();
	}
	Ref<AudioStream> res;
	const String ext = path.get_extension().to_lower();
	if (ext == "mp3") {
		res = _load_mp3(path);
	} else if (ext == "wav") {
		res = _load_wav(path);
	} else {
		print_error("unknown audio extension " + ext + " path=" + path);
	}
	cached_audio.insert(path, res);
	return res;
}

static void _load_image(String path, Ref<Image> p_image){
	Error err = ImageLoader::load_image(path, p_image);
	if (err != OK) {
		// Failed to load image , so give a pink image
		// pink color
		PackedByteArray data;
		for (int i = 0; i < 4 * 4; i++) {
			data.append(255); // R
			data.append(0); // G
			data.append(255); // B
			data.append(128); // A
		}
		p_image->set_data(4, 4, false, Image::FORMAT_RGBA8, data);
	}
}

Ref<Texture2D> SpxResMgr::_load_texture_direct(const String &p_path) {
	String path = _to_engine_path(p_path);
	// data in tmp dir would not keep in cache
	if (cached_texture.has(path)) {
		return cached_texture[path];
	}

	Ref<Image> image;
	image.instantiate();

	_load_image(path, image);

	Ref<ImageTexture> texture = ImageTexture::create_from_image(image);
	cached_texture.insert(path, texture);
	return texture;
}
Ref<Texture2D> SpxResMgr::_reload_texture(String path) {
	if (cached_texture.has(path)) {
		auto tex = (Ref<ImageTexture>)cached_texture[path];
		Ref<Image> image;
		image.instantiate();
		_load_image(path, image);
		tex->set_image(image);
		cached_texture.erase(path);
		cached_texture.insert(path, tex);
		return tex;
	}else{
		return _load_texture_direct(path);
	}
}

void SpxResMgr::reload_texture(GdString path) {
	auto path_str = SpxStr(path);
	_reload_texture(path_str);
}

Ref<Texture2D> SpxResMgr::load_texture(String path, GdBool direct) {
	if (!is_load_direct && !direct) {
		Ref<Resource> res = ResourceLoader::load(path);
		if (res.is_null()) {
			print_line("load texture failed !", path);
			return Ref<Texture2D>();
		}
		return res;
	} else {
		return _load_texture_direct(path);
	}
}

void SpxResMgr::set_game_datas(String path, Vector<String> files) {
	print_line("SpxResMgr::set_game_datas", path);
	game_data_root = path;
	platformMgr->_set_persistant_data_dir(path);
}

Ref<AudioStream> SpxResMgr::load_audio(String path, GdBool direct) {
	if (!is_load_direct && !direct) {
		Ref<Resource> res = ResourceLoader::load(path);
		if (res.is_null()) {
			print_line("load audio failed !", path);
			return Ref<AudioStream>();
		}
		return res;
	} else {
		return _load_audio_direct(path);
	}
}

Ref<SpriteFrames> SpxResMgr::get_anim_frames(const String &anim_name) {
	return anim_frames;
}

String SpxResMgr::get_anim_key_name(const String &sprite_type_name, const String &anim_name) {
	return sprite_type_name + "::" + anim_name;
}


void SpxResMgr::create_animation(GdString p_sprite_type_name, GdString p_anim_name, GdString p_context,GdInt fps, GdBool is_altas) {
	is_dynamic_anim = true;
	auto sprite_type_name = SpxStr(p_sprite_type_name);
	auto clip_name = SpxStr(p_anim_name);
	auto context = SpxStr(p_context);
	auto anim_key = get_anim_key_name(sprite_type_name, clip_name);
	auto frames = anim_frames;
	if (frames->has_animation(anim_key)) {
		return ;
	}
	frames->add_animation(anim_key);
	frames->set_animation_speed(anim_key,fps);
	
	// store frame offset information
	Vector<Vector2> frame_offsets;
	
	if (!is_altas) {
		auto strs = context.split(";");
		for (const String &path_with_offset : strs) {
			Vector2 offset(0, 0);  // default offset
			String path = path_with_offset;
			
			// check if contains offset information (format: path|offset_x,offset_y)
			if (path_with_offset.contains("|")) {
				auto parts = path_with_offset.split("|");
				if (parts.size() == 2) {
					path = parts[0];
					auto offset_parts = parts[1].split(",");
					if (offset_parts.size() >= 2) {
						offset.x = offset_parts[0].to_float();
						offset.y = offset_parts[1].to_float();
					}
				}
			}
			
			Ref<Texture2D> texture = load_texture(path);
			if (!texture.is_valid()) {
				print_error("animation parse error" + sprite_type_name + " " + anim_key + " can not find path " + path);
				return ;
			}
			frames->add_frame(anim_key, texture);
			frame_offsets.push_back(offset);
		}
	} else {
		auto strs = context.split(";");
		if (strs.size() < 2) {
			print_error("create_animation context error missing \";\"? : " + context);
			return ;
		}
		auto path = strs[0];
		Ref<Texture2D> altas_texture = load_texture(path);
		if (!altas_texture.is_valid()) {
			print_error("animation parse error" + sprite_type_name + " " + anim_key + " can not find path " + path);
			return ;
		}

		String param_str = strs[1];
		
		// support two formats:
		// 1. old format: x1,y1,w1,h1,x2,y2,w2,h2
		// 2. new format: x1,y1,w1,h1,offset_x1,offset_y1;x2,y2,w2,h2,offset_x2,offset_y2
		
		Vector<double> params;
		Vector<Vector2> frame_offsets_atlas;
		
		if (param_str.contains(";")) {
			// new format: each frame data is separated by semicolon, each frame can contain 6 parameters (region + offset)
			auto frame_strs = param_str.split(";");
			for (const String &frame_str : frame_strs) {
				auto values = frame_str.split(",");
				if (values.size() >= 4) {
					for (int i = 0; i < 4; i++) {
						params.push_back(values[i].to_float());
					}
					if (values.size() >= 6) {
						Vector2 offset(values[4].to_float(), values[5].to_float());
						frame_offsets_atlas.push_back(offset);
					} else {
						frame_offsets_atlas.push_back(Vector2(0, 0));
					}
				}
			}
		} else {
			// old format: all parameters are separated by commas
			auto paramStrs = param_str.split(",");
			
			if (paramStrs.size() % 4 != 0) {
				print_error("create_animation context error, params count % 4 != 0: " + context +" size = "+ paramStrs.size() );
				return ;
			}
			
			for (const String &str : paramStrs) {
				params.push_back(str.to_float());
			}
			
			// old format use default offset
			int frame_count = params.size() / 4;
			for (int i = 0; i < frame_count; i++) {
				frame_offsets_atlas.push_back(Vector2(0, 0));
			}
		}
		
		if (params.size() % 4 != 0) {
			print_error("create_animation context error, params count % 4 != 0: " + context +" size = "+ params.size() );
			return ;
		}
		
		auto count = params.size() / 4;
		for (int i = 0; i < count; i++) {
			Vector2 offset(0, 0);  // default offset
			
			// use parsed offset
			if (i < frame_offsets_atlas.size()) {
				offset = frame_offsets_atlas[i];
			}
			
			Ref<AtlasTexture> texture;
			texture.instantiate();
			texture->set_atlas(altas_texture);
			auto offset_param = i * 4;
			Rect2 rect2;
			rect2.position = Vector2(params[offset_param + 0], params[offset_param + 1]);
			rect2.size = Vector2(params[offset_param + 2], params[offset_param + 3]);
			texture->set_region(rect2);
			frames->add_frame(anim_key, texture);
			frame_offsets.push_back(offset);
		}
	}
	
	// store animation frame offset information
	animation_frame_offsets[anim_key] = frame_offsets;
}
void SpxResMgr::set_load_mode(GdBool is_direct_mode) {
	is_load_direct = is_direct_mode;
}
GdBool SpxResMgr::get_load_mode() {
	return is_load_direct;
}

GdRect2 SpxResMgr::get_bound_from_alpha(GdString path) {
	auto path_str = SpxStr(path);

	Ref<Texture2D> image = load_texture(path_str);
	if (image == nullptr) {
		print_line("Load texture failed ", path_str);
		return GdRect2(Vector2(0,0),Size2(4,4));
	}
	int width = image->get_width();
	int height = image->get_height();

	int min_x = width;
	int min_y = height;
	int max_x = 0;
	int max_y = 0;
	bool has_alpha = false;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (image->is_pixel_opaque(x, y)) { // Check if the pixel is not fully transparent
				has_alpha = true;
				if (x < min_x)
					min_x = x;
				if (y < min_y)
					min_y = y;
				if (x > max_x)
					max_x = x;
				if (y > max_y)
					max_y = y;
			}
		}
	}
	if (!has_alpha) {
		return Rect2();
	}

	return Rect2(Vector2(min_x, min_y), Vector2(max_x - min_x + 1, max_y - min_y + 1));
}

GdVec2 SpxResMgr::get_image_size(GdString path) {
	auto path_str = SpxStr(path);
	Ref<Texture2D> value = load_texture(path_str);
	if (value.is_valid()) {
		return value->get_size();
	} else {
		print_error("can not find a texture: " + path_str);
	}
	return GdVec2(1, 1);
}

void SpxResMgr::free_str(GdString str_ptr) {
	free_return_cstr(str_ptr);
}
GdString SpxResMgr::read_all_text(GdString p_path) {
	auto path = SpxStr(p_path);
	path = _to_engine_path(path);
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	String value = "";
	if (file.is_null()) {
		print_line("Unable to open file.", path);
	} else {
		String file_content;
		while (!file->eof_reached()) {
			String line = file->get_line();
			file_content += line + "\n";
		}
		value = file_content;
	}
	file->close();
	return SpxReturnStr(value);
}

GdBool SpxResMgr::has_file(GdString p_path) {
	auto path = SpxStr(p_path);
	path = _to_engine_path(path);
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	return !file.is_null();
}


void SpxResMgr::set_default_font(GdString font_path) {
	String path = SpxStr(font_path);
	Vector<uint8_t> font_data;
	Ref<FontFile> rawFont = ResourceLoader::load(path);
	if (!rawFont.is_null()) {
		font_data = rawFont->get_data();
	}else{
		String engine_path = _to_engine_path(path);
		Ref<FileAccess> f = FileAccess::open(engine_path, FileAccess::READ);
		if (f.is_null()) {
			ERR_PRINT("Can not open font file: " + path + " engine_path= " + engine_path );
			return ;
		}
		font_data.resize(f->get_length());
		f->get_buffer(font_data.ptrw(), font_data.size());
	}
	// update svg
#ifdef MODULE_SVG_ENABLED
	SVGUtils::set_default_font(font_data.ptrw(), (int)font_data.size());
#endif

	// update theme
	Ref<FontFile> font;
	font.instantiate();
	font->set_font_style(0);
	font->set_data(font_data);
	font->set_antialiasing(TextServer::FONT_ANTIALIASING_GRAY);
	font->set_force_autohinter(false);
	font->set_hinting(TextServer::HINTING_LIGHT);
	font->set_subpixel_positioning(TextServer::SUBPIXEL_POSITIONING_AUTO);
	font->set_multichannel_signed_distance_field(false);
	font->set_generate_mipmaps(false);
	font->set_fixed_size(0);
	font->set_allow_system_fallback(true);
	ThemeDB::get_singleton()->set_default_font(font);
}

Vector2 SpxResMgr::get_animation_frame_offset(String anim_key, int frame_index) {
	if (animation_frame_offsets.has(anim_key)) {
		const Vector<Vector2>& offsets = animation_frame_offsets[anim_key];
		if (frame_index >= 0 && frame_index < offsets.size()) {
			return offsets[frame_index];
		}
	}
	return Vector2(0, 0);
}
