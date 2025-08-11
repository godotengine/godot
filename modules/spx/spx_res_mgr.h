/**************************************************************************/
/*  spx_platform_mgr.h                                                       */
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

#ifndef SPX_RES_MGR_H
#define SPX_RES_MGR_H

#include "gdextension_spx_ext.h"
#include "scene/resources/sprite_frames.h"
#include "servers/audio/audio_stream.h"
#include "spx_base_mgr.h"
#include "scene/resources/font.h"

class AudioStreamMP3;
class AudioStreamWAV;
class Texture2D;

class SpxResMgr : SpxBaseMgr {
	SPXCLASS(SpxPlatformMgr, SpxBaseMgr)

public:
	virtual ~SpxResMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

private:
	HashMap<String, Ref<Texture2D>> cached_texture;
	HashMap<String, Ref<AudioStream>> cached_audio;
	bool is_load_direct;
	String game_data_root = "res://";
	Ref<SpriteFrames> anim_frames;
	bool is_dynamic_anim = false;
	// store animation frame offset information: anim_name -> frame_offset_list
	HashMap<String, Vector<Vector2>> animation_frame_offsets;
private:
	static Ref<AudioStreamWAV> _load_wav(const String &path);
	static Ref<AudioStream> _load_mp3(const String &path);
	Ref<Texture2D> _load_texture_direct(const String &p_path);
	Ref<AudioStream> _load_audio_direct(const String &p_path);
	String _to_engine_path(const String &p_path);

public:
	void on_awake() override;
	Ref<Texture2D> load_texture(String path, GdBool direct = false);
	Ref<AudioStream> load_audio(String path, GdBool direct = false);
	Ref<Texture2D> _reload_texture(String path);
	void set_game_datas(String path, Vector<String> files);
	Ref<SpriteFrames> get_anim_frames(const String& anim_name);
	String get_anim_key_name(const String& sprite_type_name,const String& anim_name);
	bool is_dynamic_anim_mode() const;
	Vector2 get_animation_frame_offset(String anim_key, int frame_index);

public:
	void create_animation(GdString sprite_type_name,GdString anim_name, GdString context, GdInt fps, GdBool is_altas);
	void set_load_mode(GdBool is_direct_mode);
	GdBool get_load_mode();
	GdRect2 get_bound_from_alpha(GdString p_path);
	GdVec2 get_image_size(GdString p_path);
	GdString read_all_text(GdString p_path);
	GdBool has_file(GdString p_path);
	void reload_texture(GdString path);
	void free_str(GdString str);
	void set_default_font(GdString font_path);
};

#endif // SPX_RES_MGR_H
