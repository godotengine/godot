/**************************************************************************/
/*  audio_server_debug.h                                                  */
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

#pragma once

#ifdef DEBUG_ENABLED

#include "core/math/color.h"
#include "core/variant/variant.h"

class AudioServerDebug {
	static AudioServerDebug *singleton;

	bool debug_audio_2d_visualization_enabled = false;

	RID debug_audio_2d_visualization_circle_mesh_rid;
	Vector<RID> debug_audio_2d_visualization_rings_mesh_rids;
	RID debug_audio_2d_visualization_outline_mesh_rid;

	int debug_audio_2d_visualization_ring_count;
	Color debug_audio_2d_visualization_color;
	int debug_audio_2d_visualization_mode;

public:
	static AudioServerDebug *get_singleton();

	void register_settings();
	void update_from_settings();

	void generate_rids();
	void clear_rids();

	RID get_debug_audio_2d_visualization_circle_mesh_rid() const;
	Vector<RID> get_debug_audio_2d_visualization_rings_mesh_rids() const;
	RID get_debug_audio_2d_visualization_outline_mesh_rid() const;

	int get_debug_audio_2d_visualization_ring_count() const;
	void set_debug_audio_2d_visualization_ring_count(int p_count);

	Color get_debug_audio_2d_visualization_color() const;
	void set_debug_audio_2d_visualization_color(const Color &p_color);

	int get_debug_audio_2d_visualization_mode() const;
	void set_debug_audio_2d_visualization_mode(int p_mode);

	void set_debug_audio_2d_visualization_enabled(bool p_enabled);
	bool get_debug_audio_2d_visualization_enabled() const;

	AudioServerDebug();
	~AudioServerDebug();
};

#endif // DEBUG_ENABLED
