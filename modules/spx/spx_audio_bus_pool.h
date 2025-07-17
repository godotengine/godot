/**************************************************************************/
/*  spx_audio_bus_pool.h                                                       */
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

#ifndef SPX_AUDIO_BUS_POOL_H
#define SPX_AUDIO_BUS_POOL_H

#include "core/string/ustring.h"
#include "gdextension_spx_ext.h"
#include "scene/main/node.h"

class SpxAudioBusPool {
	static SpxAudioBusPool *singleton;

public:
	static const int BUS_MASTER = 0;
	static const int BUS_SFX = 1;
	static const int BUS_MUSIC = 2;
	static StringName STR_BUS_MASTER;
	static StringName STR_BUS_SFX;
	static StringName STR_BUS_MUSIC;

private:
	static const int DEFAULT_BUS_COUNT = 4; // Default number of buses including master
	static const int BUS_EXPANSION_SIZE = 4; // Number of buses to add when expanding

	Vector<int> free_buses; // Pool of available bus IDs
	HashMap<int, bool> active_buses; // Track active buses
	int current_bus_count = 0; // Current total number of buses
private:
	void expand_buses();
	bool is_valid_bus(int id);

public:
	static SpxAudioBusPool *get_singleton();
	static void init();
	int alloc();
	void free(int id);
	void set_volume(int id, GdFloat volume);
	GdFloat get_volume(int id);
	void set_pan(int id, GdFloat pan);
	GdFloat get_pan(int id);
	StringName get_bus_name(int id);
};
#endif // SPX_AUDIO_BUS_POOL_H
