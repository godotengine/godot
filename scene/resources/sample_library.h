/*************************************************************************/
/*  sample_library.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SAMPLE_LIBRARY_H
#define SAMPLE_LIBRARY_H

#include "resource.h"
#include "scene/resources/sample.h"
#include "map.h"

class SampleLibrary : public Resource {

	GDCLASS(SampleLibrary,Resource);

	struct SampleData {

		Ref<Sample> sample;
		float db;
		float pitch_scale;

		SampleData() { db=0; pitch_scale=1; }
	};

	Map<StringName,SampleData > sample_map;

	Array _get_sample_list() const;
protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:



	void add_sample(const StringName& p_name, const Ref<Sample>& p_sample);
	bool has_sample(const StringName& p_name) const;
	void sample_set_volume_db(const StringName& p_name, float p_db);
	float sample_get_volume_db(const StringName& p_name) const;
	void sample_set_pitch_scale(const StringName& p_name, float p_pitch);
	float sample_get_pitch_scale(const StringName& p_name) const;
	Ref<Sample> get_sample(const StringName& p_name) const;
	void get_sample_list(List<StringName> *p_samples) const;
	void remove_sample(const StringName& p_name);
	StringName get_sample_idx(int p_idx) const;

	SampleLibrary();
};

#endif // SAMPLE_LIBRARY_H
