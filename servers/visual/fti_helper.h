/**************************************************************************/
/*  fti_helper.h                                                          */
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

#ifndef FTI_HELPER_H
#define FTI_HELPER_H

#include "core/math/transform.h"
#include "core/pooled_list.h"
#include "core/rid.h"

#include <cstdint>

class FTIHelper {
	struct Handle {
		static const uint64_t INVALID = UINT64_MAX;

		// conversion operator
		operator uint64_t() const { return _data; }
		Handle(uint64_t p_value) { _data = p_value; }
		Handle() { _data = INVALID; }

		union {
			struct
			{
				uint32_t _id;
				uint32_t _revision;
			};
			uint64_t _data;
		};

		void set_invalid() { _data = INVALID; }
		bool is_invalid() const { return _data == INVALID; }
		uint32_t id() const { return _id; }
		uint32_t &id() { return _id; }
		void set_id(uint32_t p_id) { _id = p_id; }
		uint32_t revision() const { return _revision; }
		void set_revision(uint32_t p_revision) { _revision = p_revision; }

		bool operator==(const Handle &p_h) const { return _data == p_h._data; }
		bool operator!=(const Handle &p_h) const { return (*this == p_h) == false; }
	};

	struct Instance {
		uint32_t revision = 0;
		RID instance;
		Transform curr;
		Transform prev;
		bool on_frame_list = false;
		bool on_tick_list = false;
		void clear() {
			instance = RID();
			curr = Transform();
			prev = Transform();
			on_frame_list = false;
			on_tick_list = false;
		}
		void pump() {
			prev = curr;
		}
	};

	PooledList<Instance, uint32_t, true, true> _instances;

	LocalVector<Handle> _instance_frame_list;
	LocalVector<Handle> _instance_tick_list[2];
	LocalVector<Handle> *_instance_tick_list_curr = &_instance_tick_list[0];
	LocalVector<Handle> *_instance_tick_list_prev = &_instance_tick_list[1];

	// Check revisions etc. in case of user error.
	Instance *get_instance(Handle h_instance);
	void instance_changed(Instance &r_instance, Handle h_instance);

public:
	Handle instance_create(RID p_instance);
	bool instance_free(Handle h_instance);
	void instance_set_transform(Handle h_instance, const Transform &p_xform);
	void instance_reset_physics_interpolation(Handle h_instance);

	void tick_update();
	void frame_update();
};

#endif // FTI_HELPER_H
