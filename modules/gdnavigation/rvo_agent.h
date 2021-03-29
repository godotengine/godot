/*************************************************************************/
/*  rvo_agent.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RVO_AGENT_H
#define RVO_AGENT_H

#include "core/object/class_db.h"
#include "nav_rid.h"

#include <Agent.h>

/**
	@author AndreaCatania
*/

class NavMap;

class RvoAgent : public NavRid {
	struct AvoidanceComputedCallback {
		ObjectID id;
		StringName method;
		Variant udata;
		Variant new_velocity;
	};

	NavMap *map = nullptr;
	RVO::Agent agent;
	AvoidanceComputedCallback callback;
	uint32_t map_update_id = 0;

public:
	RvoAgent();

	void set_map(NavMap *p_map);
	NavMap *get_map() {
		return map;
	}

	RVO::Agent *get_agent() {
		return &agent;
	}

	bool is_map_changed();

	void set_callback(ObjectID p_id, const StringName p_method, const Variant p_udata = Variant());
	bool has_callback() const;

	void dispatch_callback();
};

#endif // RVO_AGENT_H
