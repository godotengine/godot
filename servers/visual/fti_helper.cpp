/**************************************************************************/
/*  fti_helper.cpp                                                        */
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

#include "fti_helper.h"

#include "core/engine.h"
#include "core/math/transform_interpolator.h"
#include "servers/visual_server.h"

FTIHelper::Handle FTIHelper::instance_create(RID p_instance) {
	Handle h;
	Instance *i = _instances.request(h.id());
	i->clear();
	i->revision += 1;
	h.set_revision(i->revision);
	i->instance = p_instance;

	return h;
}

bool FTIHelper::instance_free(Handle h_instance) {
	ERR_FAIL_COND_V(!get_instance(h_instance), false);
	_instances.free(h_instance.id());

	// Remove from all lists.
	_instance_frame_list.erase_multiple_unordered(h_instance);
	_instance_tick_list_curr->erase_multiple_unordered(h_instance);
	_instance_tick_list_prev->erase_multiple_unordered(h_instance);
	return true;
}

FTIHelper::Instance *FTIHelper::get_instance(Handle h_instance) {
	ERR_FAIL_COND_V(h_instance.id() >= _instances.reserved_size(), nullptr);
	Instance &i = _instances[h_instance.id()];
	ERR_FAIL_COND_V(h_instance.revision() != i.revision, nullptr);
	return &i;
}

void FTIHelper::instance_changed(Instance &r_instance, Handle h_instance) {
	// Add to tick list
	if (!r_instance.on_tick_list) {
		r_instance.on_tick_list = true;
		_instance_tick_list_curr->push_back(h_instance);
	}

	// Add to frame list.
	if (!r_instance.on_frame_list) {
		r_instance.on_frame_list = true;
		_instance_frame_list.push_back(h_instance);
	}
}

void FTIHelper::instance_set_transform(Handle h_instance, const Transform &p_xform) {
	Instance *i = get_instance(h_instance);
	ERR_FAIL_NULL(i);
	i->curr = p_xform;
	instance_changed(*i, h_instance);
}

void FTIHelper::instance_reset_physics_interpolation(Handle h_instance) {
	Instance *i = get_instance(h_instance);
	ERR_FAIL_NULL(i);
	i->pump();
	instance_changed(*i, h_instance);
}

void FTIHelper::tick_update() {
	LocalVector<Handle> &curr = *_instance_tick_list_curr;
	LocalVector<Handle> &prev = *_instance_tick_list_prev;

	// First detect on the previous list but not on this tick list.
	for (uint32_t n = 0; n < prev.size(); n++) {
		const Handle &h = prev[n];
		Instance *i = get_instance(h);
		if (i) {
			if (!i->on_tick_list) {
				// Needs a reset so jittering will stop.
				i->pump();

				// Remove from interpolation list.
				if (i->on_frame_list) {
					i->on_frame_list = false;
					_instance_frame_list.erase_unordered(h);
				}
			}
		}
	}

	// Now pump all on the current list.
	for (uint32_t n = 0; n < curr.size(); n++) {
		const Handle &h = curr[n];
		Instance *i = get_instance(h);
		if (i) {
			// Reset, needs to be marked each tick.
			i->on_tick_list = false;
		}

		// Pump.
		i->pump();
	}

	// Clear previous list and flip.
	prev.clear();

	SWAP(_instance_tick_list_curr, _instance_tick_list_prev);
}

void FTIHelper::frame_update() {
	float f = Engine::get_singleton()->get_physics_interpolation_fraction();

	Transform x;
	VisualServer *vs = VisualServer::get_singleton();

	for (uint32_t n = 0; n < _instance_frame_list.size(); n++) {
		const Handle &h = _instance_frame_list[n];
		Instance *i = get_instance(h);
		if (i) {
			TransformInterpolator::interpolate_transform(i->prev, i->curr, x, f);

			// Send to server.
			vs->instance_set_transform(i->instance, x);
		}
	}
}
