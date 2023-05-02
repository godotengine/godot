/**************************************************************************/
/* custom_iterator.h                                                      */
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

#ifndef CUSTOM_ITERATOR_H
#define CUSTOM_ITERATOR_H

#include "core/object/object.h"
#include "iteration_server.h"
#include "main/main_timer_sync.h"

class CustomIterator : public Object {
	GDCLASS(CustomIterator, Object)
protected:
	static void _bind_methods();

public:
	virtual String get_name() const = 0;

	virtual IterationServer::IteratorType get_type() const = 0;

	virtual bool mixed_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) { return false; }

	virtual bool process_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) { return false; }

	virtual bool physics_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) { return false; }

	virtual bool audio_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) { return false; }
};

#endif //CUSTOM_ITERATOR_H
