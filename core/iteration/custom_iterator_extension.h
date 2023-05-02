/**************************************************************************/
/* custom_iterator_extension.h                                            */
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

#ifndef EXTENSION_ITERATOR_H
#define EXTENSION_ITERATOR_H

#include "custom_iterator.h"
#include "iteration_server.h"

class CustomIteratorExtension : public CustomIterator {
	GDCLASS(CustomIteratorExtension, CustomIterator)

protected:
	static void _bind_methods();

public:
	GDVIRTUAL0RC(String, _get_name);
	GDVIRTUAL0RC(IterationServer::IteratorType, _get_type);

	GDVIRTUAL4R(bool, _mixed_iteration, float, float, GDExtensionPtr<MainFrameTime>, float);
	GDVIRTUAL4R(bool, _process_iteration, float, float, GDExtensionPtr<MainFrameTime>, float);
	GDVIRTUAL4R(bool, _physics_iteration, float, float, GDExtensionPtr<MainFrameTime>, float);
	GDVIRTUAL4R(bool, _audio_iteration, float, float, GDExtensionPtr<MainFrameTime>, float);

	bool mixed_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) override {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_mixed_iteration, p_process_delta, p_physics_delta, p_frame_time, p_time_scale, ret);
		return ret;
	}

	bool process_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) override {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_process_iteration, p_process_delta, p_physics_delta, p_frame_time, p_time_scale, ret);
		return ret;
	}

	bool physics_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) override {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_physics_iteration, p_process_delta, p_physics_delta, p_frame_time, p_time_scale, ret);
		return ret;
	}

	bool audio_iteration(float p_process_delta, float p_physics_delta, MainFrameTime *p_frame_time, float p_time_scale) override {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_audio_iteration, p_process_delta, p_physics_delta, p_frame_time, p_time_scale, ret);
		return ret;
	}

	String get_name() const override {
		String ret;
		GDVIRTUAL_REQUIRED_CALL(_get_name, ret);
		return ret;
	}

	IterationServer::IteratorType get_type() const override {
		IterationServer::IteratorType ret = IterationServer::ITERATOR_TYPE_UNSET;
		GDVIRTUAL_REQUIRED_CALL(_get_type, ret);
		return ret;
	}
};

#endif //EXTENSION_ITERATOR_H
