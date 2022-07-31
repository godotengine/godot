/*************************************************************************/
/*  utilities.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef UTILITIES_GLES3_H
#define UTILITIES_GLES3_H

#ifdef GLES3_ENABLED

#include "servers/rendering/storage/utilities.h"

#include "platform_config.h"
#ifndef OPENGL_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include OPENGL_INCLUDE_H
#endif

namespace GLES3 {

class Utilities : public RendererUtilities {
private:
	static Utilities *singleton;

public:
	static Utilities *get_singleton() { return singleton; }

	Utilities();
	~Utilities();

	// Buffer size is specified in bytes
	static Vector<uint8_t> buffer_get_data(GLenum p_target, GLuint p_buffer, uint32_t p_buffer_size);

	/* INSTANCES */

	virtual RS::InstanceType get_base_type(RID p_rid) const override;
	virtual bool free(RID p_rid) override;

	/* DEPENDENCIES */

	virtual void base_update_dependency(RID p_base, DependencyTracker *p_instance) override;

	/* VISIBILITY NOTIFIER */
	virtual RID visibility_notifier_allocate() override;
	virtual void visibility_notifier_initialize(RID p_notifier) override;
	virtual void visibility_notifier_free(RID p_notifier) override;

	virtual void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) override;
	virtual void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) override;

	virtual AABB visibility_notifier_get_aabb(RID p_notifier) const override;
	virtual void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) override;

	/* TIMING */

	struct Info {
		uint64_t texture_mem = 0;
		uint64_t vertex_mem = 0;

		struct Render {
			uint32_t object_count;
			uint32_t draw_call_count;
			uint32_t material_switch_count;
			uint32_t surface_switch_count;
			uint32_t shader_rebind_count;
			uint32_t vertices_count;
			uint32_t _2d_item_count;
			uint32_t _2d_draw_call_count;

			void reset() {
				object_count = 0;
				draw_call_count = 0;
				material_switch_count = 0;
				surface_switch_count = 0;
				shader_rebind_count = 0;
				vertices_count = 0;
				_2d_item_count = 0;
				_2d_draw_call_count = 0;
			}
		} render, render_final, snap;

		Info() {
			render.reset();
			render_final.reset();
		}

	} info;

	virtual void capture_timestamps_begin() override {}
	virtual void capture_timestamp(const String &p_name) override {}
	virtual uint32_t get_captured_timestamps_count() const override {
		return 0;
	}
	virtual uint64_t get_captured_timestamps_frame() const override {
		return 0;
	}
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override {
		return 0;
	}
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override {
		return 0;
	}
	virtual String get_captured_timestamp_name(uint32_t p_index) const override {
		return String();
	}

	//	void render_info_begin_capture() override;
	//	void render_info_end_capture() override;
	//	int get_captured_render_info(RS::RenderInfo p_info) override;

	//	int get_render_info(RS::RenderInfo p_info) override;

	/* MISC */

	virtual void update_dirty_resources() override;
	virtual void set_debug_generate_wireframes(bool p_generate) override;

	virtual bool has_os_feature(const String &p_feature) const override;

	virtual void update_memory_info() override;

	virtual uint64_t get_rendering_info(RS::RenderingInfo p_info) override;
	virtual String get_video_adapter_name() const override;
	virtual String get_video_adapter_vendor() const override;
	virtual RenderingDevice::DeviceType get_video_adapter_type() const override;
	virtual String get_video_adapter_api_version() const override;
};

} // namespace GLES3

#endif // GLES3_ENABLED

#endif // UTILITIES_GLES3_H
