/**************************************************************************/
/*  noise_texture_generator.h                                             */
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

#include "noise.h"
#include "noise_texture_2d.h"
#include "noise_texture_3d.h"

#include "core/object/worker_thread_pool.h"
#include "core/templates/a_hash_map.h"

class NoiseTextureGenerator : public Object {
	GDCLASS(NoiseTextureGenerator, Object);

	static Mutex bake_mutex;
	static Mutex generator_task_mutex;

	static NoiseTextureGenerator *singleton;

	static bool use_threads;

	struct GeneratorTask2D {
		enum TaskStatus {
			BAKING_STARTED,
			BAKING_FINISHED,
			BAKING_FAILED,
			CALLBACK_DISPATCHED,
			CALLBACK_FAILED,
		};

		Ref<NoiseTexture2D> noise_texture;
		Ref<Noise> noise;
		Callable callback;
		Ref<Image> noise_data;
		WorkerThreadPool::TaskID thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
		GeneratorTask2D::TaskStatus status = GeneratorTask2D::TaskStatus::BAKING_STARTED;
	};

	static AHashMap<WorkerThreadPool::TaskID, GeneratorTask2D *> generator_tasks_2d;
	static void generator_noise_texture_2d_thread_bake(void *p_arg);
	static AHashMap<Ref<NoiseTexture2D>, GeneratorTask2D *> baking_noise_texture_2d;
	static void sync_2d_tasks();

	struct GeneratorTask3D {
		enum TaskStatus {
			BAKING_STARTED,
			BAKING_FINISHED,
			BAKING_FAILED,
			CALLBACK_DISPATCHED,
			CALLBACK_FAILED,
		};

		Ref<NoiseTexture3D> noise_texture;
		Ref<Noise> noise;
		Callable callback;
		Vector<Ref<Image>> noise_data;
		WorkerThreadPool::TaskID thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
		GeneratorTask3D::TaskStatus status = GeneratorTask3D::TaskStatus::BAKING_STARTED;
	};

	static AHashMap<WorkerThreadPool::TaskID, GeneratorTask3D *> generator_tasks_3d;
	static void generator_noise_texture_3d_thread_bake(void *p_arg);
	static AHashMap<Ref<NoiseTexture3D>, GeneratorTask3D *> baking_noise_texture_3d;
	static void sync_3d_tasks();

	static bool generator_emit_callback(const Callable &p_callback);

protected:
	static void _bind_methods();

public:
	static NoiseTextureGenerator *get_singleton();

	NoiseTextureGenerator();
	~NoiseTextureGenerator();

	void init();
	void sync();
	void finish();

	static bool is_noise_texture_2d_baking(Ref<NoiseTexture2D> p_noise_texture);
	static void bake_noise_texture_2d(Ref<NoiseTexture2D> p_noise_texture);
	static void bake_noise_texture_2d_async(Ref<NoiseTexture2D> p_noise_texture, const Callable &p_callback = Callable());

	static bool is_noise_texture_3d_baking(Ref<NoiseTexture3D> p_noise_texture);
	static void bake_noise_texture_3d(Ref<NoiseTexture3D> p_noise_texture);
	static void bake_noise_texture_3d_async(Ref<NoiseTexture3D> p_noise_texture, const Callable &p_callback = Callable());
};
