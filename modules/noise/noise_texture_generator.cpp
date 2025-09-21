/**************************************************************************/
/*  noise_texture_generator.cpp                                           */
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

#include "noise_texture_generator.h"

#include "servers/rendering/rendering_server.h"

NoiseTextureGenerator *NoiseTextureGenerator::singleton = nullptr;
bool NoiseTextureGenerator::use_threads = true;
Mutex NoiseTextureGenerator::bake_mutex;
Mutex NoiseTextureGenerator::generator_task_mutex;
AHashMap<Ref<NoiseTexture2D>, NoiseTextureGenerator::GeneratorTask2D *> NoiseTextureGenerator::baking_noise_texture_2d;
AHashMap<WorkerThreadPool::TaskID, NoiseTextureGenerator::GeneratorTask2D *> NoiseTextureGenerator::generator_tasks_2d;
AHashMap<Ref<NoiseTexture3D>, NoiseTextureGenerator::GeneratorTask3D *> NoiseTextureGenerator::baking_noise_texture_3d;
AHashMap<WorkerThreadPool::TaskID, NoiseTextureGenerator::GeneratorTask3D *> NoiseTextureGenerator::generator_tasks_3d;

NoiseTextureGenerator *NoiseTextureGenerator::get_singleton() {
	return singleton;
}

NoiseTextureGenerator::NoiseTextureGenerator() {
	singleton = this;

#ifndef THREADS_ENABLED
	use_threads = false;
#endif
}

NoiseTextureGenerator::~NoiseTextureGenerator() {
}

void NoiseTextureGenerator::bake_noise_texture_2d(Ref<NoiseTexture2D> p_noise_texture) {
	ERR_FAIL_COND(p_noise_texture.is_null());
	ERR_FAIL_COND(p_noise_texture->get_noise().is_null());

	if (is_noise_texture_2d_baking(p_noise_texture)) {
		ERR_FAIL_MSG("NoiseTexture2D is already baking. Wait for current bake to finish. Use 'NoiseTextureGenerator::is_noise_texture_2d_baking()' to check the bake state of a noise 2d texture.");
		return;
	}

	Ref<NoiseTexture2D> noise_texture = p_noise_texture;
	const Ref<Image> noise_data = noise_texture->bake_noise_data();
	noise_texture->set_data(noise_data);
}

void NoiseTextureGenerator::bake_noise_texture_2d_async(Ref<NoiseTexture2D> p_noise_texture, const Callable &p_callback) {
	ERR_FAIL_COND(p_noise_texture.is_null());
	ERR_FAIL_COND(p_noise_texture->get_noise().is_null());

	if (is_noise_texture_2d_baking(p_noise_texture)) {
		ERR_FAIL_MSG("NoiseTexture2D is already baking. Wait for current bake to finish. Use 'NoiseTextureGenerator::is_noise_texture_2d_baking()' to check the bake state of a noise 2d texture.");
		return;
	}

	if (!use_threads) {
		bake_noise_texture_2d(p_noise_texture);
		if (p_callback.is_valid()) {
			generator_emit_callback(p_callback);
		}
		return;
	}

	bake_mutex.lock();
	GeneratorTask2D *generator_task = memnew(GeneratorTask2D);
	baking_noise_texture_2d.insert(p_noise_texture, generator_task);
	bake_mutex.unlock();

	generator_task->noise_texture = p_noise_texture;
	generator_task->noise = p_noise_texture->get_noise();
	generator_task->callback = p_callback;
	generator_task->status = GeneratorTask2D::TaskStatus::BAKING_STARTED;
	generator_task->thread_task_id = WorkerThreadPool::get_singleton()->add_native_task(&NoiseTextureGenerator::generator_noise_texture_2d_thread_bake, generator_task, false, SNAME("NoiseTextureGenerator"));
	MutexLock generator_task_lock(generator_task_mutex);
	generator_tasks_2d.insert(generator_task->thread_task_id, generator_task);
}

void NoiseTextureGenerator::generator_noise_texture_2d_thread_bake(void *p_arg) {
	GeneratorTask2D *generator_task = static_cast<GeneratorTask2D *>(p_arg);

	Ref<NoiseTexture2D> noise_texture = generator_task->noise_texture;
	Ref<Image> noise_data = noise_texture->bake_noise_data();
	generator_task->noise_data = noise_data;

	generator_task->status = GeneratorTask2D::TaskStatus::BAKING_FINISHED;
}

bool NoiseTextureGenerator::is_noise_texture_2d_baking(Ref<NoiseTexture2D> p_noise_texture) {
	MutexLock baking_lock(bake_mutex);
	return baking_noise_texture_2d.has(p_noise_texture);
}

void NoiseTextureGenerator::sync_2d_tasks() {
	if (generator_tasks_2d.is_empty()) {
		return;
	}

	MutexLock generator_task_lock(generator_task_mutex);

	LocalVector<Callable> finished_callbacks;
	LocalVector<WorkerThreadPool::TaskID> finished_task_ids;

	for (KeyValue<WorkerThreadPool::TaskID, GeneratorTask2D *> &E : generator_tasks_2d) {
		if (WorkerThreadPool::get_singleton()->is_task_completed(E.key)) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
			finished_task_ids.push_back(E.key);

			GeneratorTask2D *generator_task = E.value;
			DEV_ASSERT(generator_task->status == GeneratorTask2D::TaskStatus::BAKING_FINISHED);

			baking_noise_texture_2d.erase(generator_task->noise_texture);
			if (generator_task->callback.is_valid()) {
				finished_callbacks.push_back(generator_task->callback);
			}

			Ref<NoiseTexture2D> noise_texture = generator_task->noise_texture;
			Ref<Image> noise_data = generator_task->noise_data;

			noise_texture->set_data(noise_data);

			memdelete(generator_task);
		}
	}

	for (WorkerThreadPool::TaskID finished_task_id : finished_task_ids) {
		generator_tasks_2d.erase(finished_task_id);
	}

	for (const Callable &callback : finished_callbacks) {
		generator_emit_callback(callback);
	}
}

void NoiseTextureGenerator::bake_noise_texture_3d(Ref<NoiseTexture3D> p_noise_texture) {
	ERR_FAIL_COND(p_noise_texture.is_null());
	ERR_FAIL_COND(p_noise_texture->get_noise().is_null());

	if (is_noise_texture_3d_baking(p_noise_texture)) {
		ERR_FAIL_MSG("NoiseTexture3D is already baking. Wait for current bake to finish. Use 'NoiseTextureGenerator::is_noise_texture_3d_baking()' to check the bake state of a noise 3d texture.");
		return;
	}

	Ref<NoiseTexture3D> noise_texture = p_noise_texture;
	const Vector<Ref<Image>> noise_data = noise_texture->bake_noise_data();
	noise_texture->set_data(noise_data);
}

void NoiseTextureGenerator::bake_noise_texture_3d_async(Ref<NoiseTexture3D> p_noise_texture, const Callable &p_callback) {
	ERR_FAIL_COND(p_noise_texture.is_null());
	ERR_FAIL_COND(p_noise_texture->get_noise().is_null());

	if (is_noise_texture_3d_baking(p_noise_texture)) {
		ERR_FAIL_MSG("NoiseTexture3D is already baking. Wait for current bake to finish. Use 'NoiseTextureGenerator::is_noise_texture_3d_baking()' to check the bake state of a noise 3d texture.");
		return;
	}

	if (!use_threads) {
		bake_noise_texture_3d(p_noise_texture);
		if (p_callback.is_valid()) {
			generator_emit_callback(p_callback);
		}
		return;
	}

	bake_mutex.lock();
	GeneratorTask3D *generator_task = memnew(GeneratorTask3D);
	baking_noise_texture_3d.insert(p_noise_texture, generator_task);
	bake_mutex.unlock();

	generator_task->noise_texture = p_noise_texture;
	generator_task->noise = p_noise_texture->get_noise();
	generator_task->callback = p_callback;
	generator_task->status = GeneratorTask3D::TaskStatus::BAKING_STARTED;
	generator_task->thread_task_id = WorkerThreadPool::get_singleton()->add_native_task(&NoiseTextureGenerator::generator_noise_texture_3d_thread_bake, generator_task, false, SNAME("NoiseTextureGenerator"));
	MutexLock generator_task_lock(generator_task_mutex);
	generator_tasks_3d.insert(generator_task->thread_task_id, generator_task);
}

void NoiseTextureGenerator::generator_noise_texture_3d_thread_bake(void *p_arg) {
	GeneratorTask3D *generator_task = static_cast<GeneratorTask3D *>(p_arg);

	Ref<NoiseTexture3D> noise_texture = generator_task->noise_texture;
	Vector<Ref<Image>> noise_data = noise_texture->bake_noise_data();
	generator_task->noise_data = noise_data;

	generator_task->status = GeneratorTask3D::TaskStatus::BAKING_FINISHED;
}

bool NoiseTextureGenerator::is_noise_texture_3d_baking(Ref<NoiseTexture3D> p_noise_texture) {
	MutexLock baking_lock(bake_mutex);
	return baking_noise_texture_3d.has(p_noise_texture);
}

void NoiseTextureGenerator::sync_3d_tasks() {
	if (generator_tasks_3d.is_empty()) {
		return;
	}

	MutexLock generator_task_lock(generator_task_mutex);

	LocalVector<Callable> finished_callbacks;
	LocalVector<WorkerThreadPool::TaskID> finished_task_ids;

	for (KeyValue<WorkerThreadPool::TaskID, GeneratorTask3D *> &E : generator_tasks_3d) {
		if (WorkerThreadPool::get_singleton()->is_task_completed(E.key)) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
			finished_task_ids.push_back(E.key);

			GeneratorTask3D *generator_task = E.value;
			DEV_ASSERT(generator_task->status == GeneratorTask3D::TaskStatus::BAKING_FINISHED);

			baking_noise_texture_3d.erase(generator_task->noise_texture);
			if (generator_task->callback.is_valid()) {
				finished_callbacks.push_back(generator_task->callback);
			}

			Ref<NoiseTexture3D> noise_texture = generator_task->noise_texture;
			Vector<Ref<Image>> noise_data = generator_task->noise_data;

			noise_texture->set_data(noise_data);

			memdelete(generator_task);
		}
	}

	for (WorkerThreadPool::TaskID finished_task_id : finished_task_ids) {
		generator_tasks_3d.erase(finished_task_id);
	}

	for (const Callable &callback : finished_callbacks) {
		generator_emit_callback(callback);
	}
}

void NoiseTextureGenerator::init() {
	if (RS::get_singleton() == nullptr) {
		use_threads = false;
	}

	if (use_threads) {
		RS::get_singleton()->connect(SNAME("frame_pre_draw"), callable_mp(this, &NoiseTextureGenerator::sync));
	}
}

void NoiseTextureGenerator::finish() {
	if (RS::get_singleton() && RS::get_singleton()->is_connected(SNAME("frame_pre_draw"), callable_mp(this, &NoiseTextureGenerator::sync))) {
		RS::get_singleton()->disconnect(SNAME("frame_pre_draw"), callable_mp(this, &NoiseTextureGenerator::sync));
	}

	MutexLock baking_lock(bake_mutex);
	{
		MutexLock generator_task_lock(generator_task_mutex);

		baking_noise_texture_2d.clear();
		baking_noise_texture_3d.clear();

		for (KeyValue<WorkerThreadPool::TaskID, GeneratorTask2D *> &E : generator_tasks_2d) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
			GeneratorTask2D *generator_task = E.value;
			memdelete(generator_task);
		}
		generator_tasks_2d.clear();

		for (KeyValue<WorkerThreadPool::TaskID, GeneratorTask3D *> &E : generator_tasks_3d) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
			GeneratorTask3D *generator_task = E.value;
			memdelete(generator_task);
		}
		generator_tasks_3d.clear();
	}
}

void NoiseTextureGenerator::sync() {
	sync_2d_tasks();
	sync_3d_tasks();
}

bool NoiseTextureGenerator::generator_emit_callback(const Callable &p_callback) {
	ERR_FAIL_COND_V(!p_callback.is_valid(), false);

	Callable::CallError ce;
	Variant result;
	p_callback.callp(nullptr, 0, result, ce);

	return ce.error == Callable::CallError::CALL_OK;
}

void NoiseTextureGenerator::_bind_methods() {
	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("bake_noise_texture_2d", "noise_texture"), &NoiseTextureGenerator::bake_noise_texture_2d);
	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("bake_noise_texture_2d_async", "noise_texture", "callback"), &NoiseTextureGenerator::bake_noise_texture_2d_async, DEFVAL(Callable()));
	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("is_noise_texture_2d_baking", "noise_texture"), &NoiseTextureGenerator::is_noise_texture_2d_baking);

	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("bake_noise_texture_3d", "noise_texture"), &NoiseTextureGenerator::bake_noise_texture_3d);
	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("bake_noise_texture_3d_async", "noise_texture", "callback"), &NoiseTextureGenerator::bake_noise_texture_3d_async, DEFVAL(Callable()));
	ClassDB::bind_static_method("NoiseTextureGenerator", D_METHOD("is_noise_texture_3d_baking", "noise_texture"), &NoiseTextureGenerator::is_noise_texture_3d_baking);
}
