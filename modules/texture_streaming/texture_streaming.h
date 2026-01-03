/**************************************************************************/
/*  texture_streaming.h                                                   */
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

#include "core/io/image.h"
#include "core/os/condition_variable.h"
#include "core/os/thread.h"
#include "core/templates/rid_owner.h"
#include "core/templates/tuple.h"

// TextureStreaming singleton
class TextureStreaming : public Object {
	GDCLASS(TextureStreaming, Object);

	static TextureStreaming *singleton;

protected:
	static void _bind_methods();

private:
	// A minimal command queue with integrated condition variable for thread synchronization.
	// Unlike CommandQueueMT, this gives full control over waiting and signaling.
	class SimpleCommandQueue {
		static const size_t MAX_COMMAND_SIZE = 1024;

		struct CommandBase {
			virtual void call() = 0;
			virtual ~CommandBase() = default;
		};

		template <typename T, typename M, typename... Args>
		struct Command : public CommandBase {
			T *instance;
			M method;
			Tuple<GetSimpleTypeT<Args>...> args;

			template <typename... FwdArgs>
			_FORCE_INLINE_ Command([[maybe_unused]] const char *p_method_name, T *p_instance, M p_method, FwdArgs &&...p_args) :
					instance(p_instance), method(p_method), args(std::forward<FwdArgs>(p_args)...) {
			}

			void call() override {
				call_impl(BuildIndexSequence<sizeof...(Args)>{});
			}

		private:
			template <size_t... I>
			_FORCE_INLINE_ void call_impl(IndexSequence<I...>) {
				(instance->*method)(std::move(get<I>())...);
			}

			template <size_t I>
			_FORCE_INLINE_ auto &get() { return ::tuple_get<I>(args); }
		};

		static const uint32_t DEFAULT_COMMAND_MEM_SIZE_KB = 64;

		mutable BinaryMutex mutex;
		mutable ConditionVariable condvar;
		LocalVector<uint8_t> command_mem;
		std::atomic<bool> pending{ false };
		std::atomic<bool> should_exit{ false };

		template <typename T, typename... Args>
		_FORCE_INLINE_ void create_command([[maybe_unused]] const char *p_method_name, Args &&...p_args) {
			constexpr uint64_t alloc_size = ((sizeof(T) + 8U - 1U) & ~(8U - 1U));
			static_assert(alloc_size < UINT32_MAX, "Type too large to fit in the command queue.");

			uint64_t size = command_mem.size();
			command_mem.resize(size + alloc_size + sizeof(uint64_t));
			*(uint64_t *)&command_mem[size] = alloc_size;
			void *cmd = &command_mem[size + sizeof(uint64_t)];
			new (cmd) T(p_method_name, std::forward<Args>(p_args)...);
		}

		void _flush_locked(MutexLock<BinaryMutex> &p_lock) {
			if (!pending.load()) {
				return;
			}

			// Buffer to copy command into before unlocking.
			// This protects against reallocs invalidating the pointer during execution.
			char cmd_buffer[MAX_COMMAND_SIZE];

			uint64_t read_ptr = 0;
			while (read_ptr < command_mem.size()) {
				uint64_t size = *(uint64_t *)&command_mem[read_ptr];
				read_ptr += sizeof(uint64_t);

				CommandBase *cmd = reinterpret_cast<CommandBase *>(&command_mem[read_ptr]);

				// Copy command to local buffer so we can safely unlock during execution.
				// The copy is shallow - complex types like Callable share internal data with the original.
				memcpy(cmd_buffer, (char *)cmd, size);

				read_ptr += size;

				// Unlock while executing the command (allows new commands to be pushed)
				p_lock.temp_unlock();
				reinterpret_cast<CommandBase *>(cmd_buffer)->call();
				p_lock.temp_relock();

				// Destroy the ORIGINAL after execution completes.
				// We must not destroy before call() because the backup is a shallow copy
				// that shares internal data (e.g., Callable's bound arguments).
				// Handle potential realloc that may have happened during unlock.
				cmd = reinterpret_cast<CommandBase *>(&command_mem[read_ptr - size]);
				cmd->~CommandBase();
			}

			command_mem.clear();
			pending.store(false);
		}

	public:
		// Push a command to the queue and signal the waiting thread.
		// Use the macro PUSH_CMD for automatic method name capture.
		template <typename T, typename M, typename... Args>
		void push_internal(const char *p_method_name, T *p_instance, M p_method, Args &&...p_args) {
			using CommandType = Command<T, M, Args...>;
			static_assert(sizeof(CommandType) <= MAX_COMMAND_SIZE);

			MutexLock lock(mutex);
			create_command<CommandType>(p_method_name, p_instance, p_method, std::forward<Args>(p_args)...);
			pending.store(true);
			condvar.notify_one();
		}

		// Signal the thread to exit and wake it up.
		void request_exit() {
			MutexLock lock(mutex);
			should_exit.store(true);
			condvar.notify_one();
		}

		// Flush all pending commands. Call from the processing thread.
		void flush() {
			MutexLock lock(mutex);
			_flush_locked(lock);
		}

		// Wait for commands or exit signal, then flush. Returns true if should continue, false if should exit.
		// This is the main loop function for the processing thread.
		bool wait_and_flush() {
			MutexLock lock(mutex);

			// Wait until we have work or should exit
			while (!pending.load() && !should_exit.load()) {
				condvar.wait(lock);
			}

			if (should_exit.load()) {
				// Flush any remaining commands before exiting
				_flush_locked(lock);
				return false;
			}

			_flush_locked(lock);
			return true;
		}

		SimpleCommandQueue() {
			command_mem.reserve(DEFAULT_COMMAND_MEM_SIZE_KB * 1024);
		}

		~SimpleCommandQueue() {
			// Flush any remaining commands
			flush();
		}
	};

	// Helper class to manage RID_Owner with index-based access.
	// Thread-safe version: protects index_to_rid with a mutex.
	template <typename T>
	class RID_IndexedOwner {
		static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;

		struct Wrapper {
			T data;
			uint32_t index;
		};

		mutable RID_Owner<Wrapper, true> owner;
		mutable BinaryMutex index_mutex;
		LocalVector<RID> index_to_rid;

	public:
		_FORCE_INLINE_ T *get_or_null(const RID &p_rid) {
			// No lock needed - RID_Owner<T, true> is already thread-safe for lookups
			Wrapper *wrapper = owner.get_or_null(p_rid);
			return wrapper ? &wrapper->data : nullptr;
		}

		_FORCE_INLINE_ T *allocate(RID &p_rid) {
			MutexLock<BinaryMutex> lock(index_mutex);
			uint32_t new_index = index_to_rid.size();
			Wrapper wrapper;
			wrapper.index = new_index;
			RID rid = owner.allocate_rid();
			owner.initialize_rid(rid, wrapper);
			index_to_rid.push_back(rid);
			Wrapper *wrapper_ptr = owner.get_or_null(rid);
			p_rid = rid;
			return wrapper_ptr ? &wrapper_ptr->data : nullptr;
		}

		_FORCE_INLINE_ uint32_t get_index(const RID &p_rid) const {
			// No lock needed - we only read from owner which is thread-safe
			Wrapper *wrapper = owner.get_or_null(p_rid);
			return wrapper ? wrapper->index : INVALID_INDEX;
		}

		_FORCE_INLINE_ void free(const RID &p_rid) {
			MutexLock<BinaryMutex> lock(index_mutex);
			Wrapper *wrapper = owner.get_or_null(p_rid);
			if (wrapper) {
				uint32_t index = wrapper->index;
				uint32_t last_index = index_to_rid.size() - 1;

				if (index != last_index) {
					// Move last element to the freed slot
					RID last_rid = index_to_rid[last_index];
					index_to_rid[index] = last_rid;

					// Update the moved element's index
					Wrapper *last_wrapper = owner.get_or_null(last_rid);
					if (last_wrapper) {
						last_wrapper->index = index;
					}
				}

				index_to_rid.resize(last_index);
				owner.free(p_rid);
			}
		}

		_FORCE_INLINE_ uint32_t get_count() const {
			MutexLock<BinaryMutex> lock(index_mutex);
			return index_to_rid.size();
		}
	};

	// Per texture streaming state
	struct StreamingState {
		// Configuration
		RID texture;
		uint16_t width = 0;
		uint16_t height = 0;
		Image::Format format = Image::FORMAT_MAX;

		// Settings & Constraints
		Callable reload_callable;
		uint16_t min_resolution = 0;
		uint16_t max_resolution = 0;

		// Runtime state
		uint16_t feedback_resolution = 0; // The resolution requested by the material feedback.
		uint16_t request_resolution = 0; // The resolution after clamping to min/max.
		uint16_t fit_resolution = 0; // The resolution that fits in the budget or minimum.
		uint16_t current_resolution = 0; // The current resolution that has been set.
		std::atomic<uint16_t> last_resolution = 0; // The last resolution that was set.
		std::atomic<uint16_t> pending_reload_resolution = 0; // Resolution currently queued for reload (0 = none pending)
		uint64_t changed_tick_msec = 0;
		uint64_t requested_tick_msec = 0;

		StreamingState() {}

		StreamingState(const StreamingState &p_other) {
			texture = p_other.texture;
			width = p_other.width;
			height = p_other.height;
			format = p_other.format;
			reload_callable = p_other.reload_callable;
			min_resolution = p_other.min_resolution;
			max_resolution = p_other.max_resolution;
			feedback_resolution = p_other.feedback_resolution;
			request_resolution = p_other.request_resolution;
			fit_resolution = p_other.fit_resolution;
			current_resolution = p_other.current_resolution;
			last_resolution = p_other.last_resolution.load();
			pending_reload_resolution = p_other.pending_reload_resolution.load();
			changed_tick_msec = p_other.changed_tick_msec;
			requested_tick_msec = p_other.requested_tick_msec;
		}
	};

	RID_Owner<StreamingState, true> streaming_info_owner;

	// A collection of bits needed for material feedback from rendering.
	// This includes a buffer which stores feedback data, and a mapping from buffer indices to texture RIDs,
	struct MaterialFeedbackBuffer {
		RID buffer; // RID for the material feedback buffer.
		uint32_t buffer_size = 0; // Size of the buffer in bytes.
		LocalVector<RID> rid_map; // Maps indices in the buffer to texture RIDs.
		RID self;

		PackedByteArray data;

		void clear();
		void resize();

		MaterialFeedbackBuffer();
		~MaterialFeedbackBuffer();
	};
	RID_Owner<MaterialFeedbackBuffer, true> feedback_buffer_owner;

	BinaryMutex material_mutex;

	// Buffer pool management
	BinaryMutex buffer_pool_mutex;
	Vector<RID> buffer_pool;

	struct MaterialInfo {
		Vector<RID> textures;

		float smoothed_max = 0.0f;
		uint64_t last_update_tick_msec = 0;

		_FORCE_INLINE_ uint32_t update(uint32_t p_value, uint64_t p_current_tick_msec, float p_decay_per_msec = 0.00001f) {
			float value = Math::log2(float(p_value));
			// Linear decay based on time
			if (last_update_tick_msec > 0) {
				uint64_t delta_msec = p_current_tick_msec - last_update_tick_msec;
				smoothed_max = MAX(0.0f, smoothed_max - (float(delta_msec) * p_decay_per_msec));
			}
			smoothed_max = MAX(float(value), smoothed_max);
			last_update_tick_msec = p_current_tick_msec;

			return 1u << uint32_t(Math::round(smoothed_max));
		}
	};
	RID_IndexedOwner<MaterialInfo> material_info_owner;

	// Settings
	bool setting_streaming_is_enabled = true;
	bool setting_budget_enabled = true;
	uint32_t setting_budget_mb = 512;
	uint32_t setting_texture_max_resolution = 8192;
	uint32_t setting_texture_min_resolution = 32u;
	uint64_t setting_texture_change_wait_msec = 100;

	// Feedback buffer processing
	// SelfList<MaterialFeedbackBuffer>::List feedback_buffer_queue;
	Thread feedback_buffer_thread;
	uint64_t feedback_buffer_last_submit_ticks = 0;

	// Budget fitting candidate structure for budget fitting.
	// Tracks each texture's constraints, target resolution, and priority metrics.
	struct FitCandidate {
		StreamingState *state = nullptr; // Pointer to the texture's streaming state (nullptr = exhausted/skip)
		uint16_t min_res = 0; // Minimum allowed resolution for this texture
		uint16_t max_res = 0; // Maximum allowed resolution for this texture
		uint16_t target_res = 0; // Current target resolution (adjusted during budget fitting)
		uint64_t inactivity_msec = 0; // Time since last shader request for this texture
		uint64_t bytes = 0; // Memory footprint at target_res (updated as target changes)
	};
	LocalVector<FitCandidate> fit_candidates; // Reused across _feedback_buffer_process calls to reduce allocations

	// Heap-based priority comparison for budget fitting.
	// Compares two candidates to determine which should be reduced first.
	// Returns true if 'a' should be reduced AFTER 'b' (for max-heap behavior with std::make_heap).
	struct FitCandidateComparator {
		_FORCE_INLINE_ bool operator()(const FitCandidate *p_a, const FitCandidate *p_b) const {
			// Skip exhausted candidates (treat as lowest priority)
			if (!p_a->state || p_a->target_res <= p_a->min_res) {
				return false; // 'a' is exhausted, so 'b' has priority
			}
			if (!p_b->state || p_b->target_res <= p_b->min_res) {
				return true; // 'b' is exhausted, so 'a' has priority
			}

			// Determine if candidates are above their requested resolution
			uint16_t a_requested = CLAMP(p_a->state->request_resolution, p_a->min_res, p_a->max_res);
			uint16_t b_requested = CLAMP(p_b->state->request_resolution, p_b->min_res, p_b->max_res);
			bool a_over = p_a->target_res > a_requested;
			bool b_over = p_b->target_res > b_requested;

			// Priority 1: Prefer reducing textures above their requested resolution
			if (a_over != b_over) {
				return b_over; // Inverted: if b is over, a should come after b in heap
			}

			// Priority 2: Among same category, prefer longer inactivity
			if (p_a->inactivity_msec != p_b->inactivity_msec) {
				return p_a->inactivity_msec < p_b->inactivity_msec; // Inverted
			}

			// Priority 3: Among equally inactive, prefer larger memory footprint
			if (p_a->bytes != p_b->bytes) {
				return p_a->bytes < p_b->bytes; // Inverted
			}

			// Priority 4: Break ties by targeting higher resolution first
			return p_a->target_res < p_b->target_res; // Inverted
		}
	};

	LocalVector<FitCandidate *> reduction_heap; // Heap of candidate pointers for efficient budget fitting

	static void _feedback_buffer_thread_func(void *p_udata);
	void _feedback_buffer_thread_main();
	void _feedback_buffer_process(uint64_t p_ticks_msec);

	// I/O thread for texture reloading - uses its own command queue
	SimpleCommandQueue io_command_queue;
	Thread texture_reload_thread;

	static void _texture_reload_thread_func(void *p_udata);
	void _texture_reload_thread_main();
	void _do_texture_reload(StreamingState *p_state);

	std::atomic<uint64_t> texture_streaming_total_memory = 0;

	void feedback_handle_data(const PackedByteArray &p_array, RID p_buffer);

	RID current_feedback_buffer;
	void feedback_frame_done_callback();
	void feedback_frame_done_callback_render_thread();
	RID feedback_buffer_get_next();

	bool initialized = false;

	// Completion helper for synchronous command queue calls.
	// Used to wait for a result from a command executed on the feedback thread.
	template <typename T>
	struct Completion {
		BinaryMutex mutex;
		ConditionVariable condvar;
		bool done = false;
		T result;

		void wait() {
			MutexLock<BinaryMutex> lock(mutex);
			while (!done) {
				condvar.wait(lock);
			}
		}

		void complete(const T &p_result) {
			MutexLock<BinaryMutex> lock(mutex);
			result = p_result;
			done = true;
			condvar.notify_all();
		}
	};

	// Separate struct for void completion (commands that don't return a value).
	struct CompletionVoid {
		BinaryMutex mutex;
		ConditionVariable condvar;
		bool done = false;

		void wait() {
			MutexLock<BinaryMutex> lock(mutex);
			while (!done) {
				condvar.wait(lock);
			}
		}

		void complete() {
			MutexLock<BinaryMutex> lock(mutex);
			done = true;
			condvar.notify_all();
		}
	};

	mutable SimpleCommandQueue command_queue;

	// Helper to push a command and wait for completion with a return value.
	// The method M should have signature: void (T::*M)(Completion<R>*, Args...)
	template <typename R, typename T, typename M, typename... Args>
	_FORCE_INLINE_ R push_and_wait(const char *p_method_name, T *p_instance, M p_method, Args &&...p_args) {
		Completion<R> completion;
		command_queue.push_internal(p_method_name, p_instance, p_method, &completion, std::forward<Args>(p_args)...);
		completion.wait();
		return completion.result;
	}

	// Helper to push a command and wait for completion without a return value.
	// The method M should have signature: void (T::*M)(CompletionVoid*, Args...)
	template <typename T, typename M, typename... Args>
	_FORCE_INLINE_ void push_and_sync(const char *p_method_name, T *p_instance, M p_method, Args &&...p_args) {
		CompletionVoid completion;
		command_queue.push_internal(p_method_name, p_instance, p_method, &completion, std::forward<Args>(p_args)...);
		completion.wait();
	}

	// Internal implementations called on feedback thread
	void _texture_configure_streaming_impl(Completion<RID> *p_completion, RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_resolution, int p_max_resolution, const Callable &p_reload_callable);
	void _texture_update_impl(CompletionVoid *p_completion, RID p_rid, int p_width, int p_height, int p_min_resolution, int p_max_resolution);
	void _texture_remove_impl(CompletionVoid *p_completion, RID p_rid);

	void _process_material_feedback_buffer(MaterialFeedbackBuffer *p_mb, uint64_t p_ticks_msec);

	void render_thread_specific_initialization();

public:
	static TextureStreaming *get_singleton();

	// Feedback Buffer API
	uint32_t feedback_buffer_material_index(RID p_material);
	RID feedback_buffer_get_uniform_rid();

	// Texture API - safe from any thread
	RID texture_configure_streaming(RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_resolution = 0, int p_max_resolution = 0, Callable p_reload_callable = Callable());
	void texture_update(RID p_rid, int p_width, int p_height, int p_min_resolution, int p_max_resolution);
	void texture_remove(RID p_rid);

	// Material API
	RID material_set_textures(RID p_feedback_rid, const Vector<RID> &p_textures);

	// Status API
	uint64_t get_memory_budget_bytes_used();

	// Properties
	void set_streaming_min_resolution(uint32_t p_resolution) {
		setting_texture_min_resolution = p_resolution;
	}

	uint32_t get_streaming_min_resolution() const {
		return setting_texture_min_resolution;
	}

	void set_streaming_max_resolution(uint32_t p_resolution) {
		setting_texture_max_resolution = p_resolution;
	}

	uint32_t get_streaming_max_resolution() const {
		return setting_texture_max_resolution;
	}

	void set_budget_enabled(bool p_enabled) {
		setting_budget_enabled = p_enabled;
	}

	bool get_budget_enabled() const {
		return setting_budget_enabled;
	}

	void set_memory_budget_mb(float p_mb) {
		setting_budget_mb = uint32_t(p_mb);
	}

	float get_memory_budget_mb() const {
		return float(setting_budget_mb);
	}

	TextureStreaming();
	virtual ~TextureStreaming();
};
