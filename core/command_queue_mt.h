/*************************************************************************/
/*  command_queue_mt.h                                                   */
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

#ifndef COMMAND_QUEUE_MT_H
#define COMMAND_QUEUE_MT_H

#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include "core/simple_type.h"
#include "core/typedefs.h"

#define COMMA(N) _COMMA_##N
#define _COMMA_0
#define _COMMA_1 ,
#define _COMMA_2 ,
#define _COMMA_3 ,
#define _COMMA_4 ,
#define _COMMA_5 ,
#define _COMMA_6 ,
#define _COMMA_7 ,
#define _COMMA_8 ,
#define _COMMA_9 ,
#define _COMMA_10 ,
#define _COMMA_11 ,
#define _COMMA_12 ,
#define _COMMA_13 ,

// 1-based comma separated list of ITEMs
#define COMMA_SEP_LIST(ITEM, LENGTH) _COMMA_SEP_LIST_##LENGTH(ITEM)
#define _COMMA_SEP_LIST_13(ITEM) \
	_COMMA_SEP_LIST_12(ITEM)     \
	, ITEM(13)
#define _COMMA_SEP_LIST_12(ITEM) \
	_COMMA_SEP_LIST_11(ITEM)     \
	, ITEM(12)
#define _COMMA_SEP_LIST_11(ITEM) \
	_COMMA_SEP_LIST_10(ITEM)     \
	, ITEM(11)
#define _COMMA_SEP_LIST_10(ITEM) \
	_COMMA_SEP_LIST_9(ITEM)      \
	, ITEM(10)
#define _COMMA_SEP_LIST_9(ITEM) \
	_COMMA_SEP_LIST_8(ITEM)     \
	, ITEM(9)
#define _COMMA_SEP_LIST_8(ITEM) \
	_COMMA_SEP_LIST_7(ITEM)     \
	, ITEM(8)
#define _COMMA_SEP_LIST_7(ITEM) \
	_COMMA_SEP_LIST_6(ITEM)     \
	, ITEM(7)
#define _COMMA_SEP_LIST_6(ITEM) \
	_COMMA_SEP_LIST_5(ITEM)     \
	, ITEM(6)
#define _COMMA_SEP_LIST_5(ITEM) \
	_COMMA_SEP_LIST_4(ITEM)     \
	, ITEM(5)
#define _COMMA_SEP_LIST_4(ITEM) \
	_COMMA_SEP_LIST_3(ITEM)     \
	, ITEM(4)
#define _COMMA_SEP_LIST_3(ITEM) \
	_COMMA_SEP_LIST_2(ITEM)     \
	, ITEM(3)
#define _COMMA_SEP_LIST_2(ITEM) \
	_COMMA_SEP_LIST_1(ITEM)     \
	, ITEM(2)
#define _COMMA_SEP_LIST_1(ITEM) \
	_COMMA_SEP_LIST_0(ITEM)     \
	ITEM(1)
#define _COMMA_SEP_LIST_0(ITEM)

// 1-based semicolon separated list of ITEMs
#define SEMIC_SEP_LIST(ITEM, LENGTH) _SEMIC_SEP_LIST_##LENGTH(ITEM)
#define _SEMIC_SEP_LIST_13(ITEM) \
	_SEMIC_SEP_LIST_12(ITEM);    \
	ITEM(13)
#define _SEMIC_SEP_LIST_12(ITEM) \
	_SEMIC_SEP_LIST_11(ITEM);    \
	ITEM(12)
#define _SEMIC_SEP_LIST_11(ITEM) \
	_SEMIC_SEP_LIST_10(ITEM);    \
	ITEM(11)
#define _SEMIC_SEP_LIST_10(ITEM) \
	_SEMIC_SEP_LIST_9(ITEM);     \
	ITEM(10)
#define _SEMIC_SEP_LIST_9(ITEM) \
	_SEMIC_SEP_LIST_8(ITEM);    \
	ITEM(9)
#define _SEMIC_SEP_LIST_8(ITEM) \
	_SEMIC_SEP_LIST_7(ITEM);    \
	ITEM(8)
#define _SEMIC_SEP_LIST_7(ITEM) \
	_SEMIC_SEP_LIST_6(ITEM);    \
	ITEM(7)
#define _SEMIC_SEP_LIST_6(ITEM) \
	_SEMIC_SEP_LIST_5(ITEM);    \
	ITEM(6)
#define _SEMIC_SEP_LIST_5(ITEM) \
	_SEMIC_SEP_LIST_4(ITEM);    \
	ITEM(5)
#define _SEMIC_SEP_LIST_4(ITEM) \
	_SEMIC_SEP_LIST_3(ITEM);    \
	ITEM(4)
#define _SEMIC_SEP_LIST_3(ITEM) \
	_SEMIC_SEP_LIST_2(ITEM);    \
	ITEM(3)
#define _SEMIC_SEP_LIST_2(ITEM) \
	_SEMIC_SEP_LIST_1(ITEM);    \
	ITEM(2)
#define _SEMIC_SEP_LIST_1(ITEM) \
	_SEMIC_SEP_LIST_0(ITEM)     \
	ITEM(1)
#define _SEMIC_SEP_LIST_0(ITEM)

// 1-based space separated list of ITEMs
#define SPACE_SEP_LIST(ITEM, LENGTH) _SPACE_SEP_LIST_##LENGTH(ITEM)
#define _SPACE_SEP_LIST_13(ITEM) \
	_SPACE_SEP_LIST_12(ITEM)     \
	ITEM(13)
#define _SPACE_SEP_LIST_12(ITEM) \
	_SPACE_SEP_LIST_11(ITEM)     \
	ITEM(12)
#define _SPACE_SEP_LIST_11(ITEM) \
	_SPACE_SEP_LIST_10(ITEM)     \
	ITEM(11)
#define _SPACE_SEP_LIST_10(ITEM) \
	_SPACE_SEP_LIST_9(ITEM)      \
	ITEM(10)
#define _SPACE_SEP_LIST_9(ITEM) \
	_SPACE_SEP_LIST_8(ITEM)     \
	ITEM(9)
#define _SPACE_SEP_LIST_8(ITEM) \
	_SPACE_SEP_LIST_7(ITEM)     \
	ITEM(8)
#define _SPACE_SEP_LIST_7(ITEM) \
	_SPACE_SEP_LIST_6(ITEM)     \
	ITEM(7)
#define _SPACE_SEP_LIST_6(ITEM) \
	_SPACE_SEP_LIST_5(ITEM)     \
	ITEM(6)
#define _SPACE_SEP_LIST_5(ITEM) \
	_SPACE_SEP_LIST_4(ITEM)     \
	ITEM(5)
#define _SPACE_SEP_LIST_4(ITEM) \
	_SPACE_SEP_LIST_3(ITEM)     \
	ITEM(4)
#define _SPACE_SEP_LIST_3(ITEM) \
	_SPACE_SEP_LIST_2(ITEM)     \
	ITEM(3)
#define _SPACE_SEP_LIST_2(ITEM) \
	_SPACE_SEP_LIST_1(ITEM)     \
	ITEM(2)
#define _SPACE_SEP_LIST_1(ITEM) \
	_SPACE_SEP_LIST_0(ITEM)     \
	ITEM(1)
#define _SPACE_SEP_LIST_0(ITEM)

#define ARG(N) p##N
#define PARAM(N) P##N p##N
#define TYPE_PARAM(N) class P##N
#define PARAM_DECL(N) typename GetSimpleTypeT<P##N>::type_t p##N

#define DECL_CMD(N)                                                    \
	template <class T, class M COMMA(N) COMMA_SEP_LIST(TYPE_PARAM, N)> \
	struct Command##N : public CommandBase {                           \
		T *instance;                                                   \
		M method;                                                      \
		SEMIC_SEP_LIST(PARAM_DECL, N);                                 \
		virtual void call() {                                          \
			(instance->*method)(COMMA_SEP_LIST(ARG, N));               \
		}                                                              \
	};

#define DECL_CMD_RET(N)                                                         \
	template <class T, class M, COMMA_SEP_LIST(TYPE_PARAM, N) COMMA(N) class R> \
	struct CommandRet##N : public SyncCommand {                                 \
		R *ret;                                                                 \
		T *instance;                                                            \
		M method;                                                               \
		SEMIC_SEP_LIST(PARAM_DECL, N);                                          \
		virtual void call() {                                                   \
			*ret = (instance->*method)(COMMA_SEP_LIST(ARG, N));                 \
		}                                                                       \
	};

#define DECL_CMD_SYNC(N)                                               \
	template <class T, class M COMMA(N) COMMA_SEP_LIST(TYPE_PARAM, N)> \
	struct CommandSync##N : public SyncCommand {                       \
		T *instance;                                                   \
		M method;                                                      \
		SEMIC_SEP_LIST(PARAM_DECL, N);                                 \
		virtual void call() {                                          \
			(instance->*method)(COMMA_SEP_LIST(ARG, N));               \
		}                                                              \
	};

#define TYPE_ARG(N) P##N
#define CMD_TYPE(N) Command##N<T, M COMMA(N) COMMA_SEP_LIST(TYPE_ARG, N)>
#define CMD_ASSIGN_PARAM(N) cmd->p##N = p##N

#define DECL_PUSH(N)                                                         \
	template <class T, class M COMMA(N) COMMA_SEP_LIST(TYPE_PARAM, N)>       \
	void push(T *p_instance, M p_method COMMA(N) COMMA_SEP_LIST(PARAM, N)) { \
		CMD_TYPE(N) *cmd = allocate_and_lock<CMD_TYPE(N)>();                 \
		cmd->instance = p_instance;                                          \
		cmd->method = p_method;                                              \
		SEMIC_SEP_LIST(CMD_ASSIGN_PARAM, N);                                 \
		unlock();                                                            \
		if (sync)                                                            \
			sync->post();                                                    \
	}

#define CMD_RET_TYPE(N) CommandRet##N<T, M, COMMA_SEP_LIST(TYPE_ARG, N) COMMA(N) R>

#define DECL_PUSH_AND_RET(N)                                                                   \
	template <class T, class M, COMMA_SEP_LIST(TYPE_PARAM, N) COMMA(N) class R>                \
	void push_and_ret(T *p_instance, M p_method, COMMA_SEP_LIST(PARAM, N) COMMA(N) R *r_ret) { \
		SyncSemaphore *ss = _alloc_sync_sem();                                                 \
		CMD_RET_TYPE(N) *cmd = allocate_and_lock<CMD_RET_TYPE(N)>();                           \
		cmd->instance = p_instance;                                                            \
		cmd->method = p_method;                                                                \
		SEMIC_SEP_LIST(CMD_ASSIGN_PARAM, N);                                                   \
		cmd->ret = r_ret;                                                                      \
		cmd->sync_sem = ss;                                                                    \
		unlock();                                                                              \
		if (sync)                                                                              \
			sync->post();                                                                      \
		ss->sem.wait();                                                                        \
		ss->in_use = false;                                                                    \
	}

#define CMD_SYNC_TYPE(N) CommandSync##N<T, M COMMA(N) COMMA_SEP_LIST(TYPE_ARG, N)>

#define DECL_PUSH_AND_SYNC(N)                                                         \
	template <class T, class M COMMA(N) COMMA_SEP_LIST(TYPE_PARAM, N)>                \
	void push_and_sync(T *p_instance, M p_method COMMA(N) COMMA_SEP_LIST(PARAM, N)) { \
		SyncSemaphore *ss = _alloc_sync_sem();                                        \
		CMD_SYNC_TYPE(N) *cmd = allocate_and_lock<CMD_SYNC_TYPE(N)>();                \
		cmd->instance = p_instance;                                                   \
		cmd->method = p_method;                                                       \
		SEMIC_SEP_LIST(CMD_ASSIGN_PARAM, N);                                          \
		cmd->sync_sem = ss;                                                           \
		unlock();                                                                     \
		if (sync)                                                                     \
			sync->post();                                                             \
		ss->sem.wait();                                                               \
		ss->in_use = false;                                                           \
	}

#define MAX_CMD_PARAMS 13

class CommandQueueMT {
	struct SyncSemaphore {
		Semaphore sem;
		bool in_use;
	};

	struct CommandBase {
		virtual void call() = 0;
		virtual void post(){};
		virtual ~CommandBase(){};
	};

	struct SyncCommand : public CommandBase {
		SyncSemaphore *sync_sem;

		virtual void post() {
			sync_sem->sem.post();
		}
	};

	DECL_CMD(0)
	SPACE_SEP_LIST(DECL_CMD, 13)

	/* comands that return */
	DECL_CMD_RET(0)
	SPACE_SEP_LIST(DECL_CMD_RET, 13)

	/* commands that don't return but sync */
	DECL_CMD_SYNC(0)
	SPACE_SEP_LIST(DECL_CMD_SYNC, 13)

	/***** BASE *******/

	enum {
		DEFAULT_COMMAND_MEM_SIZE_KB = 256,
		SYNC_SEMAPHORES = 8
	};

	uint8_t *command_mem;
	uint32_t read_ptr_and_epoch;
	uint32_t write_ptr_and_epoch;
	uint32_t dealloc_ptr;
	uint32_t command_mem_size;
	SyncSemaphore sync_sems[SYNC_SEMAPHORES];
	Mutex mutex;
	Semaphore *sync;

	template <class T>
	T *allocate() {
		// alloc size is size+T+safeguard
		uint32_t alloc_size = ((sizeof(T) + 8 - 1) & ~(8 - 1)) + 8;

		// Assert that the buffer is big enough to hold at least two messages.
		ERR_FAIL_COND_V(alloc_size * 2 + sizeof(uint32_t) > command_mem_size, nullptr);

	tryagain:
		uint32_t write_ptr = write_ptr_and_epoch >> 1;

		if (write_ptr < dealloc_ptr) {
			// behind dealloc_ptr, check that there is room
			if ((dealloc_ptr - write_ptr) <= alloc_size) {
				// There is no more room, try to deallocate something
				if (dealloc_one()) {
					goto tryagain;
				}
				return nullptr;
			}
		} else {
			// ahead of dealloc_ptr, check that there is room

			if ((command_mem_size - write_ptr) < alloc_size + sizeof(uint32_t)) {
				// no room at the end, wrap down;

				if (dealloc_ptr == 0) { // don't want write_ptr to become dealloc_ptr

					// There is no more room, try to deallocate something
					if (dealloc_one()) {
						goto tryagain;
					}
					return nullptr;
				}

				// if this happens, it's a bug
				ERR_FAIL_COND_V((command_mem_size - write_ptr) < 8, nullptr);
				// zero means, wrap to beginning

				uint32_t *p = (uint32_t *)&command_mem[write_ptr];
				*p = 1;
				write_ptr_and_epoch = 0 | (1 & ~write_ptr_and_epoch); // Invert epoch.
				// See if we can get the thread to run and clear up some more space while we wait.
				// This is required if alloc_size * 2 + 4 > COMMAND_MEM_SIZE
				if (sync) {
					sync->post();
				}
				goto tryagain;
			}
		}
		// Allocate the size and the 'in use' bit.
		// First bit used to mark if command is still in use (1)
		// or if it has been destroyed and can be deallocated (0).
		uint32_t size = (sizeof(T) + 8 - 1) & ~(8 - 1);
		uint32_t *p = (uint32_t *)&command_mem[write_ptr];
		*p = (size << 1) | 1;
		write_ptr += 8;
		// allocate the command
		T *cmd = memnew_placement(&command_mem[write_ptr], T);
		write_ptr += size;
		write_ptr_and_epoch = (write_ptr << 1) | (write_ptr_and_epoch & 1);
		return cmd;
	}

	template <class T>
	T *allocate_and_lock() {
		lock();
		T *ret;

		while ((ret = allocate<T>()) == nullptr) {
			unlock();
			// sleep a little until fetch happened and some room is made
			wait_for_flush();
			lock();
		}

		return ret;
	}

	bool flush_one(bool p_lock = true) {
		if (p_lock) {
			lock();
		}
	tryagain:

		// tried to read an empty queue
		if (read_ptr_and_epoch == write_ptr_and_epoch) {
			if (p_lock) {
				unlock();
			}
			return false;
		}

		uint32_t read_ptr = read_ptr_and_epoch >> 1;
		uint32_t size_ptr = read_ptr;
		uint32_t size = *(uint32_t *)&command_mem[read_ptr] >> 1;

		if (size == 0) {
			*(uint32_t *)&command_mem[read_ptr] = 0; // clear in-use bit.
			//end of ringbuffer, wrap
			read_ptr_and_epoch = 0 | (1 & ~read_ptr_and_epoch); // Invert epoch.
			goto tryagain;
		}

		read_ptr += 8;

		CommandBase *cmd = reinterpret_cast<CommandBase *>(&command_mem[read_ptr]);

		read_ptr += size;

		read_ptr_and_epoch = (read_ptr << 1) | (read_ptr_and_epoch & 1);

		if (p_lock) {
			unlock();
		}
		cmd->call();
		if (p_lock) {
			lock();
		}

		cmd->post();
		cmd->~CommandBase();
		*(uint32_t *)&command_mem[size_ptr] &= ~1;

		if (p_lock) {
			unlock();
		}
		return true;
	}

	void lock();
	void unlock();
	void wait_for_flush();
	SyncSemaphore *_alloc_sync_sem();
	bool dealloc_one();

public:
	/* NORMAL PUSH COMMANDS */
	DECL_PUSH(0)
	SPACE_SEP_LIST(DECL_PUSH, 13)

	/* PUSH AND RET COMMANDS */
	DECL_PUSH_AND_RET(0)
	SPACE_SEP_LIST(DECL_PUSH_AND_RET, 13)

	/* PUSH AND RET SYNC COMMANDS*/
	DECL_PUSH_AND_SYNC(0)
	SPACE_SEP_LIST(DECL_PUSH_AND_SYNC, 13)

	void wait_and_flush_one() {
		ERR_FAIL_COND(!sync);
		sync->wait();
		flush_one();
	}

	void flush_all() {
		//ERR_FAIL_COND(sync);
		lock();
		while (flush_one(false)) {
			;
		}
		unlock();
	}

	CommandQueueMT(bool p_sync);
	~CommandQueueMT();
};

#undef ARG
#undef PARAM
#undef TYPE_PARAM
#undef PARAM_DECL
#undef DECL_CMD
#undef DECL_CMD_RET
#undef DECL_CMD_SYNC
#undef TYPE_ARG
#undef CMD_TYPE
#undef CMD_ASSIGN_PARAM
#undef DECL_PUSH
#undef CMD_RET_TYPE
#undef DECL_PUSH_AND_RET
#undef CMD_SYNC_TYPE
#undef DECL_CMD_SYNC

#endif
