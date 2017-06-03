/*************************************************************************/
/*  command_queue_mt.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os/memory.h"
#include "os/mutex.h"
#include "os/semaphore.h"
#include "simple_type.h"
#include "typedefs.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class CommandQueueMT {

	struct SyncSemaphore {

		Semaphore *sem;
		bool in_use;
	};

	struct CommandBase {

		virtual void call() = 0;
		virtual ~CommandBase(){};
	};

	template <class T, class M>
	struct Command0 : public CommandBase {

		T *instance;
		M method;

		virtual void call() { (instance->*method)(); }
	};

	template <class T, class M, class P1>
	struct Command1 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;

		virtual void call() { (instance->*method)(p1); }
	};

	template <class T, class M, class P1, class P2>
	struct Command2 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;

		virtual void call() { (instance->*method)(p1, p2); }
	};

	template <class T, class M, class P1, class P2, class P3>
	struct Command3 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;

		virtual void call() { (instance->*method)(p1, p2, p3); }
	};

	template <class T, class M, class P1, class P2, class P3, class P4>
	struct Command4 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;

		virtual void call() { (instance->*method)(p1, p2, p3, p4); }
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5>
	struct Command5 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;

		virtual void call() { (instance->*method)(p1, p2, p3, p4, p5); }
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6>
	struct Command6 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;

		virtual void call() { (instance->*method)(p1, p2, p3, p4, p5, p6); }
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
	struct Command7 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;

		virtual void call() { (instance->*method)(p1, p2, p3, p4, p5, p6, p7); }
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
	struct Command8 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;
		typename GetSimpleTypeT<P8>::type_t p8;

		virtual void call() { (instance->*method)(p1, p2, p3, p4, p5, p6, p7, p8); }
	};

	/* comands that return */

	template <class T, class M, class R>
	struct CommandRet0 : public CommandBase {

		T *instance;
		M method;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)();
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class R>
	struct CommandRet1 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class R>
	struct CommandRet2 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class R>
	struct CommandRet3 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class R>
	struct CommandRet4 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3, p4);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class R>
	struct CommandRet5 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3, p4, p5);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class R>
	struct CommandRet6 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3, p4, p5, p6);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class R>
	struct CommandRet7 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3, p4, p5, p6, p7);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8, class R>
	struct CommandRet8 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;
		typename GetSimpleTypeT<P8>::type_t p8;
		R *ret;
		SyncSemaphore *sync;

		virtual void call() {
			*ret = (instance->*method)(p1, p2, p3, p4, p5, p6, p7, p8);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	/** commands that don't return but sync */

	/* comands that return */

	template <class T, class M>
	struct CommandSync0 : public CommandBase {

		T *instance;
		M method;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)();
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1>
	struct CommandSync1 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2>
	struct CommandSync2 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3>
	struct CommandSync3 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4>
	struct CommandSync4 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3, p4);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5>
	struct CommandSync5 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3, p4, p5);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6>
	struct CommandSync6 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3, p4, p5, p6);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
	struct CommandSync7 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3, p4, p5, p6, p7);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
	struct CommandSync8 : public CommandBase {

		T *instance;
		M method;
		typename GetSimpleTypeT<P1>::type_t p1;
		typename GetSimpleTypeT<P2>::type_t p2;
		typename GetSimpleTypeT<P3>::type_t p3;
		typename GetSimpleTypeT<P4>::type_t p4;
		typename GetSimpleTypeT<P5>::type_t p5;
		typename GetSimpleTypeT<P6>::type_t p6;
		typename GetSimpleTypeT<P7>::type_t p7;
		typename GetSimpleTypeT<P8>::type_t p8;

		SyncSemaphore *sync;

		virtual void call() {
			(instance->*method)(p1, p2, p3, p4, p5, p6, p7, p8);
			sync->sem->post();
			sync->in_use = false;
		}
	};

	/***** BASE *******/

	enum {
		COMMAND_MEM_SIZE_KB = 256,
		COMMAND_MEM_SIZE = COMMAND_MEM_SIZE_KB * 1024,
		SYNC_SEMAPHORES = 8
	};

	uint8_t command_mem[COMMAND_MEM_SIZE];
	uint32_t read_ptr;
	uint32_t write_ptr;
	SyncSemaphore sync_sems[SYNC_SEMAPHORES];
	Mutex *mutex;
	Semaphore *sync;

	template <class T>
	T *allocate() {

		// alloc size is size+T+safeguard
		uint32_t alloc_size = sizeof(T) + sizeof(uint32_t);

	tryagain:

		if (write_ptr < read_ptr) {
			// behind read_ptr, check that there is room
			if ((read_ptr - write_ptr) <= alloc_size)
				return NULL;
		} else if (write_ptr >= read_ptr) {
			// ahead of read_ptr, check that there is room

			if ((COMMAND_MEM_SIZE - write_ptr) < alloc_size + 4) {
				// no room at the end, wrap down;

				if (read_ptr == 0) // don't want write_ptr to become read_ptr
					return NULL;

				// if this happens, it's a bug
				ERR_FAIL_COND_V((COMMAND_MEM_SIZE - write_ptr) < sizeof(uint32_t), NULL);
				// zero means, wrap to beginning

				uint32_t *p = (uint32_t *)&command_mem[write_ptr];
				*p = 0;
				write_ptr = 0;
				goto tryagain;
			}
		}
		// allocate the size
		uint32_t *p = (uint32_t *)&command_mem[write_ptr];
		*p = sizeof(T);
		write_ptr += sizeof(uint32_t);
		// allocate the command
		T *cmd = memnew_placement(&command_mem[write_ptr], T);
		write_ptr += sizeof(T);
		return cmd;
	}

	template <class T>
	T *allocate_and_lock() {

		lock();
		T *ret;

		while ((ret = allocate<T>()) == NULL) {

			unlock();
			// sleep a little until fetch happened and some room is made
			wait_for_flush();
			lock();
		}

		return ret;
	}

	bool flush_one() {

	tryagain:

		// tried to read an empty queue
		if (read_ptr == write_ptr)
			return false;

		uint32_t size = *(uint32_t *)(&command_mem[read_ptr]);

		if (size == 0) {
			//end of ringbuffer, wrap
			read_ptr = 0;
			goto tryagain;
		}

		read_ptr += sizeof(uint32_t);

		CommandBase *cmd = reinterpret_cast<CommandBase *>(&command_mem[read_ptr]);

		cmd->call();
		cmd->~CommandBase();

		read_ptr += size;

		return true;
	}

	void lock();
	void unlock();
	void wait_for_flush();
	SyncSemaphore *_alloc_sync_sem();

public:
	/* NORMAL PUSH COMMANDS */

	template <class T, class M>
	void push(T *p_instance, M p_method) {

		Command0<T, M> *cmd = allocate_and_lock<Command0<T, M> >();

		cmd->instance = p_instance;
		cmd->method = p_method;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1>
	void push(T *p_instance, M p_method, P1 p1) {

		Command1<T, M, P1> *cmd = allocate_and_lock<Command1<T, M, P1> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2>
	void push(T *p_instance, M p_method, P1 p1, P2 p2) {

		Command2<T, M, P1, P2> *cmd = allocate_and_lock<Command2<T, M, P1, P2> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3) {

		Command3<T, M, P1, P2, P3> *cmd = allocate_and_lock<Command3<T, M, P1, P2, P3> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3, class P4>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4) {

		Command4<T, M, P1, P2, P3, P4> *cmd = allocate_and_lock<Command4<T, M, P1, P2, P3, P4> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {

		Command5<T, M, P1, P2, P3, P4, P5> *cmd = allocate_and_lock<Command5<T, M, P1, P2, P3, P4, P5> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {

		Command6<T, M, P1, P2, P3, P4, P5, P6> *cmd = allocate_and_lock<Command6<T, M, P1, P2, P3, P4, P5, P6> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {

		Command7<T, M, P1, P2, P3, P4, P5, P6, P7> *cmd = allocate_and_lock<Command7<T, M, P1, P2, P3, P4, P5, P6, P7> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;

		unlock();

		if (sync) sync->post();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
	void push(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {

		Command8<T, M, P1, P2, P3, P4, P5, P6, P7, P8> *cmd = allocate_and_lock<Command8<T, M, P1, P2, P3, P4, P5, P6, P7, P8> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;
		cmd->p8 = p8;

		unlock();

		if (sync) sync->post();
	}
	/*** PUSH AND RET COMMANDS ***/

	template <class T, class M, class R>
	void push_and_ret(T *p_instance, M p_method, R *r_ret) {

		CommandRet0<T, M, R> *cmd = allocate_and_lock<CommandRet0<T, M, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, R *r_ret) {

		CommandRet1<T, M, P1, R> *cmd = allocate_and_lock<CommandRet1<T, M, P1, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, R *r_ret) {

		CommandRet2<T, M, P1, P2, R> *cmd = allocate_and_lock<CommandRet2<T, M, P1, P2, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, R *r_ret) {

		CommandRet3<T, M, P1, P2, P3, R> *cmd = allocate_and_lock<CommandRet3<T, M, P1, P2, P3, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, R *r_ret) {

		CommandRet4<T, M, P1, P2, P3, P4, R> *cmd = allocate_and_lock<CommandRet4<T, M, P1, P2, P3, P4, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, R *r_ret) {

		CommandRet5<T, M, P1, P2, P3, P4, P5, R> *cmd = allocate_and_lock<CommandRet5<T, M, P1, P2, P3, P4, P5, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, R *r_ret) {

		CommandRet6<T, M, P1, P2, P3, P4, P5, P6, R> *cmd = allocate_and_lock<CommandRet6<T, M, P1, P2, P3, P4, P5, P6, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, R *r_ret) {

		CommandRet7<T, M, P1, P2, P3, P4, P5, P6, P7, R> *cmd = allocate_and_lock<CommandRet7<T, M, P1, P2, P3, P4, P5, P6, P7, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8, class R>
	void push_and_ret(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, R *r_ret) {

		CommandRet8<T, M, P1, P2, P3, P4, P5, P6, P7, P8, R> *cmd = allocate_and_lock<CommandRet8<T, M, P1, P2, P3, P4, P5, P6, P7, P8, R> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;
		cmd->p8 = p8;
		cmd->ret = r_ret;
		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M>
	void push_and_sync(T *p_instance, M p_method) {

		CommandSync0<T, M> *cmd = allocate_and_lock<CommandSync0<T, M> >();

		cmd->instance = p_instance;
		cmd->method = p_method;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1>
	void push_and_sync(T *p_instance, M p_method, P1 p1) {

		CommandSync1<T, M, P1> *cmd = allocate_and_lock<CommandSync1<T, M, P1> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2) {

		CommandSync2<T, M, P1, P2> *cmd = allocate_and_lock<CommandSync2<T, M, P1, P2> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3) {

		CommandSync3<T, M, P1, P2, P3> *cmd = allocate_and_lock<CommandSync3<T, M, P1, P2, P3> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4) {

		CommandSync4<T, M, P1, P2, P3, P4> *cmd = allocate_and_lock<CommandSync4<T, M, P1, P2, P3, P4> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {

		CommandSync5<T, M, P1, P2, P3, P4, P5> *cmd = allocate_and_lock<CommandSync5<T, M, P1, P2, P3, P4, P5> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {

		CommandSync6<T, M, P1, P2, P3, P4, P5, P6> *cmd = allocate_and_lock<CommandSync6<T, M, P1, P2, P3, P4, P5, P6> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {

		CommandSync7<T, M, P1, P2, P3, P4, P5, P6, P7> *cmd = allocate_and_lock<CommandSync7<T, M, P1, P2, P3, P4, P5, P6, P7> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	template <class T, class M, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
	void push_and_sync(T *p_instance, M p_method, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {

		CommandSync8<T, M, P1, P2, P3, P4, P5, P6, P7, P8> *cmd = allocate_and_lock<CommandSync8<T, M, P1, P2, P3, P4, P5, P6, P7, P8> >();

		cmd->instance = p_instance;
		cmd->method = p_method;
		cmd->p1 = p1;
		cmd->p2 = p2;
		cmd->p3 = p3;
		cmd->p4 = p4;
		cmd->p5 = p5;
		cmd->p6 = p6;
		cmd->p7 = p7;
		cmd->p8 = p8;

		SyncSemaphore *ss = _alloc_sync_sem();
		cmd->sync = ss;

		unlock();

		if (sync) sync->post();
		ss->sem->wait();
	}

	void wait_and_flush_one() {
		ERR_FAIL_COND(!sync);
		sync->wait();
		lock();
		flush_one();
		unlock();
	}

	void flush_all() {

		//ERR_FAIL_COND(sync);
		lock();
		while (true) {
			bool exit = !flush_one();
			if (exit)
				break;
		}
		unlock();
	}

	CommandQueueMT(bool p_sync);
	~CommandQueueMT();
};

#endif
