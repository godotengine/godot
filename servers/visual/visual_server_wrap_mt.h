/*************************************************************************/
/*  visual_server_wrap_mt.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef VISUAL_SERVER_WRAP_MT_H
#define VISUAL_SERVER_WRAP_MT_H


#include "servers/visual_server.h"
#include "command_queue_mt.h"
#include "os/thread.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class VisualServerWrapMT : public VisualServer {

	// the real visual server
	mutable VisualServer *visual_server;
	
	mutable CommandQueueMT command_queue;
	
	static void _thread_callback(void *_instance);
	void thread_loop();

	Thread::ID server_thread;
	volatile bool exit;
	Thread *thread;
	volatile bool draw_thread_up;
	bool create_thread;
	
	Mutex *draw_mutex;
	int draw_pending;
	void thread_draw();
	void thread_flush();

	void thread_exit();

	Mutex*alloc_mutex;


	int texture_pool_max_size;
	List<RID> texture_id_pool;

	int mesh_pool_max_size;
	List<RID> mesh_id_pool;

//#define DEBUG_SYNC

#ifdef DEBUG_SYNC
#define SYNC_DEBUG print_line("sync on: "+String(__FUNCTION__));
#else
#define SYNC_DEBUG
#endif

public:

#define FUNC0R(m_r,m_type)\
	virtual m_r m_type() { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type();\
		}\
	}

#define FUNCRID(m_type)\
	int m_type##allocn() {\
		for(int i=0;i<m_type##_pool_max_size;i++) {\
			m_type##_id_pool.push_back( visual_server->m_type##_create() );\
		}\
		return 0;\
	}\
	void m_type##_free_cached_ids() {\
		while (m_type##_id_pool.size()) {\
			free(m_type##_id_pool.front()->get());\
			m_type##_id_pool.pop_front();\
		}\
	}\
	virtual RID m_type##_create() { \
		if (Thread::get_caller_ID()!=server_thread) {\
			RID rid;\
			alloc_mutex->lock();\
			if (m_type##_id_pool.size()==0) {\
				int ret;\
				command_queue.push_and_ret( this, &VisualServerWrapMT::m_type##allocn,&ret);\
			}\
			rid=m_type##_id_pool.front()->get();\
			m_type##_id_pool.pop_front();\
			alloc_mutex->unlock();\
			return rid;\
		} else {\
			return visual_server->m_type##_create();\
		}\
	}

#define FUNC0RC(m_r,m_type)\
	virtual m_r m_type() const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type();\
		}\
	}


#define FUNC0(m_type)\
	virtual void m_type() { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type);\
		} else {\
			visual_server->m_type();\
		}\
	}

#define FUNC0C(m_type)\
	virtual void m_type() const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type);\
		} else {\
			visual_server->m_type();\
		}\
	}


#define FUNC0S(m_type)\
	virtual void m_type() { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type);\
		} else {\
			visual_server->m_type();\
		}\
	}

#define FUNC0SC(m_type)\
	virtual void m_type() const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type);\
		} else {\
			visual_server->m_type();\
		}\
	}


///////////////////////////////////////////////


#define FUNC1R(m_r,m_type,m_arg1)\
	virtual m_r m_type(m_arg1 p1) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1);\
		}\
	}

#define FUNC1RC(m_r,m_type,m_arg1)\
	virtual m_r m_type(m_arg1 p1) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1);\
		}\
	}


#define FUNC1S(m_type,m_arg1)\
	virtual void m_type(m_arg1 p1) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1);\
		} else {\
			visual_server->m_type(p1);\
		}\
	}

#define FUNC1SC(m_type,m_arg1)\
	virtual void m_type(m_arg1 p1) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1);\
		} else {\
			visual_server->m_type(p1);\
		}\
	}


#define FUNC1(m_type,m_arg1)\
	virtual void m_type(m_arg1 p1) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1);\
		} else {\
			visual_server->m_type(p1);\
		}\
	}

#define FUNC1C(m_type,m_arg1)\
	virtual void m_type(m_arg1 p1) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1);\
		} else {\
			visual_server->m_type(p1);\
		}\
	}




#define FUNC2R(m_r,m_type,m_arg1, m_arg2)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2);\
		}\
	}

#define FUNC2RC(m_r,m_type,m_arg1, m_arg2)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2);\
		}\
	}


#define FUNC2S(m_type,m_arg1, m_arg2)\
	virtual void m_type(m_arg1 p1, m_arg2 p2) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2);\
		} else {\
			visual_server->m_type(p1, p2);\
		}\
	}

#define FUNC2SC(m_type,m_arg1, m_arg2)\
	virtual void m_type(m_arg1 p1, m_arg2 p2) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2);\
		} else {\
			visual_server->m_type(p1, p2);\
		}\
	}


#define FUNC2(m_type,m_arg1, m_arg2)\
	virtual void m_type(m_arg1 p1, m_arg2 p2) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2);\
		} else {\
			visual_server->m_type(p1, p2);\
		}\
	}

#define FUNC2C(m_type,m_arg1, m_arg2)\
	virtual void m_type(m_arg1 p1, m_arg2 p2) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2);\
		} else {\
			visual_server->m_type(p1, p2);\
		}\
	}




#define FUNC3R(m_r,m_type,m_arg1, m_arg2, m_arg3)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3);\
		}\
	}

#define FUNC3RC(m_r,m_type,m_arg1, m_arg2, m_arg3)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3,&ret);\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3);\
		}\
	}


#define FUNC3S(m_type,m_arg1, m_arg2, m_arg3)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3);\
		} else {\
			visual_server->m_type(p1, p2, p3);\
		}\
	}

#define FUNC3SC(m_type,m_arg1, m_arg2, m_arg3)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3);\
		} else {\
			visual_server->m_type(p1, p2, p3);\
		}\
	}


#define FUNC3(m_type,m_arg1, m_arg2, m_arg3)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3);\
		} else {\
			visual_server->m_type(p1, p2, p3);\
		}\
	}

#define FUNC3C(m_type,m_arg1, m_arg2, m_arg3)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3);\
		} else {\
			visual_server->m_type(p1, p2, p3);\
		}\
	}




#define FUNC4R(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4);\
		}\
	}

#define FUNC4RC(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4);\
		}\
	}


#define FUNC4S(m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4);\
		}\
	}

#define FUNC4SC(m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4);\
		}\
	}


#define FUNC4(m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4);\
		}\
	}

#define FUNC4C(m_type,m_arg1, m_arg2, m_arg3, m_arg4)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4);\
		}\
	}




#define FUNC5R(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}

#define FUNC5RC(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}


#define FUNC5S(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}

#define FUNC5SC(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}


#define FUNC5(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}

#define FUNC5C(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5);\
		}\
	}




#define FUNC6R(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}

#define FUNC6RC(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6,&ret);\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}


#define FUNC6S(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}

#define FUNC6SC(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}


#define FUNC6(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}

#define FUNC6C(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6);\
		}\
	}




#define FUNC7R(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}

#define FUNC7RC(m_r,m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual m_r m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			m_r ret;\
			command_queue.push_and_ret( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7,&ret);\
			SYNC_DEBUG\
			return ret;\
		} else {\
			return visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}


#define FUNC7S(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}

#define FUNC7SC(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push_and_sync( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}


#define FUNC7(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}

#define FUNC7C(m_type,m_arg1, m_arg2, m_arg3, m_arg4, m_arg5, m_arg6, m_arg7)\
	virtual void m_type(m_arg1 p1, m_arg2 p2, m_arg3 p3, m_arg4 p4, m_arg5 p5, m_arg6 p6, m_arg7 p7) const { \
		if (Thread::get_caller_ID()!=server_thread) {\
			command_queue.push( visual_server, &VisualServer::m_type,p1, p2, p3, p4, p5, p6, p7);\
		} else {\
			visual_server->m_type(p1, p2, p3, p4, p5, p6, p7);\
		}\
	}




	//FUNC0R(RID,texture_create);
	FUNCRID(texture);
	FUNC5(texture_allocate,RID,int,int,Image::Format,uint32_t);
	FUNC3(texture_set_data,RID,const Image&,CubeMapSide);
	FUNC2RC(Image,texture_get_data,RID,CubeMapSide);
	FUNC2(texture_set_flags,RID,uint32_t);
	FUNC1RC(Image::Format,texture_get_format,RID);
	FUNC1RC(uint32_t,texture_get_flags,RID);
	FUNC1RC(uint32_t,texture_get_width,RID);
	FUNC1RC(uint32_t,texture_get_height,RID);
	FUNC3(texture_set_size_override,RID,int,int);
	FUNC1RC(bool,texture_can_stream,RID);
	FUNC3C(texture_set_reload_hook,RID,ObjectID,const StringName&);

	/* SHADER API */

	FUNC1R(RID,shader_create,ShaderMode);
	FUNC2(shader_set_mode,RID,ShaderMode);
	FUNC1RC(ShaderMode,shader_get_mode,RID);
	FUNC7(shader_set_code,RID,const String&,const String&,const String&,int,int,int);
	FUNC1RC(String,shader_get_vertex_code,RID);
	FUNC1RC(String,shader_get_fragment_code,RID);
	FUNC1RC(String,shader_get_light_code,RID);
	FUNC2SC(shader_get_param_list,RID,List<PropertyInfo>*);

	FUNC3(shader_set_default_texture_param,RID,const StringName&,RID);
	FUNC2RC(RID,shader_get_default_texture_param,RID,const StringName&);


	/*virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) {
		if (Thread::get_caller_ID()!=server_thread) {
			command_queue.push_and_sync( visual_server, &VisualServer::shader_get_param_list,p_shader,p_param_list);
		} else {
			visual_server->m_type(p1, p2, p3, p4, p5);
		}
	}*/

//	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list);


	/* COMMON MATERIAL API */

	FUNC0R(RID,material_create);
	FUNC2(material_set_shader,RID,RID);
	FUNC1RC(RID,material_get_shader,RID);

	FUNC3(material_set_param,RID,const StringName&,const Variant&);
	FUNC2RC(Variant,material_get_param,RID,const StringName&);

	FUNC3(material_set_flag,RID,MaterialFlag,bool);
	FUNC2RC(bool,material_get_flag,RID,MaterialFlag);

	FUNC2(material_set_depth_draw_mode,RID,MaterialDepthDrawMode);
	FUNC1RC(MaterialDepthDrawMode,material_get_depth_draw_mode,RID);

	FUNC2(material_set_blend_mode,RID,MaterialBlendMode);
	FUNC1RC(MaterialBlendMode,material_get_blend_mode,RID);

	FUNC2(material_set_line_width,RID,float);
	FUNC1RC(float,material_get_line_width,RID);

	/* FIXED MATERIAL */


	FUNC0R(RID,fixed_material_create);

	FUNC3(fixed_material_set_flag,RID, FixedMaterialFlags , bool );
	FUNC2RC(bool, fixed_material_get_flag,RID, FixedMaterialFlags);

	FUNC3(fixed_material_set_param,RID, FixedMaterialParam, const Variant& );
	FUNC2RC(Variant, fixed_material_get_param,RID ,FixedMaterialParam);

	FUNC3(fixed_material_set_texture,RID ,FixedMaterialParam, RID );
	FUNC2RC(RID, fixed_material_get_texture,RID,FixedMaterialParam);



	FUNC3(fixed_material_set_texcoord_mode,RID,FixedMaterialParam, FixedMaterialTexCoordMode );
	FUNC2RC(FixedMaterialTexCoordMode, fixed_material_get_texcoord_mode,RID,FixedMaterialParam);

	FUNC2(fixed_material_set_light_shader,RID,FixedMaterialLightShader);
	FUNC1RC(FixedMaterialLightShader, fixed_material_get_light_shader,RID);

	FUNC2(fixed_material_set_uv_transform,RID,const Transform&);
	FUNC1RC(Transform, fixed_material_get_uv_transform,RID);

	FUNC2(fixed_material_set_point_size,RID ,float);
	FUNC1RC(float,fixed_material_get_point_size,RID);

	/* SURFACE API */
	FUNCRID(mesh);

	FUNC2(mesh_set_morph_target_count,RID,int);
	FUNC1RC(int,mesh_get_morph_target_count,RID);


	FUNC2(mesh_set_morph_target_mode,RID,MorphTargetMode);
	FUNC1RC(MorphTargetMode,mesh_get_morph_target_mode,RID);

	FUNC2(mesh_add_custom_surface,RID,const Variant&); //this is used by each platform in a different way

	FUNC5(mesh_add_surface,RID,PrimitiveType,const Array&,const Array&,bool);
	FUNC2RC(Array,mesh_get_surface_arrays,RID,int);
	FUNC2RC(Array,mesh_get_surface_morph_arrays,RID,int);

	FUNC4(mesh_surface_set_material,RID, int, RID,bool);
	FUNC2RC(RID,mesh_surface_get_material,RID, int);

	FUNC2RC(int,mesh_surface_get_array_len,RID, int);
	FUNC2RC(int,mesh_surface_get_array_index_len,RID, int);
	FUNC2RC(uint32_t,mesh_surface_get_format,RID, int);
	FUNC2RC(PrimitiveType,mesh_surface_get_primitive_type,RID, int);

	FUNC2(mesh_remove_surface,RID,int);
	FUNC1RC(int,mesh_get_surface_count,RID);


	FUNC2(mesh_set_custom_aabb,RID,const AABB&);
	FUNC1RC(AABB,mesh_get_custom_aabb,RID);


	/* MULTIMESH API */

	FUNC0R(RID,multimesh_create);
	FUNC2(multimesh_set_instance_count,RID,int);
	FUNC1RC(int,multimesh_get_instance_count,RID);

	FUNC2(multimesh_set_mesh,RID,RID);
	FUNC2(multimesh_set_aabb,RID,const AABB&);
	FUNC3(multimesh_instance_set_transform,RID,int,const Transform&);
	FUNC3(multimesh_instance_set_color,RID,int,const Color&);

	FUNC1RC(RID,multimesh_get_mesh,RID);
	FUNC2RC(AABB,multimesh_get_aabb,RID,const AABB&);
	FUNC2RC(Transform,multimesh_instance_get_transform,RID,int);
	FUNC2RC(Color,multimesh_instance_get_color,RID,int);

	FUNC2(multimesh_set_visible_instances,RID,int);
	FUNC1RC(int,multimesh_get_visible_instances,RID);

	/* IMMEDIATE API */


	FUNC0R(RID,immediate_create);
	FUNC3(immediate_begin,RID,PrimitiveType,RID);
	FUNC2(immediate_vertex,RID,const Vector3&);
	FUNC2(immediate_normal,RID,const Vector3&);
	FUNC2(immediate_tangent,RID,const Plane&);
	FUNC2(immediate_color,RID,const Color&);
	FUNC2(immediate_uv,RID,const Vector2&);
	FUNC2(immediate_uv2,RID,const Vector2&);
	FUNC1(immediate_end,RID);
	FUNC1(immediate_clear,RID);
	FUNC2(immediate_set_material,RID,RID);
	FUNC1RC(RID,immediate_get_material,RID);


	/* PARTICLES API */

	FUNC0R(RID,particles_create);

	FUNC2(particles_set_amount,RID, int );
	FUNC1RC(int,particles_get_amount,RID);

	FUNC2(particles_set_emitting,RID, bool );
	FUNC1RC(bool,particles_is_emitting,RID);

	FUNC2(particles_set_visibility_aabb,RID, const AABB&);
	FUNC1RC(AABB,particles_get_visibility_aabb,RID);

	FUNC2(particles_set_emission_half_extents,RID, const Vector3&);
	FUNC1RC(Vector3,particles_get_emission_half_extents,RID);

	FUNC2(particles_set_emission_base_velocity,RID, const Vector3&);
	FUNC1RC(Vector3,particles_get_emission_base_velocity,RID);

	FUNC2(particles_set_emission_points,RID, const DVector<Vector3>& );
	FUNC1RC(DVector<Vector3>,particles_get_emission_points,RID);

	FUNC2(particles_set_gravity_normal,RID, const Vector3& );
	FUNC1RC(Vector3,particles_get_gravity_normal,RID);

	FUNC3(particles_set_variable,RID, ParticleVariable ,float);
	FUNC2RC(float,particles_get_variable,RID, ParticleVariable );

	FUNC3(particles_set_randomness,RID, ParticleVariable ,float);
	FUNC2RC(float,particles_get_randomness,RID, ParticleVariable );

	FUNC3(particles_set_color_phase_pos,RID, int , float);
	FUNC2RC(float,particles_get_color_phase_pos,RID, int );

	FUNC2(particles_set_color_phases,RID, int );
	FUNC1RC(int,particles_get_color_phases,RID);

	FUNC3(particles_set_color_phase_color,RID, int , const Color& );
	FUNC2RC(Color,particles_get_color_phase_color,RID, int );

	FUNC2(particles_set_attractors,RID, int);
	FUNC1RC(int,particles_get_attractors,RID);

	FUNC3(particles_set_attractor_pos,RID, int, const Vector3&);
	FUNC2RC(Vector3,particles_get_attractor_pos,RID,int);

	FUNC3(particles_set_attractor_strength,RID, int, float);
	FUNC2RC(float,particles_get_attractor_strength,RID,int);

	FUNC3(particles_set_material,RID, RID,bool);
	FUNC1RC(RID,particles_get_material,RID);

	FUNC2(particles_set_height_from_velocity,RID, bool);
	FUNC1RC(bool,particles_has_height_from_velocity,RID);

	FUNC2(particles_set_use_local_coordinates,RID, bool);
	FUNC1RC(bool,particles_is_using_local_coordinates,RID);


	/* Light API */

	FUNC1R(RID,light_create,LightType);
	FUNC1RC(LightType,light_get_type,RID);

	FUNC3(light_set_color,RID,LightColor , const Color& );
	FUNC2RC(Color,light_get_color,RID,LightColor );


	FUNC2(light_set_shadow,RID,bool );
	FUNC1RC(bool,light_has_shadow,RID);

	FUNC2(light_set_volumetric,RID,bool );
	FUNC1RC(bool,light_is_volumetric,RID);

	FUNC2(light_set_projector,RID,RID );
	FUNC1RC(RID,light_get_projector,RID);

	FUNC3(light_set_param,RID, LightParam , float );
	FUNC2RC(float,light_get_param,RID, LightParam );

	FUNC2(light_set_operator,RID,LightOp);
	FUNC1RC(LightOp,light_get_operator,RID);

	FUNC2(light_omni_set_shadow_mode,RID,LightOmniShadowMode);
	FUNC1RC(LightOmniShadowMode,light_omni_get_shadow_mode,RID);

	FUNC2(light_directional_set_shadow_mode,RID,LightDirectionalShadowMode);
	FUNC1RC(LightDirectionalShadowMode,light_directional_get_shadow_mode,RID);
	FUNC3(light_directional_set_shadow_param,RID,LightDirectionalShadowParam, float );
	FUNC2RC(float,light_directional_get_shadow_param,RID,LightDirectionalShadowParam );


	/* SKELETON API */

	FUNC0R(RID,skeleton_create);
	FUNC2(skeleton_resize,RID,int );
	FUNC1RC(int,skeleton_get_bone_count,RID) ;
	FUNC3(skeleton_bone_set_transform,RID,int, const Transform&);
	FUNC2R(Transform,skeleton_bone_get_transform,RID,int );

	/* ROOM API */

	FUNC0R(RID,room_create);
	FUNC2(room_set_bounds,RID, const BSP_Tree&);
	FUNC1RC(BSP_Tree,room_get_bounds,RID);

	/* PORTAL API */

	FUNC0R(RID,portal_create);
	FUNC2(portal_set_shape,RID,const Vector<Point2>&);
	FUNC1RC(Vector<Point2>,portal_get_shape,RID);
	FUNC2(portal_set_enabled,RID, bool);
	FUNC1RC(bool,portal_is_enabled,RID);
	FUNC2(portal_set_disable_distance,RID, float);
	FUNC1RC(float,portal_get_disable_distance,RID);
	FUNC2(portal_set_disabled_color,RID, const Color&);
	FUNC1RC(Color,portal_get_disabled_color,RID);
	FUNC2(portal_set_connect_range,RID, float);
	FUNC1RC(float,portal_get_connect_range,RID);


	FUNC0R(RID,baked_light_create);
	FUNC2(baked_light_set_mode,RID,BakedLightMode);
	FUNC1RC(BakedLightMode,baked_light_get_mode,RID);

	FUNC2(baked_light_set_octree,RID,DVector<uint8_t>);
	FUNC1RC(DVector<uint8_t>,baked_light_get_octree,RID);

	FUNC2(baked_light_set_light,RID,DVector<uint8_t>);
	FUNC1RC(DVector<uint8_t>,baked_light_get_light,RID);

	FUNC2(baked_light_set_sampler_octree,RID,const DVector<int>&);
	FUNC1RC(DVector<int>,baked_light_get_sampler_octree,RID);

	FUNC2(baked_light_set_lightmap_multiplier,RID,float);
	FUNC1RC(float,baked_light_get_lightmap_multiplier,RID);

	FUNC3(baked_light_add_lightmap,RID,RID,int);
	FUNC1(baked_light_clear_lightmaps,RID);


	FUNC0R(RID,baked_light_sampler_create);

	FUNC3(baked_light_sampler_set_param,RID, BakedLightSamplerParam , float );
	FUNC2RC(float,baked_light_sampler_get_param,RID, BakedLightSamplerParam );

	FUNC2(baked_light_sampler_set_resolution,RID,int);
	FUNC1RC(int,baked_light_sampler_get_resolution,RID);

	/* CAMERA API */

	FUNC0R(RID,camera_create);
	FUNC4(camera_set_perspective,RID,float , float , float );
	FUNC4(camera_set_orthogonal,RID,float, float , float );
	FUNC2(camera_set_transform,RID,const Transform& );

	FUNC2(camera_set_visible_layers,RID,uint32_t);
	FUNC1RC(uint32_t,camera_get_visible_layers,RID);

	FUNC2(camera_set_environment,RID,RID);
	FUNC1RC(RID,camera_get_environment,RID);


	FUNC2(camera_set_use_vertical_aspect,RID,bool);
	FUNC2RC(bool,camera_is_using_vertical_aspect,RID,bool);


	/* VIEWPORT API */

	FUNC0R(RID,viewport_create);

	FUNC2(viewport_attach_to_screen,RID,int );
	FUNC1(viewport_detach,RID);

	FUNC2(viewport_set_as_render_target,RID,bool);
	FUNC2(viewport_set_render_target_update_mode,RID,RenderTargetUpdateMode);
	FUNC1RC(RenderTargetUpdateMode,viewport_get_render_target_update_mode,RID);
	FUNC1RC(RID,viewport_get_render_target_texture,RID);

	FUNC2(viewport_set_render_target_vflip,RID,bool);
	FUNC1RC(bool,viewport_get_render_target_vflip,RID);
	FUNC2(viewport_set_render_target_to_screen_rect,RID,const Rect2&);

	FUNC1(viewport_queue_screen_capture,RID);
	FUNC1RC(Image,viewport_get_screen_capture,RID);

	FUNC2(viewport_set_rect,RID,const ViewportRect&);
	FUNC1RC(ViewportRect,viewport_get_rect,RID);

	FUNC2(viewport_set_hide_scenario,RID,bool );
	FUNC2(viewport_set_hide_canvas,RID,bool );
	FUNC2(viewport_attach_camera,RID,RID );
	FUNC2(viewport_set_scenario,RID,RID );

	FUNC1RC(RID,viewport_get_attached_camera,RID);
	FUNC1RC(RID,viewport_get_scenario,RID );
	FUNC2(viewport_attach_canvas,RID,RID);
	FUNC2(viewport_remove_canvas,RID,RID);
	FUNC3(viewport_set_canvas_transform,RID,RID,const Matrix32&);
	FUNC2RC(Matrix32,viewport_get_canvas_transform,RID,RID);
	FUNC2(viewport_set_global_canvas_transform,RID,const Matrix32&);
	FUNC1RC(Matrix32,viewport_get_global_canvas_transform,RID);
	FUNC3(viewport_set_canvas_layer,RID,RID ,int);
	FUNC2(viewport_set_transparent_background,RID,bool);
	FUNC1RC(bool,viewport_has_transparent_background,RID);


	/* ENVIRONMENT API */

	FUNC0R(RID,environment_create);

	FUNC2(environment_set_background,RID,EnvironmentBG);
	FUNC1RC(EnvironmentBG,environment_get_background,RID);

	FUNC3(environment_set_background_param,RID,EnvironmentBGParam, const Variant&);
	FUNC2RC(Variant,environment_get_background_param,RID,EnvironmentBGParam );

	FUNC3(environment_set_enable_fx,RID,EnvironmentFx,bool);
	FUNC2RC(bool,environment_is_fx_enabled,RID,EnvironmentFx);


	FUNC3(environment_fx_set_param,RID,EnvironmentFxParam,const Variant&);
	FUNC2RC(Variant,environment_fx_get_param,RID,EnvironmentFxParam);


	/* SCENARIO API */

	FUNC0R(RID,scenario_create);

	FUNC2(scenario_set_debug,RID,ScenarioDebugMode);
	FUNC2(scenario_set_environment,RID, RID);
	FUNC2RC(RID,scenario_get_environment,RID, RID);
	FUNC2(scenario_set_fallback_environment,RID, RID);


	/* INSTANCING API */

	FUNC0R(RID,instance_create);

	FUNC2(instance_set_base,RID, RID);
	FUNC1RC(RID,instance_get_base,RID);

	FUNC2(instance_set_scenario,RID, RID);
	FUNC1RC(RID,instance_get_scenario,RID);

	FUNC2(instance_set_layer_mask,RID, uint32_t);
	FUNC1RC(uint32_t,instance_get_layer_mask,RID);

	FUNC1RC(AABB,instance_get_base_aabb,RID);

	FUNC2(instance_attach_object_instance_ID,RID,uint32_t);
	FUNC1RC(uint32_t,instance_get_object_instance_ID,RID);

	FUNC2(instance_attach_skeleton,RID,RID);
	FUNC1RC(RID,instance_get_skeleton,RID);

	FUNC3(instance_set_morph_target_weight,RID,int, float);
	FUNC2RC(float,instance_get_morph_target_weight,RID,int);

	FUNC2(instance_set_transform,RID, const Transform&);
	FUNC1RC(Transform,instance_get_transform,RID);

	FUNC2(instance_set_exterior,RID, bool );
	FUNC1RC(bool,instance_is_exterior,RID);

	FUNC2(instance_set_room,RID, RID );
	FUNC1RC(RID,instance_get_room,RID ) ;

	FUNC2(instance_set_extra_visibility_margin,RID, real_t  );
	FUNC1RC(real_t,instance_get_extra_visibility_margin,RID );

	FUNC2RC(Vector<RID>,instances_cull_aabb,const AABB& , RID );
	FUNC3RC(Vector<RID>,instances_cull_ray,const Vector3& ,const Vector3&, RID );
	FUNC2RC(Vector<RID>,instances_cull_convex,const Vector<Plane>& , RID );

	FUNC3(instance_geometry_set_flag,RID,InstanceFlags ,bool );
	FUNC2RC(bool,instance_geometry_get_flag,RID,InstanceFlags );

	FUNC2(instance_geometry_set_material_override,RID, RID );
	FUNC1RC(RID,instance_geometry_get_material_override,RID);

	FUNC3(instance_geometry_set_draw_range,RID,float ,float);
	FUNC1RC(float,instance_geometry_get_draw_range_max,RID);
	FUNC1RC(float,instance_geometry_get_draw_range_min,RID);

	FUNC2(instance_geometry_set_baked_light,RID, RID );
	FUNC1RC(RID,instance_geometry_get_baked_light,RID);

	FUNC2(instance_geometry_set_baked_light_sampler,RID, RID );
	FUNC1RC(RID,instance_geometry_get_baked_light_sampler,RID);

	FUNC2(instance_geometry_set_baked_light_texture_index,RID, int);
	FUNC1RC(int,instance_geometry_get_baked_light_texture_index,RID);

	FUNC2(instance_light_set_enabled,RID,bool);
	FUNC1RC(bool,instance_light_is_enabled,RID);

	/* CANVAS (2D) */

	FUNC0R(RID,canvas_create);
	FUNC3(canvas_set_item_mirroring,RID,RID,const Point2&);
	FUNC2RC(Point2,canvas_get_item_mirroring,RID,RID);

	FUNC0R(RID,canvas_item_create);

	FUNC2(canvas_item_set_parent,RID,RID );
	FUNC1RC(RID,canvas_item_get_parent,RID);

	FUNC2(canvas_item_set_visible,RID,bool );
	FUNC1RC(bool,canvas_item_is_visible,RID);

	FUNC2(canvas_item_set_blend_mode,RID,MaterialBlendMode );


	//FUNC(canvas_item_set_rect,RID, const Rect2& p_rect);
	FUNC2(canvas_item_set_transform,RID, const Matrix32& );
	FUNC2(canvas_item_set_clip,RID, bool );
	FUNC3(canvas_item_set_custom_rect,RID, bool ,const Rect2&);
	FUNC2(canvas_item_set_opacity,RID, float );
	FUNC2RC(float,canvas_item_get_opacity,RID, float );
	FUNC2(canvas_item_set_on_top,RID, bool );
	FUNC1RC(bool,canvas_item_is_on_top,RID);

	FUNC2(canvas_item_set_self_opacity,RID, float );
	FUNC2RC(float,canvas_item_get_self_opacity,RID, float );

	FUNC2(canvas_item_attach_viewport,RID, RID );

	FUNC5(canvas_item_add_line,RID, const Point2& , const Point2& ,const Color& ,float );
	FUNC3(canvas_item_add_rect,RID, const Rect2& , const Color& );
	FUNC4(canvas_item_add_circle,RID, const Point2& , float ,const Color& );
	FUNC5(canvas_item_add_texture_rect,RID, const Rect2& , RID ,bool ,const Color& );
	FUNC5(canvas_item_add_texture_rect_region,RID, const Rect2& , RID ,const Rect2& ,const Color& );

	FUNC7(canvas_item_add_style_box,RID, const Rect2& , RID ,const Vector2& ,const Vector2&, bool ,const Color& );
	FUNC6(canvas_item_add_primitive,RID, const Vector<Point2>& , const Vector<Color>& ,const Vector<Point2>& , RID ,float );
	FUNC5(canvas_item_add_polygon,RID, const Vector<Point2>& , const Vector<Color>& ,const Vector<Point2>& , RID );
	FUNC7(canvas_item_add_triangle_array,RID, const Vector<int>& , const Vector<Point2>& , const Vector<Color>& ,const Vector<Point2>& , RID , int );
	FUNC7(canvas_item_add_triangle_array_ptr,RID, int , const int* , const Point2* , const Color* ,const Point2* , RID );


	FUNC2(canvas_item_add_set_transform,RID,const Matrix32& );
	FUNC2(canvas_item_add_set_blend_mode,RID, MaterialBlendMode );
	FUNC2(canvas_item_add_clip_ignore,RID, bool );

	FUNC2(canvas_item_set_sort_children_by_y,RID,bool);
	FUNC2(canvas_item_set_z,RID,int);
	FUNC2(canvas_item_set_z_as_relative_to_parent,RID,bool);

	FUNC2(canvas_item_set_shader,RID, RID );
	FUNC1RC(RID,canvas_item_get_shader,RID );

	FUNC2(canvas_item_set_use_parent_shader,RID, bool );


	FUNC3(canvas_item_set_shader_param,RID,const StringName&,const Variant&);
	FUNC2RC(Variant,canvas_item_get_shader_param,RID,const StringName&);

	FUNC1(canvas_item_clear,RID);
	FUNC1(canvas_item_raise,RID);

	/* CANVAS LIGHT */
	FUNC0R(RID,canvas_light_create);
	FUNC2(canvas_light_attach_to_canvas,RID,RID);
	FUNC2(canvas_light_set_enabled,RID,bool);
	FUNC2(canvas_light_set_transform,RID,const Matrix32&);
	FUNC2(canvas_light_set_texture,RID,RID);
	FUNC2(canvas_light_set_texture_offset,RID,const Vector2&);
	FUNC2(canvas_light_set_color,RID,const Color&);
	FUNC2(canvas_light_set_height,RID,float);
	FUNC3(canvas_light_set_z_range,RID,int,int);
	FUNC2(canvas_light_set_item_mask,RID,int);

	FUNC2(canvas_light_set_blend_mode,RID,CanvasLightBlendMode);
	FUNC2(canvas_light_set_shadow_enabled,RID,bool);
	FUNC2(canvas_light_set_shadow_buffer_size,RID,int);
	FUNC2(canvas_light_set_shadow_filter,RID,int);

	/* CANVAS OCCLUDER */

	FUNC0R(RID,canvas_light_occluder_create);
	FUNC2(canvas_light_occluder_attach_to_canvas,RID,RID);
	FUNC2(canvas_light_occluder_set_enabled,RID,bool);
	FUNC2(canvas_light_occluder_set_shape,RID,const DVector<Vector2>&);

	/* CURSOR */
	FUNC2(cursor_set_rotation,float , int ); // radians
	FUNC3(cursor_set_texture,RID , const Point2 &, int );
	FUNC2(cursor_set_visible,bool , int );
	FUNC2(cursor_set_pos,const Point2& , int );

	/* BLACK BARS */

	FUNC4(black_bars_set_margins,int , int , int , int );
	FUNC4(black_bars_set_images,RID , RID , RID , RID );

	/* FREE */

	FUNC1(free,RID);

	/* CUSTOM SHADE MODEL */

	FUNC2(custom_shade_model_set_shader,int , RID );
	FUNC1RC(RID,custom_shade_model_get_shader,int );
	FUNC2(custom_shade_model_set_name,int , const String& );
	FUNC1RC(String,custom_shade_model_get_name,int );
	FUNC2(custom_shade_model_set_param_info,int , const List<PropertyInfo>& );
	FUNC2SC(custom_shade_model_get_param_info,int , List<PropertyInfo>* );

	/* EVENT QUEUING */


	virtual void init();
	virtual void finish();
	virtual void draw();
	virtual void flush();
	FUNC0RC(bool,has_changed);

	/* RENDER INFO */

	FUNC1R(int,get_render_info,RenderInfo );
	virtual bool has_feature(Features p_feature) const { return visual_server->has_feature(p_feature); }

	FUNC2(set_boot_image,const Image& , const Color& );
	FUNC1(set_default_clear_color,const Color& );

	FUNC0R(RID,get_test_cube );


	VisualServerWrapMT(VisualServer* p_contained,bool p_create_thread);
	~VisualServerWrapMT();

};


#endif
