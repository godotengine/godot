#ifndef THREAD_WINRT_H
#define THREAD_WINRT_H

#ifdef WINRT_ENABLED

#include "os/thread.h"

#include <thread>

class ThreadWinrt : public Thread {

	std::thread thread;

	static Thread* create_func_winrt(ThreadCreateCallback p_callback,void *,const Settings&);
	static ID get_thread_ID_func_winrt();
	static void wait_to_finish_func_winrt(Thread* p_thread);

	ThreadWinrt();
public:


	virtual ID get_ID() const;

	static void make_default();


	~ThreadWinrt();

};


#endif

#endif

