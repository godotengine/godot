#include "thread_winrt.h"

#include "os/memory.h"

Thread* ThreadWinrt::create_func_winrt(ThreadCreateCallback p_callback,void *p_user,const Settings&) {

	ThreadWinrt* thread = memnew(ThreadWinrt);
	std::thread new_thread(p_callback, p_user);
	std::swap(thread->thread, new_thread);

	return thread;
};

Thread::ID ThreadWinrt::get_thread_ID_func_winrt() {

	return std::hash<std::thread::id>()(std::this_thread::get_id());
};

void ThreadWinrt::wait_to_finish_func_winrt(Thread* p_thread) {

	ThreadWinrt *tp=static_cast<ThreadWinrt*>(p_thread);
	tp->thread.join();
};


Thread::ID ThreadWinrt::get_ID() const {

	return std::hash<std::thread::id>()(thread.get_id());
};

void ThreadWinrt::make_default() {

};

ThreadWinrt::ThreadWinrt() {

};

ThreadWinrt::~ThreadWinrt() {

};

