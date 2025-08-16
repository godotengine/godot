#include "microthread.hpp"

extern "C" void microthread_set_tp(void*);

namespace microthread
{
	static Thread main_thread {nullptr};

	void trampoline(Thread* thread)
	{
		thread->startfunc();
	}
	void oneshot_exit()
	{
		auto* thread = self();
		// after this point stack unusable
		free((char *)thread + sizeof(Thread) - Thread::STACK_SIZE);
		syscall(THREAD_SYSCALLS_BASE+1, 0);
		__builtin_unreachable();
	}

	/* glibc sets up its own main thread, *required* by C++ exceptions */
#ifndef __GLIBC__
	__attribute__((constructor, used))
	static void init_threads()
	{
		microthread_set_tp(&main_thread);
	}
#endif
}

asm(".section .text\n"
".global microthread_set_tp\n"
".type microthread_set_tp, @function\n"
"microthread_set_tp:\n"
"  mv tp, a0\n"
"  ret\n");

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

// This function never returns (so no ret)
asm(".global threadcall_destructor\n"
".type threadcall_destructor, @function\n"
"threadcall_destructor:\n"
"	li a7, " STRINGIFY(THREAD_SYSCALLS_BASE+9) "\n"
"	ecall\n");
