#pragma once
#include <include/common.hpp>
#include <include/syscall.hpp>
#include <functional>
#include <cstdlib>
#include <memory>

/***
 * Example usage:
 *
 *	auto thread = microthread::create(
 *		[] (int a, int b, int c) -> long {
 *			printf("Hello from a microthread!\n"
 *					"a = %d, b = %d, c = %d\n",
 *					a, b, c);
 *			return a + b + c;
 *		}, 111, 222, 333);
 *
 *  long retval = microthread::join(thread);
 *  printf("microthread exit status: %ld\n", retval);
 *
 *  Note: microthreads require the native threads system calls.
 *  microthreads do not support thread-local storage.
 *  The thread function may also return void, in which case the
 *  return value becomes zero (0).
***/

namespace microthread
{
struct Thread;

/* Create a new thread using the given function @func,
   and pass all further arguments to the function as is.
   Returns the new thread. The new thread starts immediately. */
template <typename T, typename... Args>
auto    create(const T& func, Args&&... args);

/* Create a new self-governing thread that deletes itself on completion.
   Calling exit() in a sovereign thread is undefined behavior.
   Returns thread id on success. */
template <typename T, typename... Args>
int     oneshot(const T& func, Args&&... args);

/* Create a new self-governing thread that directly starts on the
   threads start function, with a special return address that
   self-deletes and exits the thread safely. Arguments cannot be passed,
   but a limited amount of capture storage exists. */
using direct_funcptr = void(*)();
inline long direct(direct_funcptr);

/* Waits for a thread to finish and then returns the exit status
   of the thread. The thread is then deleted, freeing memory. */
long    join(Thread*);

/* Exit the current thread with the given exit status. Never returns. */
void    exit(long status);

/* Yield until condition is true */
template <typename Functor>
void    yield_until(Functor&& condition);
/* Return back to another suspended thread. Returns 0 on success. */
long    yield();
long    yield_to(int tid); /* Return to a specific suspended thread. */
long    yield_to(Thread*);

/* Block a thread with a specific reason. */
long    block(int reason = 0);
template <typename Functor>
void    block(Functor&& condition, int reason = 0);
long    unblock(int tid);
/* Wake thread with @reason that was blocked, returns -1 if nothing happened. */
long    wakeup_one_blocked(int reason);

Thread* self();            /* Returns the current thread */
int     gettid();          /* Returns the current thread id */


/** implementation details **/

struct Thread
{
	static const size_t STACK_SIZE = 256*1024;

	Thread(std::function<void()> start)
	 	: startfunc{std::move(start)} {}

	long resume()   { return yield_to(this); }
	long suspend()  { return yield(); }
	void exit(long status);

	bool has_exited() const;

	~Thread() noexcept {}

	int tid = 0;
	union {
		long  return_value;
		std::function<void()> startfunc;
	};
};
struct ThreadDeleter {
	void operator() (Thread* thread) {
		join(thread);
	}
};
static_assert(Thread::STACK_SIZE > sizeof(Thread) + 16384);
using Thread_ptr = std::unique_ptr<Thread, ThreadDeleter>;

inline bool Thread::has_exited() const {
	return this->tid == 0;
}

inline Thread* self() {
#ifdef __GNUG__
	register Thread* tp asm("tp");
	asm("" : "=r"(tp));
#else
	Thread* tp;
	asm("mv %0, tp" : "=r"(tp));
#endif
	return tp;
}

inline int gettid() {
	return self()->tid;
}

template <typename T, typename... Args>
inline auto create(const T& func, Args&&... args)
{
	static_assert( std::is_invocable_v<T, Args...> );

	char* stack_bot = (char*) malloc(Thread::STACK_SIZE);
	if (stack_bot == nullptr) return Thread_ptr{};
	constexpr size_t ArgSize = sizeof(std::tuple {std::move(args)...});
	char* stack_top = stack_bot + Thread::STACK_SIZE - sizeof(Thread) - ArgSize;
	// store arguments on stack
	auto* tuple = new (stack_top) std::tuple{std::move(args)...};

	// store the thread after arguments (at the very top)
	char* thread_addr = stack_top + ArgSize;
	Thread* thread = new (thread_addr) Thread(
		[func, tuple] () -> void
		{
			if constexpr (std::is_same_v<void, decltype(func(args...))>)
			{
				std::apply(func, std::move(*tuple));
				self()->exit(0);
			} else {
				self()->exit( std::apply(func, std::move(*tuple)) );
			}
		});

	extern void trampoline(Thread*);
	asm("" ::: "memory"); /* avoid dead-store optimization */
	/* stack, func, tls, flags = CHILD_CLEARTID, stack_base, stack_size */
	const long sp   = (long) stack_top;
	const long tls  = (long) thread;
	(void) syscall(THREAD_SYSCALLS_BASE+0, sp, (long) &trampoline, tls, 0x200000,
		(long)stack_bot, Thread::STACK_SIZE);
	// parent path (reordering doesn't matter)
	return Thread_ptr(thread);
}
template <typename T, typename... Args>
inline int oneshot(const T& func, Args&&... args)
{
	static_assert( std::is_invocable_v<T, Args...> );
	static_assert(std::is_same_v<void, decltype(func(args...))>,
				"Free threads have no return value!");
	char* stack_bot = (char*) malloc(Thread::STACK_SIZE);
	if (UNLIKELY(stack_bot == nullptr)) return -12; /* ENOMEM */

	constexpr size_t ArgSize = sizeof(std::tuple {std::move(args)...});
	char* stack_top = stack_bot + Thread::STACK_SIZE - sizeof(Thread) - ArgSize;
	// store arguments on stack
	auto* tuple = new (stack_top) std::tuple{std::move(args)...};

	// store the thread after arguments (at the very top)
	char* thread_addr = stack_top + ArgSize;
	Thread* thread = new (thread_addr) Thread(
		[func, tuple] {
			std::apply(func, std::move(*tuple));
			extern void oneshot_exit();
			oneshot_exit();
		});
	asm ("" ::: "memory"); // prevent dead-store optimization
	extern void trampoline(Thread*);
	const long sp   = (long) stack_top;
	const long tls  = (long) thread;
	return syscall(THREAD_SYSCALLS_BASE+0, sp, (long) &trampoline, tls, 0,
		(long)stack_bot, Thread::STACK_SIZE);
}

extern "C" long threadcall_create(direct_funcptr, void(*)());
extern "C" void threadcall_destructor();

inline long direct(direct_funcptr starter)
{
	register direct_funcptr a0 asm("a0") = starter;
	register void(*a1)()       asm("a1") = threadcall_destructor;
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+8;
	register long a0_out asm("a0");
	// Clobbers memory because it's like a function call
	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*a0), "r"(a1), "m"(*a1), "r"(syscall_id) : "memory");
	return a0_out;
}

inline long join(Thread* thread)
{
	// yield until the tid value is zeroed
	while (!thread->has_exited()) {
		yield();
		// thread->tid might have changed since yielding
		asm ("" : : : "memory");
	}
	const long rv = thread->return_value;
	free((char *)thread + sizeof(Thread) - Thread::STACK_SIZE);
	return rv;
}
inline long join(Thread_ptr& tp)
{
	return join(tp.release());
}

template <typename Functor>
inline void yield_until(Functor&& condition)
{
	do {
		yield();
		asm("" ::: "memory");
	} while (!condition());
}
inline long yield()
{
	register long a0 asm("a0");
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+2;

	asm volatile ("scall" : "=r"(a0) : "r"(syscall_id) : "memory");

	return a0;
}
inline long yield_to(int tid)
{
	register long a0 asm("a0") = tid;
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+3;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id) : "memory");

	return a0;
}
inline long yield_to(Thread* thread)
{
	return yield_to(thread->tid);
}

inline long block(int reason)
{
	register long a0 asm("a0") = reason;
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+4;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id) : "memory");

	return a0;
}
template <typename Functor>
inline void block(Functor&& condition, int reason)
{
	while (!condition()) {
		if (block(reason) < 0) break;
	}
}
inline long wakeup_one_blocked(int reason)
{
	register long a0 asm("a0") = reason;
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+5;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id) : "memory");

	return a0;
}
inline long unblock(int tid)
{
	register long a0 asm("a0") = tid;
	register long syscall_id asm("a7") = THREAD_SYSCALLS_BASE+6;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id) : "memory");

	return a0;
}

__attribute__((noreturn))
inline void exit(long exitcode)
{
	self()->exit(exitcode);
	__builtin_unreachable();
}

__attribute__((noreturn))
inline void Thread::exit(long exitcode)
{
	this->tid = 0;
	this->return_value = exitcode;
	syscall(THREAD_SYSCALLS_BASE+1, exitcode);
	__builtin_unreachable();
}

}
