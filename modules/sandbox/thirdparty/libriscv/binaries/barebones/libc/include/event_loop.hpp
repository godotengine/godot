#include "function.hpp"
#include "ringbuffer.hpp"

template <size_t Capacity = 16>
struct Events {
	using Work = Function<void()>;

	FixedRingBuffer<Capacity, Work> ring;
	bool in_use = false;

	void consume_work();
	bool add(const Work&);
};

template <size_t Capacity>
inline void Events<Capacity>::consume_work()
{
	this->in_use = true;
	while (const auto* wrk = ring.read()) {
		(*wrk)();
	}
	this->in_use = false;
}

template <size_t Capacity>
inline bool Events<Capacity>::add(const Work& work) {
	if (in_use == false) {
		return ring.write(work);
	}
	return false;
}

/**
 * SharedEvents is an events structure designed to be
 * shared between the host and the script.
**/
template <typename Argument = void*, size_t Capacity = 16>
struct SharedEvents {
	using Callback = void(*)(Argument);
	struct Work {
		Callback callback;
		Argument argument;
	};

	FixedRingBuffer<Capacity, Work> ring;
	bool in_use = false;

	bool has_work() const noexcept { return !ring.empty(); }
	void consume_work();
	bool add(Callback cb, Argument arg);
	bool host_add(uintptr_t cb, uintptr_t arg);
};

template <typename Argument, size_t Capacity>
inline void SharedEvents<Argument, Capacity>::consume_work()
{
	this->in_use = true;
	try {
		while (const auto* wrk = ring.read()) {
			wrk->callback(wrk->argument);
		}
		this->in_use = false;
	} catch (...) {
		this->in_use = false;
		throw;
	}
}

template <typename Argument, size_t Capacity>
inline bool SharedEvents<Argument, Capacity>::add(Callback cb, Argument arg) {
	if (in_use == false) {
		return ring.write({cb, arg});
	}
	return false;
}

template <typename Argument, size_t Capacity>
inline bool SharedEvents<Argument, Capacity>::host_add(uintptr_t cb, uintptr_t arg) {
	if (in_use == false) {
		return ring.write({(Callback&)cb, (Argument)arg});
	}
	return false;
}
