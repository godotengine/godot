#ifndef UTIL_RINGBUFFER_HPP
#define UTIL_RINGBUFFER_HPP

#include <cassert>
#include <cstring>
#include <cstdint>

template <size_t N, typename T>
class FixedRingBuffer {
public:
	bool write(T item)
	{
		if (!full()) {
			buffer_at(this->m_writer ++) = std::move(item);
			return true;
		}
		return false;
	}

	const T* read()
	{
		if (!empty()) {
			return &buffer_at(this->m_reader ++);
		}
		return nullptr;
	}

	bool discard()
	{
		if (!this->empty()) {
			this->m_reader ++;
			return true;
		}
		return false;
	}
	void clear() {
		this->m_reader = this->m_writer = 0;
	}

	size_t size() const {
		return this->m_writer - this->m_reader;
	}
	constexpr size_t capacity() const {
		return N;
	}

	bool full() const {
		return size() == capacity();
	}
	bool empty() const {
		return size() == 0;
	}

	const T* data() const {
		return this->m_buffer;
	}

private:
	T& buffer_at(size_t idx) {
		return this->m_buffer[idx % capacity()];
	}

	size_t m_reader = 0;
	size_t m_writer = 0;
	T m_buffer[N];
};

#endif
