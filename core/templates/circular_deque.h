/**************************************************************************/
/*  circular_deque.h                                                      */
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

#ifndef CIRCULAR_DEQUE_H
#define CIRCULAR_DEQUE_H

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/templates/vector.h"

#include <initializer_list>
#include <type_traits>

// Capacity can't be increased automatically.
// Instead, you should call "CircularDeque::reserve()" or "CircularDeque::adjust_capacity()" to grow the capacity explicitly.
// If "allow_override_edges" is false, the CircularDeque can't push new element if is full.
template <class T, class U = int32_t, bool force_trivial = false, bool allow_override_edges = true>
class CircularDeque {
private:
#define DEFAULT_CAPACITY 16
	U front_pos = -1;
	U rear_pos = -1;

	U count = 0;
	U capacity = 0;

	T *data = nullptr;

public:
	_FORCE_INLINE_ bool is_full() const {
		return (rear_pos == capacity - 1 && front_pos == 0) || (rear_pos == front_pos - 1);
	}

	_FORCE_INLINE_ U size() const { return count; }
	_FORCE_INLINE_ U get_capacity() const { return capacity; }

	_FORCE_INLINE_ bool is_empty() const {
		return front_pos == -1 || rear_pos == -1;
	}

	_FORCE_INLINE_ void push_back(const T &p_elem) {
		if constexpr (!allow_override_edges) {
			ERR_FAIL_COND(is_full());
			count++;
		} else {
			if (is_full()) {
				if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
					data[front_pos].~T();
				}

				if (front_pos == capacity - 1) {
					front_pos = 0;
				} else {
					front_pos++;
				}
			} else {
				count++;
			}
		}

		if (front_pos == -1) {
			rear_pos = front_pos = 0;
		} else if (rear_pos == capacity - 1) {
			rear_pos = 0;
		} else {
			rear_pos++;
		}

		if constexpr (!std::is_trivially_constructible<T>::value && !force_trivial) {
			memnew_placement(&data[rear_pos], T(p_elem));
		} else {
			data[rear_pos] = p_elem;
		}
	}

	_FORCE_INLINE_ void push_front(const T &p_elem) {
		if constexpr (!allow_override_edges) {
			ERR_FAIL_COND(is_full());
			count++;
		} else {
			if (is_full()) {
				if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
					data[rear_pos].~T();
				}

				if (rear_pos == 0) {
					rear_pos = capacity - 1;
				} else {
					rear_pos--;
				}
			} else {
				count++;
			}
		}

		if (rear_pos == -1) {
			front_pos = rear_pos = capacity - 1;
		} else if (front_pos == 0) {
			front_pos = capacity - 1;
		} else {
			front_pos--;
		}

		if constexpr (!std::is_trivially_constructible<T>::value && !force_trivial) {
			memnew_placement(&data[front_pos], T(p_elem));
		} else {
			data[front_pos] = p_elem;
		}
	}

	_FORCE_INLINE_ void pop_front() {
		if (is_empty()) {
			return;
		}

		if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
			data[front_pos].~T();
		}

		if (front_pos == rear_pos) {
			front_pos = -1;
			rear_pos = -1;
		} else if (front_pos == capacity - 1) {
			front_pos = 0;
		} else {
			front_pos++;
		}
		count--;
	}

	_FORCE_INLINE_ void pop_back() {
		if (is_empty()) {
			return;
		}

		if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
			data[rear_pos].~T();
		}

		if (front_pos == rear_pos) {
			front_pos = -1;
			rear_pos = -1;
		} else if (rear_pos == 0) {
			rear_pos = capacity - 1;
		} else {
			rear_pos--;
		}
		count--;
	}

	void remove_at(U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, size());
		p_index += front_pos;
		if (p_index >= capacity) {
			p_index -= capacity;
		}

		if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
			data[p_index].~T();
		}

		count--;

		if (front_pos == rear_pos) {
			front_pos = -1;
			rear_pos = -1;
		} else if (p_index > front_pos && (p_index - front_pos <= MAX(rear_pos - p_index, rear_pos + 1))) {
			for (; p_index > front_pos; --p_index) {
				data[p_index] = data[p_index - 1];
			}

			if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
				data[front_pos].~T();
			}

			if (front_pos == capacity - 1) {
				front_pos = 0;
			} else {
				front_pos++;
			}
		} else {
			for (; p_index < rear_pos; ++p_index) {
				data[p_index] = data[p_index + 1];
			}

			if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
				data[rear_pos].~T();
			}

			if (rear_pos == 0) {
				rear_pos = capacity - 1;
			} else {
				rear_pos--;
			}
		}
	}

	void insert(U p_index, const T &p_elem, bool p_pop_front_if_full = true) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, size() + 1);
		if (p_index == 0) {
			push_front(p_elem);
			return;
		} else if (p_index == count - 1) {
			push_back(p_elem);
			return;
		}

		if constexpr (!allow_override_edges) {
			ERR_FAIL_COND(is_full());
			count++;
		} else {
			if (is_full()) {
				if (p_pop_front_if_full) {
					data[front_pos].~T();

					if (front_pos == capacity - 1) {
						front_pos = 0;
					} else {
						front_pos++;
					}
				} else {
					data[rear_pos].~T();

					if (rear_pos == 0) {
						rear_pos = capacity - 1;
					} else {
						rear_pos--;
					}
				}
			} else {
				count++;
			}
		}

		p_index += front_pos;
		if (p_index >= capacity) {
			p_index -= capacity;
		}

		if (p_index > front_pos && (p_index - front_pos <= MAX(rear_pos - p_index, rear_pos + 1))) {
			p_index--;
			if (front_pos == 0) {
				data[capacity - 1] = data[0];
				front_pos = capacity - 1;
			} else {
				front_pos--;
			}

			for (U i = (front_pos == capacity - 1) ? 0 : front_pos; i < p_index; ++i) {
				data[i] = data[i + 1];
			}
		} else {
			p_index++;
			if (rear_pos == capacity - 1) {
				data[0] = data[rear_pos];
				rear_pos = 0;
			} else {
				rear_pos++;
			}

			for (U i = (rear_pos == 0) ? capacity - 1 : rear_pos; i > p_index; --i) {
				data[i] = data[i - 1];
			}
		}

		if constexpr (!std::is_trivially_constructible<T>::value && !force_trivial) {
			memnew_placement(&data[p_index], T(p_elem));
		} else {
			data[p_index] = p_elem;
		}
	}

	int64_t find(const T &p_val, int p_from = 0) const {
		CRASH_BAD_UNSIGNED_INDEX(p_from, size());
		if (rear_pos >= front_pos) {
			for (U i = p_from + front_pos; i <= rear_pos; ++i) {
				if (data[i] == p_val) {
					return int64_t(i - front_pos);
				}
			}
		} else {
			p_from += front_pos;
			if (p_from < capacity) {
				for (U i = p_from; i < capacity; ++i) {
					if (data[i] == p_val) {
						return int64_t(i - front_pos);
					}
				}
				p_from = 0;
			} else {
				p_from -= capacity;
			}

			for (U i = p_from; i <= rear_pos; ++i) {
				if (data[i] == p_val) {
					return int64_t(count - 1 - (rear_pos - i));
				}
			}
		}
		return -1;
	}

	int64_t rfind(const T &p_val, int p_from = -1) const {
		p_from += capacity;
		CRASH_BAD_UNSIGNED_INDEX(p_from, size());
		if (rear_pos >= front_pos) {
			for (U i = front_pos + front_pos; i >= front_pos; --i) {
				if (data[i] == p_val) {
					return int64_t(i - front_pos);
				}
			}
		} else {
			p_from = rear_pos - (count - 1 - p_from);
			if (p_from >= 0) {
				for (U i = p_from; i >= 0; --i) {
					if (data[i] == p_val) {
						return int64_t(count - 1 - (rear_pos - i));
					}
				}
				p_from = capacity - 1;
			} else {
				p_from += capacity;
			}

			for (U i = p_from; i >= front_pos; --i) {
				if (data[i] == p_val) {
					return int64_t(i - front_pos);
				}
			}
		}
		return -1;
	}

	_FORCE_INLINE_ bool erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) {
			remove_at(idx);
			return true;
		}
		return false;
	}

	_FORCE_INLINE_ void clear() { resize(0); }

	_FORCE_INLINE_ void reset() {
		clear();
		if (data) {
			memfree(data);
			data = nullptr;
			capacity = 0;
			front_pos = -1;
			rear_pos = -1;
		}
	}

	void resize(U p_size) {
		CRASH_COND(p_size > capacity);
		if (p_size < size()) {
			if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
				for (U i = size() - p_size; i > 0; i--) {
					data[rear_pos].~T();
					if (rear_pos == 0) {
						rear_pos = capacity - 1;
					}
				}
			} else {
				rear_pos -= (size() - p_size);
				if (rear_pos < 0) {
					rear_pos += capacity;
				}
			}
		} else if (p_size > size()) {
			for (U i = p_size - size(); i > 0; --i) {
				if (front_pos == -1) {
					rear_pos = front_pos = 0;
				} else if (rear_pos == capacity - 1) {
					rear_pos = 0;
				} else {
					rear_pos++;
				}

				if constexpr (!std::is_trivially_constructible<T>::value && !force_trivial) {
					memnew_placement(&data[rear_pos], T);
				} else {
					data[rear_pos] = {};
				}
			}
		}

		count = p_size;
		if (count == 0) {
			rear_pos = -1;
			front_pos = -1;
		}
	}

	_FORCE_INLINE_ void reserve(U p_capacity) {
		if (p_capacity > get_capacity()) {
			data = (T *)memrealloc(data, p_capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");

			if (front_pos > rear_pos) {
				U move_count = capacity - 1 - front_pos;
				for (U i = 0; i < move_count; ++i) {
					data[p_capacity - i] = data[capacity - i];
				}

				if constexpr (!std::is_trivially_destructible<T>::value && !force_trivial) {
					for (U i = p_capacity - get_capacity() - 1; i >= 0; --i) {
						data[rear_pos + i].~T();
					}
				}

				rear_pos += p_capacity - get_capacity();
			}
			capacity = p_capacity;
		}
	}

	void adjust_capacity(U p_capacity) {
		if (p_capacity > get_capacity()) {
			reserve(p_capacity);
		} else if (p_capacity < get_capacity()) {
			// Resize.
			if (p_capacity < size()) {
				for (U i = size() - p_capacity; i > 0; ++i) {
					pop_back();
				}
			}

			// Move elements if need.
			if (rear_pos > p_capacity) {
				for (U i = 0; i < count; ++i) {
					data[i] = data[front_pos + i];
				}
				front_pos = 0;
				rear_pos = count - 1;
			} else if (front_pos > rear_pos) {
				U move_count = capacity - rear_pos;
				for (U i = 0; i < move_count; ++i) {
					data[front_pos + i] = data[rear_pos + i];
				}
				rear_pos = front_pos + 1;
			}

			// Realloc.
			data = (T *)memrealloc(data, p_capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");

			capacity = p_capacity;
		}
	}

	_FORCE_INLINE_ const T &operator[](U p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, size());
		if (front_pos + p_index >= capacity) {
			return data[front_pos + p_index - capacity];
		} else {
			return data[front_pos + p_index];
		}
	}

	_FORCE_INLINE_ T &operator[](U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, size());
		if (front_pos + p_index >= capacity) {
			return data[front_pos + p_index - capacity];
		} else {
			return data[front_pos + p_index];
		}
	}

	operator Vector<T>() const {
		Vector<T> ret;
		ret.resize(size());
		T *w = ret.ptrw();
		if (rear_pos >= front_pos) {
			memcpy(w, data + front_pos, sizeof(T) * size());
		} else {
			memcpy(w, data + front_pos, sizeof(T) * (capacity - front_pos));
			memcpy(w, data, sizeof(T) * (rear_pos + 1));
		}
		return ret;
	}

	_FORCE_INLINE_ CircularDeque(U p_capacity = DEFAULT_CAPACITY) {
		reserve(p_capacity);
	}
	_FORCE_INLINE_ CircularDeque(std::initializer_list<T> p_init) {
		reserve(p_init.size());
		for (const T &element : p_init) {
			push_back(element);
		}
	}
	_FORCE_INLINE_ CircularDeque(const CircularDeque &p_from) {
		reserve(p_from.get_capacity());

		if (p_from.is_empty()) {
			return;
		}

		U from_size = p_from.size();

		resize(from_size);
		for (U i = 0; i < from_size; i++) {
			data[i] = p_from[i];
		}
		front_pos = 0;
		rear_pos = from_size - 1;
	}

	inline void operator=(const CircularDeque &p_from) {
		adjust_capacity(p_from.get_capacity());

		if (p_from.is_empty()) {
			return;
		}

		U from_size = p_from.size();

		resize(from_size);
		for (U i = 0; i < from_size; i++) {
			data[i] = p_from[i];
		}
		front_pos = 0;
		rear_pos = from_size - 1;
	}

	inline void operator=(const Vector<T> &p_from) {
		reserve(p_from.get_capacity());

		if (p_from.is_empty()) {
			return;
		}

		U from_size = p_from.size();

		resize(from_size);
		for (U i = 0; i < from_size; i++) {
			data[i] = p_from[i];
		}
		front_pos = 0;
		rear_pos = from_size - 1;
	}

	_FORCE_INLINE_ ~CircularDeque() {
		if (data) {
			reset();
		}
	}
};

#endif // CIRCULAR_DEQUE_H
