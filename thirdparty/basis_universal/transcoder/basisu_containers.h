// basisu_containers.h
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <algorithm>

#if defined(__linux__) && !defined(ANDROID)
// Only for malloc_usable_size() in basisu_containers_impl.h
#include <malloc.h>
#define HAS_MALLOC_USABLE_SIZE 1
#endif

// Set to 1 to always check vector operator[], front(), and back() even in release.
#define BASISU_VECTOR_FORCE_CHECKING 0

// If 1, the vector container will not query the CRT to get the size of resized memory blocks.
#define BASISU_VECTOR_DETERMINISTIC 1

#ifdef _MSC_VER
#define BASISU_FORCE_INLINE __forceinline
#else
#define BASISU_FORCE_INLINE inline
#endif

#define BASISU_HASHMAP_TEST 0

namespace basisu
{
	enum { cInvalidIndex = -1 };

	template <typename S> inline S clamp(S value, S low, S high) { return (value < low) ? low : ((value > high) ? high : value); }

	template <typename S> inline S maximum(S a, S b) { return (a > b) ? a : b; }
	template <typename S> inline S maximum(S a, S b, S c) { return maximum(maximum(a, b), c); }
	template <typename S> inline S maximum(S a, S b, S c, S d) { return maximum(maximum(maximum(a, b), c), d); }

	template <typename S> inline S minimum(S a, S b) { return (a < b) ? a : b; }
	template <typename S> inline S minimum(S a, S b, S c) { return minimum(minimum(a, b), c); }
	template <typename S> inline S minimum(S a, S b, S c, S d) { return minimum(minimum(minimum(a, b), c), d); }

#ifdef _MSC_VER
	__declspec(noreturn)
#else
	[[noreturn]]
#endif
	void container_abort(const char* pMsg, ...);

	namespace helpers
	{
		inline bool is_power_of_2(uint32_t x) { return x && ((x & (x - 1U)) == 0U); }
		inline bool is_power_of_2(uint64_t x) { return x && ((x & (x - 1U)) == 0U); }

		template<class T> const T& minimum(const T& a, const T& b) { return (b < a) ? b : a; }
		template<class T> const T& maximum(const T& a, const T& b) { return (a < b) ? b : a; }

		inline uint32_t floor_log2i(uint32_t v)
		{
			uint32_t l = 0;
			while (v > 1U)
			{
				v >>= 1;
				l++;
			}
			return l;
		}

		inline uint32_t floor_log2i(uint64_t v)
		{
			uint32_t l = 0;
			while (v > 1U)
			{
				v >>= 1;
				l++;
			}
			return l;
		}

		inline uint32_t next_pow2(uint32_t val)
		{
			val--;
			val |= val >> 16;
			val |= val >> 8;
			val |= val >> 4;
			val |= val >> 2;
			val |= val >> 1;
			return val + 1;
		}

		inline uint64_t next_pow2(uint64_t val)
		{
			val--;
			val |= val >> 32;
			val |= val >> 16;
			val |= val >> 8;
			val |= val >> 4;
			val |= val >> 2;
			val |= val >> 1;
			return val + 1;
		}
	} // namespace helpers

	template <typename T>
	inline T* construct(T* p)
	{
		return new (static_cast<void*>(p)) T;
	}

	template <typename T, typename U>
	inline T* construct(T* p, const U& init)
	{
		return new (static_cast<void*>(p)) T(init);
	}

	template <typename T>
	inline void construct_array(T* p, size_t n)
	{
		T* q = p + n;
		for (; p != q; ++p)
			new (static_cast<void*>(p)) T;
	}

	template <typename T, typename U>
	inline void construct_array(T* p, size_t n, const U& init)
	{
		T* q = p + n;
		for (; p != q; ++p)
			new (static_cast<void*>(p)) T(init);
	}

	template <typename T>
	inline void destruct(T* p)
	{
		p->~T();
	}

	template <typename T> inline void destruct_array(T* p, size_t n)
	{
		T* q = p + n;
		for (; p != q; ++p)
			p->~T();
	}

	template<typename T>
	struct scalar_type
	{
		enum { cFlag = false };
		static inline void construct(T* p) { basisu::construct(p); }
		static inline void construct(T* p, const T& init) { basisu::construct(p, init); }
		static inline void construct_array(T* p, size_t n) { basisu::construct_array(p, n); }
		static inline void destruct(T* p) { basisu::destruct(p); }
		static inline void destruct_array(T* p, size_t n) { basisu::destruct_array(p, n); }
	};

	template<typename T> struct scalar_type<T*>
	{
		enum { cFlag = true };
		static inline void construct(T** p) { memset(p, 0, sizeof(T*)); }
		static inline void construct(T** p, T* init) { *p = init; }
		static inline void construct_array(T** p, size_t n) { memset(p, 0, sizeof(T*) * n); }
		static inline void destruct(T** p) { p; }
		static inline void destruct_array(T** p, size_t n) { p, n; }
	};

#define BASISU_DEFINE_BUILT_IN_TYPE(X) \
	template<> struct scalar_type<X> { \
	enum { cFlag = true }; \
	static inline void construct(X* p) { memset(p, 0, sizeof(X)); } \
	static inline void construct(X* p, const X& init) { memcpy(p, &init, sizeof(X)); } \
	static inline void construct_array(X* p, size_t n) { memset(p, 0, sizeof(X) * n); } \
	static inline void destruct(X* p) { p; } \
	static inline void destruct_array(X* p, size_t n) { p, n; } };

	BASISU_DEFINE_BUILT_IN_TYPE(bool)
	BASISU_DEFINE_BUILT_IN_TYPE(char)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned char)
	BASISU_DEFINE_BUILT_IN_TYPE(short)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned short)
	BASISU_DEFINE_BUILT_IN_TYPE(int)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned int)
	BASISU_DEFINE_BUILT_IN_TYPE(long)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned long)
#ifdef __GNUC__
	BASISU_DEFINE_BUILT_IN_TYPE(long long)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned long long)
#else
	BASISU_DEFINE_BUILT_IN_TYPE(__int64)
	BASISU_DEFINE_BUILT_IN_TYPE(unsigned __int64)
#endif
	BASISU_DEFINE_BUILT_IN_TYPE(float)
	BASISU_DEFINE_BUILT_IN_TYPE(double)
	BASISU_DEFINE_BUILT_IN_TYPE(long double)

#undef BASISU_DEFINE_BUILT_IN_TYPE

	template<typename T>
	struct bitwise_movable { enum { cFlag = false }; };

#define BASISU_DEFINE_BITWISE_MOVABLE(Q) template<> struct bitwise_movable<Q> { enum { cFlag = true }; };

	template<typename T>
	struct bitwise_copyable { enum { cFlag = false }; };

#define BASISU_DEFINE_BITWISE_COPYABLE(Q) template<> struct bitwise_copyable<Q> { enum { cFlag = true }; };

#define BASISU_IS_POD(T) __is_pod(T)

#define BASISU_IS_SCALAR_TYPE(T) (scalar_type<T>::cFlag)

#if !defined(BASISU_HAVE_STD_TRIVIALLY_COPYABLE) && defined(__GNUC__) && (__GNUC__ < 5)
#define BASISU_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#else
#define BASISU_IS_TRIVIALLY_COPYABLE(...) std::is_trivially_copyable<__VA_ARGS__>::value
#endif

	// TODO: clean this up, it's still confusing (copying vs. movable).
#define BASISU_IS_BITWISE_COPYABLE(T) (BASISU_IS_SCALAR_TYPE(T) || BASISU_IS_POD(T) || BASISU_IS_TRIVIALLY_COPYABLE(T) || std::is_trivial<T>::value || (bitwise_copyable<T>::cFlag))

#define BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(T) (BASISU_IS_BITWISE_COPYABLE(T) || (bitwise_movable<T>::cFlag))

#define BASISU_HAS_DESTRUCTOR(T) ((!scalar_type<T>::cFlag) && (!__is_pod(T)) && (!std::is_trivially_destructible<T>::value))

	typedef char(&yes_t)[1];
	typedef char(&no_t)[2];

	template <class U> yes_t class_test(int U::*);
	template <class U> no_t class_test(...);

	template <class T> struct is_class
	{
		enum { value = (sizeof(class_test<T>(0)) == sizeof(yes_t)) };
	};

	template <typename T> struct is_pointer
	{
		enum { value = false };
	};

	template <typename T> struct is_pointer<T*>
	{
		enum { value = true };
	};

	struct empty_type { };

	BASISU_DEFINE_BITWISE_COPYABLE(empty_type);
	BASISU_DEFINE_BITWISE_MOVABLE(empty_type);

	template<typename T> struct rel_ops
	{
		friend bool operator!=(const T& x, const T& y) { return (!(x == y)); }
		friend bool operator> (const T& x, const T& y) { return (y < x); }
		friend bool operator<=(const T& x, const T& y) { return (!(y < x)); }
		friend bool operator>=(const T& x, const T& y) { return (!(x < y)); }
	};

	struct elemental_vector
	{
		void* m_p;
		size_t m_size;
		size_t m_capacity;

		typedef void (*object_mover)(void* pDst, void* pSrc, size_t num);

		bool increase_capacity(size_t min_new_capacity, bool grow_hint, size_t element_size, object_mover pRelocate, bool nofail);
	};

	// Returns true if a+b would overflow a size_t.
	inline bool add_overflow_check(size_t a, size_t b)
	{
		size_t c = a + b;
		return c < a;
	}
		
	// Returns false on overflow, true if OK.
	template<typename T>
	inline bool can_fit_into_size_t(T val)
	{
		static_assert(std::is_integral<T>::value, "T must be an integral type");

		return (val >= 0) && (static_cast<size_t>(val) == val);
	}

	// Returns true if a*b would overflow a size_t.
	inline bool mul_overflow_check(size_t a, size_t b)
	{
		// Avoid the division on 32-bit platforms
		if (sizeof(size_t) == sizeof(uint32_t))
			return !can_fit_into_size_t(static_cast<uint64_t>(a) * b);
		else
			return b && (a > (SIZE_MAX / b));
	}

	template<typename T>
	class writable_span;
		
	template<typename T>
	class readable_span
	{
	public:
		using value_type = T;
		using size_type = size_t;
		using const_pointer = const T*;
		using const_reference = const T&;
		using const_iterator = const T*;
		
		inline readable_span() :
			m_p(nullptr),
			m_size(0)
		{
		}

		inline readable_span(const writable_span<T>& other);
		inline readable_span& operator= (const writable_span<T>& rhs);

		inline readable_span(const_pointer p, size_t n)
		{
			set(p, n);
		}

		inline readable_span(const_pointer s, const_pointer e)
		{
			set(s, e);
		}

		inline readable_span(const readable_span& other) :
			m_p(other.m_p),
			m_size(other.m_size)
		{
			assert(!m_size || m_p);
		}

		inline readable_span(readable_span&& other) :
			m_p(other.m_p),
			m_size(other.m_size)
		{
			assert(!m_size || m_p);

			other.m_p = nullptr;
			other.m_size = 0;
		}

		template <size_t N>
		inline readable_span(const T(&arr)[N]) :
			m_p(arr),
			m_size(N)
		{
		}

		template <size_t N>
		inline readable_span& set(const T(&arr)[N])
		{
			m_p = arr;
			m_size = N;
			return *this;
		}

		inline readable_span& set(const_pointer p, size_t n)
		{
			if (!p && n)
			{
				assert(0);
				m_p = nullptr;
				m_size = 0;
			}
			else
			{
				m_p = p;
				m_size = n;
			}

			return *this;
		}

		inline readable_span& set(const_pointer s, const_pointer e)
		{
			if ((e < s) || (!s && e))
			{
				assert(0);
				m_p = nullptr;
				m_size = 0;
			}
			else
			{
				m_p = s;
				m_size = e - s;
			}

			return *this;
		}

		inline bool operator== (const readable_span& rhs) const
		{
			return (m_p == rhs.m_p) && (m_size == rhs.m_size);
		}

		inline bool operator!= (const readable_span& rhs) const
		{
			return (m_p != rhs.m_p) || (m_size != rhs.m_size);
		}

		// only true if the region is totally inside the span
		inline bool is_inside_ptr(const_pointer p, size_t n) const
		{
			if (!is_valid())
			{
				assert(0);
				return false;
			}

			if (!p)
			{
				assert(!n);
				return false;
			}

			return (p >= m_p) && ((p + n) <= end());
		}

		inline bool is_inside(size_t ofs, size_t size) const
		{
			if (add_overflow_check(ofs, size))
			{
				assert(0);
				return false;
			}

			if (!is_valid())
			{
				assert(0);
				return false;
			}

			if ((ofs + size) > m_size)
				return false;

			return true;
		}

		inline readable_span subspan(size_t ofs, size_t n) const
		{
			if (!is_valid())
			{
				assert(0);
				return readable_span((const_pointer)nullptr, (size_t)0);
			}

			if (add_overflow_check(ofs, n))
			{
				assert(0);
				return readable_span((const_pointer)nullptr, (size_t)0);
			}

			if ((ofs + n) > m_size)
			{
				assert(0);
				return readable_span((const_pointer)nullptr, (size_t)0);
			}

			return readable_span(m_p + ofs, n);
		}

		void clear()
		{
			m_p = nullptr;
			m_size = 0;
		}

		inline bool empty() const { return !m_size; }

		// true if the span is non-nullptr and is not empty
		inline bool is_valid() const { return m_p && m_size; }

		inline bool is_nullptr() const { return m_p == nullptr; }

		inline size_t size() const { return m_size; }
		inline size_t size_in_bytes() const { assert(can_fit_into_size_t((uint64_t)m_size * sizeof(T))); return m_size * sizeof(T); }

		inline const_pointer get_ptr() const { return m_p; }

		inline const_iterator begin() const { return m_p; }
		inline const_iterator end() const { assert(m_p || !m_size); return m_p + m_size; }

		inline const_iterator cbegin() const { return m_p; }
		inline const_iterator cend() const { assert(m_p || !m_size); return m_p + m_size; }

		inline const_reference front() const
		{
			if (!(m_p && m_size))
				container_abort("readable_span invalid\n");

			return m_p[0];
		}

		inline const_reference back() const
		{
			if (!(m_p && m_size))
				container_abort("readable_span invalid\n");

			return m_p[m_size - 1];
		}

		inline readable_span& operator= (const readable_span& rhs)
		{
			m_p = rhs.m_p;
			m_size = rhs.m_size;
			return *this;
		}

		inline readable_span& operator= (readable_span&& rhs)
		{
			if (this != &rhs)
			{
				m_p = rhs.m_p;
				m_size = rhs.m_size;
				rhs.m_p = nullptr;
				rhs.m_size = 0;
			}

			return *this;
		}

		inline const_reference operator* () const
		{
			if (!(m_p && m_size))
				container_abort("readable_span invalid\n");

			return *m_p;
		}

		inline const_pointer operator-> () const
		{
			if (!(m_p && m_size))
				container_abort("readable_span invalid\n");

			return m_p;
		}

		inline readable_span& remove_prefix(size_t n)
		{
			if ((!m_p) || (n > m_size))
			{
				assert(0);
				return *this;
			}

			m_p += n;
			m_size -= n;
			return *this;
		}

		inline readable_span& remove_suffix(size_t n)
		{
			if ((!m_p) || (n > m_size))
			{
				assert(0);
				return *this;
			}

			m_size -= n;
			return *this;
		}

		inline readable_span& enlarge(size_t n)
		{
			if (!m_p)
			{
				assert(0);
				return *this;
			}

			if (add_overflow_check(m_size, n))
			{
				assert(0);
				return *this;
			}

			m_size += n;
			return *this;
		}

		bool copy_from(size_t src_ofs, size_t src_size, T* pDst, size_t dst_ofs) const
		{
			if (!src_size)
				return true;

			if (!pDst)
			{
				assert(0);
				return false;
			}

			if (!is_inside(src_ofs, src_size))
			{
				assert(0);
				return false;
			}

			const_pointer pS = m_p + src_ofs;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				const uint64_t num_bytes = (uint64_t)src_size * sizeof(T);

				if (!can_fit_into_size_t(num_bytes))
				{
					assert(0);
					return false;
				}

				memcpy(pDst, pS, (size_t)num_bytes);
			}
			else
			{
				T* pD = pDst + dst_ofs;
				T* pDst_end = pD + src_size;

				while (pD != pDst_end)
					*pD++ = *pS++;
			}

			return true;
		}

		inline const_reference operator[] (size_t idx) const
		{
			if ((!is_valid()) || (idx >= m_size))
				container_abort("readable_span: invalid span or index\n");

			return m_p[idx];
		}

		inline uint16_t read_le16(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return false;
			}

			const uint8_t a = (uint8_t)m_p[ofs];
			const uint8_t b = (uint8_t)m_p[ofs + 1];
			return a | (b << 8u);
		}

		template<typename R>
		inline R read_val(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(R)))
			{
				assert(0);
				return (R)0;
			}

			return *reinterpret_cast<const R*>(&m_p[ofs]);
		}

		inline uint16_t read_be16(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t b = (uint8_t)m_p[ofs];
			const uint8_t a = (uint8_t)m_p[ofs + 1];
			return a | (b << 8u);
		}

		inline uint32_t read_le32(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t a = (uint8_t)m_p[ofs];
			const uint8_t b = (uint8_t)m_p[ofs + 1];
			const uint8_t c = (uint8_t)m_p[ofs + 2];
			const uint8_t d = (uint8_t)m_p[ofs + 3];
			return a | (b << 8u) | (c << 16u) | (d << 24u);
		}

		inline uint32_t read_be32(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t d = (uint8_t)m_p[ofs];
			const uint8_t c = (uint8_t)m_p[ofs + 1];
			const uint8_t b = (uint8_t)m_p[ofs + 2];
			const uint8_t a = (uint8_t)m_p[ofs + 3];
			return a | (b << 8u) | (c << 16u) | (d << 24u);
		}

		inline uint64_t read_le64(size_t ofs) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return 0;
			}
			const uint64_t l = read_le32(ofs);
			const uint64_t h = read_le32(ofs + sizeof(uint32_t));
			return l | (h << 32u);
		}

		inline uint64_t read_be64(size_t ofs) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return 0;
			}
			const uint64_t h = read_be32(ofs);
			const uint64_t l = read_be32(ofs + sizeof(uint32_t));
			return l | (h << 32u);
		}

	private:
		const_pointer m_p;
		size_t m_size;
	};

	template<typename T>
	class writable_span
	{
		friend readable_span<T>;

	public:
		using value_type = T;
		using size_type = size_t;
		using const_pointer = const T*;
		using const_reference = const T&;
		using const_iterator = const T*;
		using pointer = T*;
		using reference = T&;
		using iterator = T*;

		inline writable_span() :
			m_p(nullptr),
			m_size(0)
		{
		}

		inline writable_span(T* p, size_t n)
		{
			set(p, n);
		}

		inline writable_span(T* s, T* e)
		{
			set(s, e);
		}

		inline writable_span(const writable_span& other) :
			m_p(other.m_p),
			m_size(other.m_size)
		{
			assert(!m_size || m_p);
		}

		inline writable_span(writable_span&& other) :
			m_p(other.m_p),
			m_size(other.m_size)
		{
			assert(!m_size || m_p);

			other.m_p = nullptr;
			other.m_size = 0;
		}

		template <size_t N>
		inline writable_span(T(&arr)[N]) :
			m_p(arr),
			m_size(N)
		{
		}

		readable_span<T> get_readable_span() const
		{
			return readable_span<T>(m_p, m_size);
		}

		template <size_t N>
		inline writable_span& set(T(&arr)[N])
		{
			m_p = arr;
			m_size = N;
			return *this;
		}

		inline writable_span& set(T* p, size_t n)
		{
			if (!p && n)
			{
				assert(0);
				m_p = nullptr;
				m_size = 0;
			}
			else
			{
				m_p = p;
				m_size = n;
			}

			return *this;
		}

		inline writable_span& set(T* s, T* e)
		{
			if ((e < s) || (!s && e))
			{
				assert(0);
				m_p = nullptr;
				m_size = 0;
			}
			else
			{
				m_p = s;
				m_size = e - s;
			}

			return *this;
		}

		inline bool operator== (const writable_span& rhs) const
		{
			return (m_p == rhs.m_p) && (m_size == rhs.m_size);
		}

		inline bool operator== (const readable_span<T>& rhs) const
		{
			return (m_p == rhs.m_p) && (m_size == rhs.m_size);
		}

		inline bool operator!= (const writable_span& rhs) const
		{
			return (m_p != rhs.m_p) || (m_size != rhs.m_size);
		}

		inline bool operator!= (const readable_span<T>& rhs) const
		{
			return (m_p != rhs.m_p) || (m_size != rhs.m_size);
		}

		// only true if the region is totally inside the span
		inline bool is_inside_ptr(const_pointer p, size_t n) const
		{
			if (!is_valid())
			{
				assert(0);
				return false;
			}

			if (!p)
			{
				assert(!n);
				return false;
			}

			return (p >= m_p) && ((p + n) <= end());
		}

		inline bool is_inside(size_t ofs, size_t size) const
		{
			if (add_overflow_check(ofs, size))
			{
				assert(0);
				return false;
			}

			if (!is_valid())
			{
				assert(0);
				return false;
			}

			if ((ofs + size) > m_size)
				return false;

			return true;
		}

		inline writable_span subspan(size_t ofs, size_t n) const
		{
			if (!is_valid())
			{
				assert(0);
				return writable_span((T*)nullptr, (size_t)0);
			}

			if (add_overflow_check(ofs, n))
			{
				assert(0);
				return writable_span((T*)nullptr, (size_t)0);
			}

			if ((ofs + n) > m_size)
			{
				assert(0);
				return writable_span((T*)nullptr, (size_t)0);
			}

			return writable_span(m_p + ofs, n);
		}

		void clear()
		{
			m_p = nullptr;
			m_size = 0;
		}

		inline bool empty() const { return !m_size; }

		// true if the span is non-nullptr and is not empty
		inline bool is_valid() const { return m_p && m_size; }

		inline bool is_nullptr() const { return m_p == nullptr; }

		inline size_t size() const { return m_size; }
		inline size_t size_in_bytes() const { assert(can_fit_into_size_t((uint64_t)m_size * sizeof(T))); return m_size * sizeof(T); }

		inline T* get_ptr() const { return m_p; }

		inline iterator begin() const { return m_p; }
		inline iterator end() const { assert(m_p || !m_size); return m_p + m_size; }
		
		inline const_iterator cbegin() const { return m_p; }
		inline const_iterator cend() const { assert(m_p || !m_size); return m_p + m_size; }

		inline T& front() const
		{
			if (!(m_p && m_size))
				container_abort("writable_span invalid\n");

			return m_p[0];
		}

		inline T& back() const
		{
			if (!(m_p && m_size))
				container_abort("writable_span invalid\n");

			return m_p[m_size - 1];
		}

		inline writable_span& operator= (const writable_span& rhs)
		{
			m_p = rhs.m_p;
			m_size = rhs.m_size;
			return *this;
		}

		inline writable_span& operator= (writable_span&& rhs)
		{
			if (this != &rhs)
			{
				m_p = rhs.m_p;
				m_size = rhs.m_size;
				rhs.m_p = nullptr;
				rhs.m_size = 0;
			}

			return *this;
		}

		inline T& operator* () const
		{
			if (!(m_p && m_size))
				container_abort("writable_span invalid\n");

			return *m_p;
		}

		inline T* operator-> () const
		{
			if (!(m_p && m_size))
				container_abort("writable_span invalid\n");

			return m_p;
		}

		inline bool set_all(size_t ofs, size_t size, const_reference val)
		{
			if (!size)
				return true;

			if (!is_inside(ofs, size))
			{
				assert(0);
				return false;
			}

			T* pDst = m_p + ofs;

			if ((sizeof(T) == sizeof(uint8_t)) && (BASISU_IS_BITWISE_COPYABLE(T)))
			{
				memset(pDst, (int)((uint8_t)val), size);
			}
			else
			{

				T* pDst_end = pDst + size;

				while (pDst != pDst_end)
					*pDst++ = val;
			}

			return true;
		}

		inline bool set_all(const_reference val)
		{
			return set_all(0, m_size, val);
		}

		inline writable_span& remove_prefix(size_t n)
		{
			if ((!m_p) || (n > m_size))
			{
				assert(0);
				return *this;
			}

			m_p += n;
			m_size -= n;
			return *this;
		}

		inline writable_span& remove_suffix(size_t n)
		{
			if ((!m_p) || (n > m_size))
			{
				assert(0);
				return *this;
			}

			m_size -= n;
			return *this;
		}

		inline writable_span& enlarge(size_t n)
		{
			if (!m_p)
			{
				assert(0);
				return *this;
			}

			if (add_overflow_check(m_size, n))
			{
				assert(0);
				return *this;
			}

			m_size += n;
			return *this;
		}

		// copy from this span to the destination ptr
		bool copy_from(size_t src_ofs, size_t src_size, T* pDst, size_t dst_ofs) const
		{
			if (!src_size)
				return true;

			if (!pDst)
			{
				assert(0);
				return false;
			}

			if (!is_inside(src_ofs, src_size))
			{
				assert(0);
				return false;
			}

			const_pointer pS = m_p + src_ofs;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				const uint64_t num_bytes = (uint64_t)src_size * sizeof(T);

				if (!can_fit_into_size_t(num_bytes))
				{
					assert(0);
					return false;
				}

				memcpy(pDst, pS, (size_t)num_bytes);
			}
			else
			{
				T* pD = pDst + dst_ofs;
				T* pDst_end = pD + src_size;

				while (pD != pDst_end)
					*pD++ = *pS++;
			}

			return true;
		}

		// copy from the source ptr into this span
		bool copy_into(const_pointer pSrc, size_t src_ofs, size_t src_size, size_t dst_ofs) const
		{
			if (!src_size)
				return true;

			if (!pSrc)
			{
				assert(0);
				return false;
			}

			if (add_overflow_check(src_ofs, src_size) || add_overflow_check(dst_ofs, src_size))
			{
				assert(0);
				return false;
			}

			if (!is_valid())
			{
				assert(0);
				return false;
			}

			if (!is_inside(dst_ofs, src_size))
			{
				assert(0);
				return false;
			}

			const_pointer pS = pSrc + src_ofs;
			T* pD = m_p + dst_ofs;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				const uint64_t num_bytes = (uint64_t)src_size * sizeof(T);

				if (!can_fit_into_size_t(num_bytes))
				{
					assert(0);
					return false;
				}

				memcpy(pD, pS, (size_t)num_bytes);
			}
			else
			{
				T* pDst_end = pD + src_size;

				while (pD != pDst_end)
					*pD++ = *pS++;
			}

			return true;
		}

		// copy from a source span into this span
		bool copy_into(const readable_span<T>& src, size_t src_ofs, size_t src_size, size_t dst_ofs) const
		{
			if (!src.is_inside(src_ofs, src_size))
			{
				assert(0);
				return false;
			}

			return copy_into(src.get_ptr(), src_ofs, src_size, dst_ofs);
		}

		// copy from a source span into this span
		bool copy_into(const writable_span& src, size_t src_ofs, size_t src_size, size_t dst_ofs) const
		{
			if (!src.is_inside(src_ofs, src_size))
			{
				assert(0);
				return false;
			}

			return copy_into(src.get_ptr(), src_ofs, src_size, dst_ofs);
		}

		inline T& operator[] (size_t idx) const
		{
			if ((!is_valid()) || (idx >= m_size))
				container_abort("writable_span: invalid span or index\n");

			return m_p[idx];
		}

		template<typename R>
		inline R read_val(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(R)))
			{
				assert(0);
				return (R)0;
			}

			return *reinterpret_cast<const R*>(&m_p[ofs]);
		}

		template<typename R>
		inline bool write_val(size_t ofs, R val) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(R)))
			{
				assert(0);
				return false;
			}

			*reinterpret_cast<R*>(&m_p[ofs]) = val;
			return true;
		}

		inline bool write_le16(size_t ofs, uint16_t val) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return false;
			}

			m_p[ofs] = (uint8_t)val;
			m_p[ofs + 1] = (uint8_t)(val >> 8u);
			return true;
		}

		inline bool write_be16(size_t ofs, uint16_t val) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return false;
			}

			m_p[ofs + 1] = (uint8_t)val;
			m_p[ofs] = (uint8_t)(val >> 8u);
			return true;
		}

		inline bool write_le32(size_t ofs, uint32_t val) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return false;
			}

			m_p[ofs] = (uint8_t)val;
			m_p[ofs + 1] = (uint8_t)(val >> 8u);
			m_p[ofs + 2] = (uint8_t)(val >> 16u);
			m_p[ofs + 3] = (uint8_t)(val >> 24u);
			return true;
		}

		inline bool write_be32(size_t ofs, uint32_t val) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return false;
			}

			m_p[ofs + 3] = (uint8_t)val;
			m_p[ofs + 2] = (uint8_t)(val >> 8u);
			m_p[ofs + 1] = (uint8_t)(val >> 16u);
			m_p[ofs] = (uint8_t)(val >> 24u);
			return true;
		}

		inline bool write_le64(size_t ofs, uint64_t val) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return false;
			}

			return write_le32(ofs, (uint32_t)val) && write_le32(ofs + sizeof(uint32_t), (uint32_t)(val >> 32u));
		}

		inline bool write_be64(size_t ofs, uint64_t val) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return false;
			}

			return write_be32(ofs + sizeof(uint32_t), (uint32_t)val) && write_be32(ofs, (uint32_t)(val >> 32u));
		}

		inline uint16_t read_le16(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t a = (uint8_t)m_p[ofs];
			const uint8_t b = (uint8_t)m_p[ofs + 1];
			return a | (b << 8u);
		}

		inline uint16_t read_be16(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint16_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t b = (uint8_t)m_p[ofs];
			const uint8_t a = (uint8_t)m_p[ofs + 1];
			return a | (b << 8u);
		}

		inline uint32_t read_le32(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t a = (uint8_t)m_p[ofs];
			const uint8_t b = (uint8_t)m_p[ofs + 1];
			const uint8_t c = (uint8_t)m_p[ofs + 2];
			const uint8_t d = (uint8_t)m_p[ofs + 3];
			return a | (b << 8u) | (c << 16u) | (d << 24u);
		}

		inline uint32_t read_be32(size_t ofs) const
		{
			static_assert(sizeof(T) == 1, "T must be byte size");

			if (!is_inside(ofs, sizeof(uint32_t)))
			{
				assert(0);
				return 0;
			}

			const uint8_t d = (uint8_t)m_p[ofs];
			const uint8_t c = (uint8_t)m_p[ofs + 1];
			const uint8_t b = (uint8_t)m_p[ofs + 2];
			const uint8_t a = (uint8_t)m_p[ofs + 3];
			return a | (b << 8u) | (c << 16u) | (d << 24u);
		}

		inline uint64_t read_le64(size_t ofs) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return 0;
			}
			const uint64_t l = read_le32(ofs);
			const uint64_t h = read_le32(ofs + sizeof(uint32_t));
			return l | (h << 32u);
		}

		inline uint64_t read_be64(size_t ofs) const
		{
			if (!add_overflow_check(ofs, sizeof(uint64_t)))
			{
				assert(0);
				return 0;
			}
			const uint64_t h = read_be32(ofs);
			const uint64_t l = read_be32(ofs + sizeof(uint32_t));
			return l | (h << 32u);
		}

	private:
		T* m_p;
		size_t m_size;
	};

	template<typename T>
	inline readable_span<T>::readable_span(const writable_span<T>& other) :
		m_p(other.m_p),
		m_size(other.m_size)
	{
	}

	template<typename T>
	inline readable_span<T>& readable_span<T>::operator= (const writable_span<T>& rhs)
	{
		m_p = rhs.m_p;
		m_size = rhs.m_size;
		return *this;
	}

	template<typename T>
	inline bool span_copy(const writable_span<T>& dst, const readable_span<T>& src)
	{
		return dst.copy_into(src, 0, src.size(), 0);
	}

	template<typename T>
	inline bool span_copy(const writable_span<T>& dst, const writable_span<T>& src)
	{
		return dst.copy_into(src, 0, src.size(), 0);
	}

	template<typename T>
	inline bool span_copy(const writable_span<T>& dst, size_t dst_ofs, const writable_span<T>& src, size_t src_ofs, size_t len)
	{
		return dst.copy_into(src, src_ofs, len, dst_ofs);
	}

	template<typename T>
	inline bool span_copy(const writable_span<T>& dst, size_t dst_ofs, const readable_span<T>& src, size_t src_ofs, size_t len)
	{
		return dst.copy_into(src, src_ofs, len, dst_ofs);
	}

	template<typename T>
	class vector : public rel_ops< vector<T> >
	{
	public:
		typedef T* iterator;
		typedef const T* const_iterator;
		typedef T value_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		inline vector() :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
		}

		inline vector(size_t n, const T& init) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			increase_capacity(n, false);
			construct_array(m_p, n, init);
			m_size = n;
		}

		inline vector(vector&& other) :
			m_p(other.m_p),
			m_size(other.m_size),
			m_capacity(other.m_capacity)
		{
			other.m_p = nullptr;
			other.m_size = 0;
			other.m_capacity = 0;
		}

		inline vector(const vector& other) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			increase_capacity(other.m_size, false);

			m_size = other.m_size;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				if ((m_p) && (other.m_p))
				{
					memcpy((void *)m_p, other.m_p, m_size * sizeof(T));
				}
			}
			else
			{
				T* pDst = m_p;
				const T* pSrc = other.m_p;
				for (size_t i = m_size; i > 0; i--)
					construct(pDst++, *pSrc++);
			}
		}

		inline explicit vector(size_t size) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			resize(size);
		}

		inline explicit vector(std::initializer_list<T> init_list) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			resize(init_list.size());

			size_t idx = 0;
			for (const T& elem : init_list)
				m_p[idx++] = elem;

			assert(idx == m_size);
		}

		inline vector(const readable_span<T>& rs) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			set(rs);
		}

		inline vector(const writable_span<T>& ws) :
			m_p(nullptr),
			m_size(0),
			m_capacity(0)
		{
			set(ws);
		}

		// Set contents of vector to contents of the readable span
		bool set(const readable_span<T>& rs)
		{
			if (!rs.is_valid())
			{
				assert(0);
				return false;
			}

			const size_t new_size = rs.size();

			// Could call resize(), but it'll redundantly construct trivial types.
			if (m_size != new_size)
			{
				if (new_size < m_size)
				{
					if (BASISU_HAS_DESTRUCTOR(T))
					{
						scalar_type<T>::destruct_array(m_p + new_size, m_size - new_size);
					}
				}
				else
				{
					if (new_size > m_capacity)
					{
						if (!increase_capacity(new_size, false, true))
							return false;
					}
				}

				// Don't bother constructing trivial types, because we're going to memcpy() over them anyway.
				if (!BASISU_IS_BITWISE_COPYABLE(T))
				{
					scalar_type<T>::construct_array(m_p + m_size, new_size - m_size);
				}

				m_size = new_size;
			}

			if (!rs.copy_from(0, rs.size(), m_p, 0))
			{
				assert(0);
				return false;
			}

			return true;
		}

		// Set contents of vector to contents of the writable span
		inline bool set(const writable_span<T>& ws)
		{
			return set(ws.get_readable_span());
		}

		inline ~vector()
		{
			if (m_p)
			{
				if (BASISU_HAS_DESTRUCTOR(T))
				{
					scalar_type<T>::destruct_array(m_p, m_size);
				}

				free(m_p);
			}
		}

		inline vector& operator= (const vector& other)
		{
			if (this == &other)
				return *this;

			if (m_capacity >= other.m_size)
				resize(0);
			else
			{
				clear();
				increase_capacity(other.m_size, false);
			}

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				if ((m_p) && (other.m_p))
					memcpy((void *)m_p, other.m_p, other.m_size * sizeof(T));
			}
			else
			{
				T* pDst = m_p;
				const T* pSrc = other.m_p;
				for (size_t i = other.m_size; i > 0; i--)
					construct(pDst++, *pSrc++);
			}

			m_size = other.m_size;

			return *this;
		}

		inline vector& operator= (vector&& rhs)
		{
			if (this != &rhs)
			{
				clear();

				m_p = rhs.m_p;
				m_size = rhs.m_size;
				m_capacity = rhs.m_capacity;

				rhs.m_p = nullptr;
				rhs.m_size = 0;
				rhs.m_capacity = 0;
			}
			return *this;
		}

		BASISU_FORCE_INLINE const T* begin() const { return m_p; }
		BASISU_FORCE_INLINE T* begin() { return m_p; }

		BASISU_FORCE_INLINE const T* end() const { return m_p + m_size; }
		BASISU_FORCE_INLINE T* end() { return m_p + m_size; }

		BASISU_FORCE_INLINE bool empty() const { return !m_size; }

		BASISU_FORCE_INLINE size_t size() const { return m_size; }
		BASISU_FORCE_INLINE uint32_t size_u32() const { assert(m_size <= UINT32_MAX); return static_cast<uint32_t>(m_size); }

		BASISU_FORCE_INLINE size_t size_in_bytes() const { return m_size * sizeof(T); }
		BASISU_FORCE_INLINE uint32_t size_in_bytes_u32() const { assert((m_size * sizeof(T)) <= UINT32_MAX); return static_cast<uint32_t>(m_size * sizeof(T)); }

		BASISU_FORCE_INLINE size_t capacity() const { return m_capacity; }

#if !BASISU_VECTOR_FORCE_CHECKING
		BASISU_FORCE_INLINE const T& operator[] (size_t i) const { assert(i < m_size); return m_p[i]; }
		BASISU_FORCE_INLINE T& operator[] (size_t i) { assert(i < m_size); return m_p[i]; }
#else
		BASISU_FORCE_INLINE const T& operator[] (size_t i) const
		{
			if (i >= m_size)
				container_abort("vector::operator[] invalid index: %zu, max entries %u, type size %zu\n", i, m_size, sizeof(T));

			return m_p[i];
		}
		BASISU_FORCE_INLINE T& operator[] (size_t i)
		{
			if (i >= m_size)
				container_abort("vector::operator[] invalid index: %zu, max entries %u, type size %zu\n", i, m_size, sizeof(T));

			return m_p[i];
		}
#endif

		// at() always includes range checking, even in final builds, unlike operator [].
		BASISU_FORCE_INLINE const T& at(size_t i) const
		{
			if (i >= m_size)
				container_abort("vector::at() invalid index: %zu, max entries %u, type size %zu\n", i, m_size, sizeof(T));

			return m_p[i];
		}
		BASISU_FORCE_INLINE T& at(size_t i)
		{
			if (i >= m_size)
				container_abort("vector::at() invalid index: %zu, max entries %u, type size %zu\n", i, m_size, sizeof(T));

			return m_p[i];
		}

#if !BASISU_VECTOR_FORCE_CHECKING
		BASISU_FORCE_INLINE const T& front() const { assert(m_size); return m_p[0]; }
		BASISU_FORCE_INLINE T& front() { assert(m_size); return m_p[0]; }

		BASISU_FORCE_INLINE const T& back() const { assert(m_size); return m_p[m_size - 1]; }
		BASISU_FORCE_INLINE T& back() { assert(m_size); return m_p[m_size - 1]; }
#else
		BASISU_FORCE_INLINE const T& front() const
		{
			if (!m_size)
				container_abort("front: vector is empty, type size %zu\n", sizeof(T));

			return m_p[0];
		}
		BASISU_FORCE_INLINE T& front()
		{
			if (!m_size)
				container_abort("front: vector is empty, type size %zu\n", sizeof(T));

			return m_p[0];
		}

		BASISU_FORCE_INLINE const T& back() const
		{
			if (!m_size)
				container_abort("back: vector is empty, type size %zu\n", sizeof(T));

			return m_p[m_size - 1];
		}
		BASISU_FORCE_INLINE T& back()
		{
			if (!m_size)
				container_abort("back: vector is empty, type size %zu\n", sizeof(T));

			return m_p[m_size - 1];
		}
#endif

		BASISU_FORCE_INLINE const T* get_ptr() const { return m_p; }
		BASISU_FORCE_INLINE T* get_ptr() { return m_p; }

		BASISU_FORCE_INLINE const T* data() const { return m_p; }
		BASISU_FORCE_INLINE T* data() { return m_p; }

		// clear() sets the container to empty, then frees the allocated block.
		inline void clear()
		{
			if (m_p)
			{
				if (BASISU_HAS_DESTRUCTOR(T))
				{
					scalar_type<T>::destruct_array(m_p, m_size);
				}

				free(m_p);

				m_p = nullptr;
				m_size = 0;
				m_capacity = 0;
			}
		}

		inline void clear_no_destruction()
		{
			if (m_p)
			{
				free(m_p);
				m_p = nullptr;
				m_size = 0;
				m_capacity = 0;
			}
		}

		inline void reserve(size_t new_capacity)
		{
			if (!try_reserve(new_capacity))
				container_abort("vector:reserve: try_reserve failed!\n");
		}

		inline bool try_reserve(size_t new_capacity)
		{
			if (new_capacity > m_capacity)
			{
				if (!increase_capacity(new_capacity, false, true))
					return false;
			}
			else if (new_capacity < m_capacity)
			{
				// Must work around the lack of a "decrease_capacity()" method.
				// This case is rare enough in practice that it's probably not worth implementing an optimized in-place resize.
				vector tmp;
				if (!tmp.increase_capacity(helpers::maximum(m_size, new_capacity), false, true))
					return false;

				tmp = *this;
				swap(tmp);
			}

			return true;
		}

		// try_resize(0) sets the container to empty, but does not free the allocated block.
		inline bool try_resize(size_t new_size, bool grow_hint = false)
		{
			if (m_size != new_size)
			{
				if (new_size < m_size)
				{
					if (BASISU_HAS_DESTRUCTOR(T))
					{
						scalar_type<T>::destruct_array(m_p + new_size, m_size - new_size);
					}
				}
				else
				{
					if (new_size > m_capacity)
					{
						if (!increase_capacity(new_size, (new_size == (m_size + 1)) || grow_hint, true))
							return false;
					}

					scalar_type<T>::construct_array(m_p + m_size, new_size - m_size);
				}

				m_size = new_size;
			}

			return true;
		}

		// resize(0) sets the container to empty, but does not free the allocated block.
		inline void resize(size_t new_size, bool grow_hint = false)
		{
			if (!try_resize(new_size, grow_hint))
				container_abort("vector::resize failed, new size %zu\n", new_size);
		}

		// If size >= capacity/2, reset() sets the container's size to 0 but doesn't free the allocated block (because the container may be similarly loaded in the future).
		// Otherwise it blows away the allocated block. See http://www.codercorner.com/blog/?p=494
		inline void reset()
		{
			if (m_size >= (m_capacity >> 1))
				resize(0);
			else
				clear();
		}

		inline T* try_enlarge(size_t i)
		{
			size_t cur_size = m_size;

			if (add_overflow_check(cur_size, i))
				return nullptr;

			if (!try_resize(cur_size + i, true))
				return nullptr;

			return get_ptr() + cur_size;
		}

		inline T* enlarge(size_t i)
		{
			T* p = try_enlarge(i);
			if (!p)
				container_abort("vector::enlarge failed, amount %zu!\n", i);
			return p;
		}

		BASISU_FORCE_INLINE void push_back(const T& obj)
		{
			assert(!m_p || (&obj < m_p) || (&obj >= (m_p + m_size)));

			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					container_abort("vector::push_back: vector too large\n");

				increase_capacity(m_size + 1, true);
			}

			scalar_type<T>::construct(m_p + m_size, obj);
			m_size++;
		}

		BASISU_FORCE_INLINE void push_back_value(T&& obj)
		{
			assert(!m_p || (&obj < m_p) || (&obj >= (m_p + m_size)));

			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					container_abort("vector::push_back_value: vector too large\n");

				increase_capacity(m_size + 1, true);
			}

			new ((void*)(m_p + m_size)) T(std::move(obj));
			m_size++;
		}

		inline bool try_push_back(const T& obj)
		{
			assert(!m_p || (&obj < m_p) || (&obj >= (m_p + m_size)));

			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					return false;

				if (!increase_capacity(m_size + 1, true, true))
					return false;
			}

			scalar_type<T>::construct(m_p + m_size, obj);
			m_size++;

			return true;
		}

		inline bool try_push_back(T&& obj)
		{
			assert(!m_p || (&obj < m_p) || (&obj >= (m_p + m_size)));

			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					return false;

				if (!increase_capacity(m_size + 1, true, true))
					return false;
			}

			new ((void*)(m_p + m_size)) T(std::move(obj));
			m_size++;

			return true;
		}

		// obj is explictly passed in by value, not ref
		inline void push_back_value(T obj)
		{
			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					container_abort("vector::push_back_value: vector too large\n");

				increase_capacity(m_size + 1, true);
			}

			scalar_type<T>::construct(m_p + m_size, obj);
			m_size++;
		}

		// obj is explictly passed in by value, not ref
		inline bool try_push_back_value(T obj)
		{
			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					return false;

				if (!increase_capacity(m_size + 1, true, true))
					return false;
			}

			scalar_type<T>::construct(m_p + m_size, obj);
			m_size++;

			return true;
		}

		template<typename... Args>
		BASISU_FORCE_INLINE void emplace_back(Args&&... args)
		{
			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					container_abort("vector::enlarge: vector too large\n");

				increase_capacity(m_size + 1, true);
			}

			new ((void*)(m_p + m_size)) T(std::forward<Args>(args)...); // perfect forwarding
			m_size++;
		}

		template<typename... Args>
		BASISU_FORCE_INLINE bool try_emplace_back(Args&&... args)
		{
			if (m_size >= m_capacity)
			{
				if (add_overflow_check(m_size, 1))
					return false;

				if (!increase_capacity(m_size + 1, true, true))
					return false;
			}

			new ((void*)(m_p + m_size)) T(std::forward<Args>(args)...); // perfect forwarding
			m_size++;

			return true;
		}

		inline void pop_back()
		{
			assert(m_size);

			if (m_size)
			{
				m_size--;
				scalar_type<T>::destruct(&m_p[m_size]);
			}
		}

		inline bool try_insert(size_t index, const T* p, size_t n)
		{
			assert(index <= m_size);

			if (index > m_size)
				return false;

			if (!n)
				return true;

			const size_t orig_size = m_size;

			if (add_overflow_check(m_size, n))
				return false;

			if (!try_resize(m_size + n, true))
				return false;

			const size_t num_to_move = orig_size - index;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				// This overwrites the destination object bits, but bitwise copyable means we don't need to worry about destruction.
				memmove(m_p + index + n, m_p + index, sizeof(T) * num_to_move);
			}
			else
			{
				const T* pSrc = m_p + orig_size - 1;
				T* pDst = const_cast<T*>(pSrc) + n;

				for (size_t i = 0; i < num_to_move; i++)
				{
					assert((uint64_t)(pDst - m_p) < (uint64_t)m_size);

					*pDst = std::move(*pSrc);
					pDst--;
					pSrc--;
				}
			}

			T* pDst = m_p + index;

			if (BASISU_IS_BITWISE_COPYABLE(T))
			{
				// This copies in the new bits, overwriting the existing objects, which is OK for copyable types that don't need destruction.
				memcpy(pDst, p, sizeof(T) * n);
			}
			else
			{
				for (size_t i = 0; i < n; i++)
				{
					assert((uint64_t)(pDst - m_p) < (uint64_t)m_size);
					*pDst++ = *p++;
				}
			}

			return true;
		}

		inline void insert(size_t index, const T* p, size_t n)
		{
			if (!try_insert(index, p, n))
				container_abort("vector::insert() failed!\n");
		}

		inline bool try_insert(T* p, const T& obj)
		{
			if (p < begin())
			{
				assert(0);
				return false;
			}

			uint64_t ofs = p - begin();

			if (ofs > m_size)
			{
				assert(0);
				return false;
			}

			if ((size_t)ofs != ofs)
			{
				assert(0);
				return false;
			}

			return try_insert((size_t)ofs, &obj, 1);
		}

		inline void insert(T* p, const T& obj)
		{
			if (!try_insert(p, obj))
				container_abort("vector::insert() failed!\n");
		}
				
		// push_front() isn't going to be very fast - it's only here for usability.
		inline void push_front(const T& obj)
		{
			insert(0, &obj, 1);
		}

		inline bool try_push_front(const T& obj)
		{
			return try_insert(0, &obj, 1);
		}

		vector& append(const vector& other)
		{
			if (other.m_size)
				insert(m_size, &other[0], other.m_size);
			return *this;
		}

		bool try_append(const vector& other)
		{
			if (other.m_size)
				return try_insert(m_size, &other[0], other.m_size);

			return true;
		}

		vector& append(const T* p, size_t n)
		{
			if (n)
				insert(m_size, p, n);
			return *this;
		}

		bool try_append(const T* p, size_t n)
		{
			if (n)
				return try_insert(m_size, p, n);

			return true;
		}

		inline bool erase(size_t start, size_t n)
		{
			if (add_overflow_check(start, n))
			{
				assert(0);
				return false;
			}

			assert((start + n) <= m_size);

			if ((start + n) > m_size)
			{
				assert(0);
				return false;
			}

			if (!n)
				return true;

			const size_t num_to_move = m_size - (start + n);

			T* pDst = m_p + start;

			const T* pSrc = m_p + start + n;

			if (BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(T))
			{
				// This test is overly cautious.
				if ((!BASISU_IS_BITWISE_COPYABLE(T)) || (BASISU_HAS_DESTRUCTOR(T)))
				{
					// Type has been marked explictly as bitwise movable, which means we can move them around but they may need to be destructed.
					// First destroy the erased objects.
					scalar_type<T>::destruct_array(pDst, n);
				}

				// Copy "down" the objects to preserve, filling in the empty slots.
				memmove((void *)pDst, pSrc, num_to_move * sizeof(T));
			}
			else
			{
				// Type is not bitwise copyable or movable. 
				// Move them down one at a time by using the equals operator, and destroying anything that's left over at the end.
				T* pDst_end = pDst + num_to_move;

				while (pDst != pDst_end)
				{
					*pDst = std::move(*pSrc);

					++pDst;
					++pSrc;
				}

				scalar_type<T>::destruct_array(pDst_end, n);
			}

			m_size -= n;

			return true;
		}

		inline bool erase_index(size_t index)
		{
			return erase(index, 1);
		}

		inline bool erase(T* p)
		{
			assert((p >= m_p) && (p < (m_p + m_size)));

			if (p < m_p)
				return false;

			return erase_index(static_cast<size_t>(p - m_p));
		}

		inline bool erase(T* pFirst, T* pEnd)
		{
			assert(pFirst <= pEnd);
			assert(pFirst >= begin() && pFirst <= end());
			assert(pEnd >= begin() && pEnd <= end());

			if ((pFirst < begin()) || (pEnd < pFirst))
			{
				assert(0);
				return false;
			}

			uint64_t ofs = pFirst - begin();
			if ((size_t)ofs != ofs)
			{
				assert(0);
				return false;
			}

			uint64_t n = pEnd - pFirst;
			if ((size_t)n != n)
			{
				assert(0);
				return false;
			}

			return erase((size_t)ofs, (size_t)n);
		}

		bool erase_unordered(size_t index)
		{
			if (index >= m_size)
			{
				assert(0);
				return false;
			}

			if ((index + 1) < m_size)
			{
				(*this)[index] = std::move(back());
			}

			pop_back();
			return true;
		}

		inline bool operator== (const vector& rhs) const
		{
			if (m_size != rhs.m_size)
				return false;
			else if (m_size)
			{
				if (scalar_type<T>::cFlag)
					return memcmp(m_p, rhs.m_p, sizeof(T) * m_size) == 0;
				else
				{
					const T* pSrc = m_p;
					const T* pDst = rhs.m_p;
					for (size_t i = m_size; i; i--)
						if (!(*pSrc++ == *pDst++))
							return false;
				}
			}

			return true;
		}

		inline bool operator< (const vector& rhs) const
		{
			const size_t min_size = helpers::minimum(m_size, rhs.m_size);

			const T* pSrc = m_p;
			const T* pSrc_end = m_p + min_size;
			const T* pDst = rhs.m_p;

			while ((pSrc < pSrc_end) && (*pSrc == *pDst))
			{
				pSrc++;
				pDst++;
			}

			if (pSrc < pSrc_end)
				return *pSrc < *pDst;

			return m_size < rhs.m_size;
		}

		inline void swap(vector& other)
		{
			std::swap(m_p, other.m_p);
			std::swap(m_size, other.m_size);
			std::swap(m_capacity, other.m_capacity);
		}

		inline void sort()
		{
			std::sort(begin(), end());
		}

		inline void unique()
		{
			if (!empty())
			{
				sort();

				resize(std::unique(begin(), end()) - begin());
			}
		}

		inline void reverse()
		{
			const size_t j = m_size >> 1;

			for (size_t i = 0; i < j; i++)
				std::swap(m_p[i], m_p[m_size - 1 - i]);
		}

		inline bool find(const T& key, size_t &idx) const
		{
			idx = 0;

			const T* p = m_p;
			const T* p_end = m_p + m_size;

			size_t index = 0;

			while (p != p_end)
			{
				if (key == *p)
				{
					idx = index;
					return true;
				}

				p++;
				index++;
			}

			return false;
		}

		inline bool find_sorted(const T& key, size_t& idx) const
		{
			idx = 0;

			if (!m_size)
				return false;

			// Inclusive range
			size_t low = 0, high = m_size - 1;

			while (low <= high)
			{
				size_t mid = (size_t)(((uint64_t)low + (uint64_t)high) >> 1);

				const T* pTrial_key = m_p + mid;

				// Sanity check comparison operator
				assert(!((*pTrial_key < key) && (key < *pTrial_key)));

				if (*pTrial_key < key)
				{
					if (add_overflow_check(mid, 1))
						break;

					low = mid + 1;
				}
				else if (key < *pTrial_key)
				{
					if (!mid)
						break;

					high = mid - 1;
				}
				else
				{
					idx = mid;
					return true;
				}
			}

			return false;
		}

		inline size_t count_occurences(const T& key) const
		{
			size_t c = 0;

			const T* p = m_p;
			const T* p_end = m_p + m_size;

			while (p != p_end)
			{
				if (key == *p)
					c++;

				p++;
			}

			return c;
		}

		inline void set_all(const T& o)
		{
			if ((sizeof(T) == 1) && (scalar_type<T>::cFlag))
			{
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
				memset(m_p, *reinterpret_cast<const uint8_t*>(&o), m_size);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
			}
			else
			{
				T* pDst = m_p;
				T* pDst_end = pDst + m_size;
				while (pDst != pDst_end)
					*pDst++ = o;
			}
		}

		// Caller assumes ownership of the heap block associated with the container. Container is cleared.
		// Caller must use free() on the returned pointer.
		inline void* assume_ownership()
		{
			T* p = m_p;
			m_p = nullptr;
			m_size = 0;
			m_capacity = 0;
			return p;
		}

		// Caller is granting ownership of the indicated heap block.
		// Block must have size constructed elements, and have enough room for capacity elements.
		// The block must have been allocated using malloc().
		// Important: This method is used in Basis Universal. If you change how this container allocates memory, you'll need to change any users of this method.
		inline bool grant_ownership(T* p, size_t size, size_t capacity)
		{
			// To prevent the caller from obviously shooting themselves in the foot.
			if (((p + capacity) > m_p) && (p < (m_p + m_capacity)))
			{
				// Can grant ownership of a block inside the container itself!
				assert(0);
				return false;
			}

			if (size > capacity)
			{
				assert(0);
				return false;
			}

			if (!p)
			{
				if (capacity)
				{
					assert(0);
					return false;
				}
			}
			else if (!capacity)
			{
				assert(0);
				return false;
			}

			clear();
			m_p = p;
			m_size = size;
			m_capacity = capacity;
			return true;
		}

		readable_span<T> get_readable_span() const
		{
			return readable_span<T>(m_p, m_size);
		}

		writable_span<T> get_writable_span()
		{
			return writable_span<T>(m_p, m_size);
		}

	private:
		T* m_p;
		size_t m_size;		   // the number of constructed objects
		size_t m_capacity;	// the size of the allocation

		template<typename Q> struct is_vector { enum { cFlag = false }; };
		template<typename Q> struct is_vector< vector<Q> > { enum { cFlag = true }; };

		static void object_mover(void* pDst_void, void* pSrc_void, size_t num)
		{
			T* pSrc = static_cast<T*>(pSrc_void);
			T* const pSrc_end = pSrc + num;
			T* pDst = static_cast<T*>(pDst_void);

			while (pSrc != pSrc_end)
			{
				new ((void*)(pDst)) T(std::move(*pSrc));
				scalar_type<T>::destruct(pSrc);

				++pSrc;
				++pDst;
			}
		}

		inline bool increase_capacity(size_t min_new_capacity, bool grow_hint, bool nofail = false)
		{
			return reinterpret_cast<elemental_vector*>(this)->increase_capacity(
				min_new_capacity, grow_hint, sizeof(T),
				(BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(T) || (is_vector<T>::cFlag)) ? nullptr : object_mover, nofail);
		}
	};

	template<typename T> struct bitwise_movable< vector<T> > { enum { cFlag = true }; };

	// Hash map
	// rg TODO 9/8/2024: I've upgraded this class to support 64-bit size_t, and it needs a lot more testing.

	const uint32_t SIZE_T_BITS = sizeof(size_t) * 8U;

	inline uint32_t safe_shift_left(uint32_t v, uint32_t l)
	{
		return (l < 32U) ? (v << l) : 0;
	}

	inline uint64_t safe_shift_left(uint64_t v, uint32_t l)
	{
		return (l < 64U) ? (v << l) : 0;
	}

	template <typename T>
	struct hasher
	{
		inline size_t operator() (const T& key) const { return static_cast<size_t>(key); }
	};

	template <typename T>
	struct equal_to
	{
		inline bool operator()(const T& a, const T& b) const { return a == b; }
	};

	// Important: The Hasher and Equals objects must be bitwise movable!
	template<typename Key, typename Value = empty_type, typename Hasher = hasher<Key>, typename Equals = equal_to<Key> >
	class hash_map
	{
	public:
		class iterator;
		class const_iterator;

	private:
		friend class iterator;
		friend class const_iterator;

		enum state
		{
			cStateInvalid = 0,
			cStateValid = 1
		};

		enum
		{
			cMinHashSize = 4U
		};

	public:
		typedef hash_map<Key, Value, Hasher, Equals> hash_map_type;
		typedef std::pair<Key, Value> value_type;
		typedef Key                   key_type;
		typedef Value                 referent_type;
		typedef Hasher                hasher_type;
		typedef Equals                equals_type;

		hash_map() :
			m_num_valid(0),
			m_grow_threshold(0),
			m_hash_shift(SIZE_T_BITS)
		{
			static_assert((SIZE_T_BITS == 32) || (SIZE_T_BITS == 64), "SIZE_T_BITS must be 32 or 64");
		}

		hash_map(const hash_map& other) :
			m_values(other.m_values),
			m_num_valid(other.m_num_valid),
			m_grow_threshold(other.m_grow_threshold),
			m_hash_shift(other.m_hash_shift),
			m_hasher(other.m_hasher),
			m_equals(other.m_equals)
		{
			static_assert((SIZE_T_BITS == 32) || (SIZE_T_BITS == 64), "SIZE_T_BITS must be 32 or 64");
		}

		hash_map(hash_map&& other) :
			m_values(std::move(other.m_values)),
			m_num_valid(other.m_num_valid),
			m_grow_threshold(other.m_grow_threshold),
			m_hash_shift(other.m_hash_shift),
			m_hasher(std::move(other.m_hasher)),
			m_equals(std::move(other.m_equals))
		{
			static_assert((SIZE_T_BITS == 32) || (SIZE_T_BITS == 64), "SIZE_T_BITS must be 32 or 64");

			other.m_hash_shift = SIZE_T_BITS;
			other.m_num_valid = 0;
			other.m_grow_threshold = 0;
		}

		hash_map& operator= (const hash_map& other)
		{
			if (this == &other)
				return *this;

			clear();

			m_values = other.m_values;
			m_hash_shift = other.m_hash_shift;
			m_num_valid = other.m_num_valid;
			m_grow_threshold = other.m_grow_threshold;
			m_hasher = other.m_hasher;
			m_equals = other.m_equals;

			return *this;
		}

		hash_map& operator= (hash_map&& other)
		{
			if (this == &other)
				return *this;

			clear();

			m_values = std::move(other.m_values);
			m_hash_shift = other.m_hash_shift;
			m_num_valid = other.m_num_valid;
			m_grow_threshold = other.m_grow_threshold;
			m_hasher = std::move(other.m_hasher);
			m_equals = std::move(other.m_equals);

			other.m_hash_shift = SIZE_T_BITS;
			other.m_num_valid = 0;
			other.m_grow_threshold = 0;

			return *this;
		}

		inline ~hash_map()
		{
			clear();
		}

		inline const Equals& get_equals() const { return m_equals; }
		inline Equals& get_equals() { return m_equals; }
		inline void set_equals(const Equals& equals) { m_equals = equals; }

		inline const Hasher& get_hasher() const { return m_hasher; }
		inline Hasher& get_hasher() { return m_hasher; }
		inline void set_hasher(const Hasher& hasher) { m_hasher = hasher; }

		inline void clear()
		{
			if (m_values.empty())
				return;

			if (BASISU_HAS_DESTRUCTOR(Key) || BASISU_HAS_DESTRUCTOR(Value))
			{
				node* p = &get_node(0);
				node* p_end = p + m_values.size();

				size_t num_remaining = m_num_valid;
				while (p != p_end)
				{
					if (p->state)
					{
						destruct_value_type(p);
						num_remaining--;
						if (!num_remaining)
							break;
					}

					p++;
				}
			}

			m_values.clear_no_destruction();

			m_hash_shift = SIZE_T_BITS;
			m_num_valid = 0;
			m_grow_threshold = 0;
		}

		inline void reset()
		{
			if (!m_num_valid)
				return;

			if (BASISU_HAS_DESTRUCTOR(Key) || BASISU_HAS_DESTRUCTOR(Value))
			{
				node* p = &get_node(0);
				node* p_end = p + m_values.size();

				size_t num_remaining = m_num_valid;
				while (p != p_end)
				{
					if (p->state)
					{
						destruct_value_type(p);
						p->state = cStateInvalid;

						num_remaining--;
						if (!num_remaining)
							break;
					}

					p++;
				}
			}
			else if (sizeof(node) <= 16)
			{
				memset(&m_values[0], 0, m_values.size_in_bytes());
			}
			else
			{
				node* p = &get_node(0);
				node* p_end = p + m_values.size();

				size_t num_remaining = m_num_valid;
				while (p != p_end)
				{
					if (p->state)
					{
						p->state = cStateInvalid;

						num_remaining--;
						if (!num_remaining)
							break;
					}

					p++;
				}
			}

			m_num_valid = 0;
		}

		inline size_t size()
		{
			return m_num_valid;
		}

		inline size_t get_table_size()
		{
			return m_values.size();
		}

		inline bool empty()
		{
			return !m_num_valid;
		}

		inline bool reserve(size_t new_capacity)
		{
			if (!new_capacity)
				return true;

			uint64_t new_hash_size = new_capacity;

			new_hash_size = new_hash_size * 2ULL;

			if (!helpers::is_power_of_2(new_hash_size))
				new_hash_size = helpers::next_pow2(new_hash_size);

			new_hash_size = helpers::maximum<uint64_t>(cMinHashSize, new_hash_size);

			if (!can_fit_into_size_t(new_hash_size))
			{
				assert(0);
				return false;
			}

			assert(new_hash_size >= new_capacity);

			if (new_hash_size <= m_values.size())
				return true;

			return rehash((size_t)new_hash_size);
		}

		class iterator
		{
			friend class hash_map<Key, Value, Hasher, Equals>;
			friend class hash_map<Key, Value, Hasher, Equals>::const_iterator;

		public:
			inline iterator() : m_pTable(nullptr), m_index(0) { }
			inline iterator(hash_map_type& table, size_t index) : m_pTable(&table), m_index(index) { }
			inline iterator(const iterator& other) : m_pTable(other.m_pTable), m_index(other.m_index) { }

			inline iterator& operator= (const iterator& other)
			{
				m_pTable = other.m_pTable;
				m_index = other.m_index;
				return *this;
			}

			// post-increment
			inline iterator operator++(int)
			{
				iterator result(*this);
				++*this;
				return result;
			}

			// pre-increment
			inline iterator& operator++()
			{
				probe();
				return *this;
			}

			inline value_type& operator*() const { return *get_cur(); }
			inline value_type* operator->() const { return get_cur(); }

			inline bool operator == (const iterator& b) const { return (m_pTable == b.m_pTable) && (m_index == b.m_index); }
			inline bool operator != (const iterator& b) const { return !(*this == b); }
			inline bool operator == (const const_iterator& b) const { return (m_pTable == b.m_pTable) && (m_index == b.m_index); }
			inline bool operator != (const const_iterator& b) const { return !(*this == b); }

		private:
			hash_map_type* m_pTable;
			size_t m_index;

			inline value_type* get_cur() const
			{
				assert(m_pTable && (m_index < m_pTable->m_values.size()));
				assert(m_pTable->get_node_state(m_index) == cStateValid);

				return &m_pTable->get_node(m_index);
			}

			inline void probe()
			{
				assert(m_pTable);
				m_index = m_pTable->find_next(m_index);
			}
		};

		class const_iterator
		{
			friend class hash_map<Key, Value, Hasher, Equals>;
			friend class hash_map<Key, Value, Hasher, Equals>::iterator;

		public:
			inline const_iterator() : m_pTable(nullptr), m_index(0) { }
			inline const_iterator(const hash_map_type& table, size_t index) : m_pTable(&table), m_index(index) { }
			inline const_iterator(const iterator& other) : m_pTable(other.m_pTable), m_index(other.m_index) { }
			inline const_iterator(const const_iterator& other) : m_pTable(other.m_pTable), m_index(other.m_index) { }

			inline const_iterator& operator= (const const_iterator& other)
			{
				m_pTable = other.m_pTable;
				m_index = other.m_index;
				return *this;
			}

			inline const_iterator& operator= (const iterator& other)
			{
				m_pTable = other.m_pTable;
				m_index = other.m_index;
				return *this;
			}

			// post-increment
			inline const_iterator operator++(int)
			{
				const_iterator result(*this);
				++*this;
				return result;
			}

			// pre-increment
			inline const_iterator& operator++()
			{
				probe();
				return *this;
			}

			inline const value_type& operator*() const { return *get_cur(); }
			inline const value_type* operator->() const { return get_cur(); }

			inline bool operator == (const const_iterator& b) const { return (m_pTable == b.m_pTable) && (m_index == b.m_index); }
			inline bool operator != (const const_iterator& b) const { return !(*this == b); }
			inline bool operator == (const iterator& b) const { return (m_pTable == b.m_pTable) && (m_index == b.m_index); }
			inline bool operator != (const iterator& b) const { return !(*this == b); }

		private:
			const hash_map_type* m_pTable;
			size_t m_index;

			inline const value_type* get_cur() const
			{
				assert(m_pTable && (m_index < m_pTable->m_values.size()));
				assert(m_pTable->get_node_state(m_index) == cStateValid);

				return &m_pTable->get_node(m_index);
			}

			inline void probe()
			{
				assert(m_pTable);
				m_index = m_pTable->find_next(m_index);
			}
		};

		inline const_iterator begin() const
		{
			if (!m_num_valid)
				return end();

			return const_iterator(*this, find_next(std::numeric_limits<size_t>::max()));
		}

		inline const_iterator end() const
		{
			return const_iterator(*this, m_values.size());
		}

		inline iterator begin()
		{
			if (!m_num_valid)
				return end();

			return iterator(*this, find_next(std::numeric_limits<size_t>::max()));
		}

		inline iterator end()
		{
			return iterator(*this, m_values.size());
		}

		// insert_result.first will always point to inserted key/value (or the already existing key/value).
		// insert_result.second will be true if a new key/value was inserted, or false if the key already existed (in which case first will point to the already existing value).
		typedef std::pair<iterator, bool> insert_result;

		inline insert_result insert(const Key& k, const Value& v = Value())
		{
			insert_result result;
			if (!insert_no_grow(result, k, v))
			{
				if (!try_grow())
					container_abort("hash_map::try_grow() failed");

				// This must succeed.
				if (!insert_no_grow(result, k, v))
					container_abort("hash_map::insert() failed");
			}

			return result;
		}

		inline bool try_insert(insert_result& result, const Key& k, const Value& v = Value())
		{
			if (!insert_no_grow(result, k, v))
			{
				if (!try_grow())
					return false;

				if (!insert_no_grow(result, k, v))
					return false;
			}

			return true;
		}

		inline insert_result insert(Key&& k, Value&& v = Value())
		{
			insert_result result;
			if (!insert_no_grow_move(result, std::move(k), std::move(v)))
			{
				if (!try_grow())
					container_abort("hash_map::try_grow() failed");

				// This must succeed.
				if (!insert_no_grow_move(result, std::move(k), std::move(v)))
					container_abort("hash_map::insert() failed");
			}

			return result;
		}

		inline bool try_insert(insert_result& result, Key&& k, Value&& v = Value())
		{
			if (!insert_no_grow_move(result, std::move(k), std::move(v)))
			{
				if (!try_grow())
					return false;

				if (!insert_no_grow_move(result, std::move(k), std::move(v)))
					return false;
			}

			return true;
		}

		inline insert_result insert(const value_type& v)
		{
			return insert(v.first, v.second);
		}

		inline bool try_insert(insert_result& result, const value_type& v)
		{
			return try_insert(result, v.first, v.second);
		}

		inline insert_result insert(value_type&& v)
		{
			return insert(std::move(v.first), std::move(v.second));
		}

		inline bool try_insert(insert_result& result, value_type&& v)
		{
			return try_insert(result, std::move(v.first), std::move(v.second));
		}
				
		inline const_iterator find(const Key& k) const
		{
			return const_iterator(*this, find_index(k));
		}

		inline iterator find(const Key& k)
		{
			return iterator(*this, find_index(k));
		}

		inline bool contains(const Key& k) const
		{
			const size_t idx = find_index(k);
			return idx != m_values.size();
		}

		inline bool erase(const Key& k)
		{
			size_t i = find_index(k);

			if (i >= m_values.size())
				return false;

			node* pDst = &get_node(i);
			destruct_value_type(pDst);
			pDst->state = cStateInvalid;

			m_num_valid--;

			for (; ; )
			{
				size_t r, j = i;

				node* pSrc = pDst;

				do
				{
					if (!i)
					{
						i = m_values.size() - 1;
						pSrc = &get_node(i);
					}
					else
					{
						i--;
						pSrc--;
					}

					if (!pSrc->state)
						return true;

					r = hash_key(pSrc->first);

				} while ((i <= r && r < j) || (r < j && j < i) || (j < i && i <= r));

				move_node(pDst, pSrc);

				pDst = pSrc;
			}
		}

		inline void swap(hash_map_type& other)
		{
			m_values.swap(other.m_values);
			std::swap(m_hash_shift, other.m_hash_shift);
			std::swap(m_num_valid, other.m_num_valid);
			std::swap(m_grow_threshold, other.m_grow_threshold);
			std::swap(m_hasher, other.m_hasher);
			std::swap(m_equals, other.m_equals);
		}

	private:
		struct node : public value_type
		{
			uint8_t state;
		};

		static inline void construct_value_type(value_type* pDst, const Key& k, const Value& v)
		{
			if (BASISU_IS_BITWISE_COPYABLE(Key))
				memcpy(&pDst->first, &k, sizeof(Key));
			else
				scalar_type<Key>::construct(&pDst->first, k);

			if (BASISU_IS_BITWISE_COPYABLE(Value))
				memcpy(&pDst->second, &v, sizeof(Value));
			else
				scalar_type<Value>::construct(&pDst->second, v);
		}

		static inline void construct_value_type(value_type* pDst, const value_type* pSrc)
		{
			if ((BASISU_IS_BITWISE_COPYABLE(Key)) && (BASISU_IS_BITWISE_COPYABLE(Value)))
			{
				memcpy(pDst, pSrc, sizeof(value_type));
			}
			else
			{
				if (BASISU_IS_BITWISE_COPYABLE(Key))
					memcpy(&pDst->first, &pSrc->first, sizeof(Key));
				else
					scalar_type<Key>::construct(&pDst->first, pSrc->first);

				if (BASISU_IS_BITWISE_COPYABLE(Value))
					memcpy(&pDst->second, &pSrc->second, sizeof(Value));
				else
					scalar_type<Value>::construct(&pDst->second, pSrc->second);
			}
		}

		static inline void destruct_value_type(value_type* p)
		{
			scalar_type<Key>::destruct(&p->first);
			scalar_type<Value>::destruct(&p->second);
		}

		// Moves nodes *pSrc to *pDst efficiently from one hashmap to another.
		// pDst should NOT be constructed on entry.
		static inline void move_node(node* pDst, node* pSrc, bool update_src_state = true)
		{
			assert(!pDst->state);

			if (BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(Key) && BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(Value))
			{
				memcpy(pDst, pSrc, sizeof(node));

				assert(pDst->state == cStateValid);
			}
			else
			{
				if (BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(Key))
					memcpy(&pDst->first, &pSrc->first, sizeof(Key));
				else
				{
					new ((void*)&pDst->first) Key(std::move(pSrc->first));
					scalar_type<Key>::destruct(&pSrc->first);
				}

				if (BASISU_IS_BITWISE_COPYABLE_OR_MOVABLE(Value))
					memcpy(&pDst->second, &pSrc->second, sizeof(Value));
				else
				{
					new ((void*)&pDst->second) Value(std::move(pSrc->second));
					scalar_type<Value>::destruct(&pSrc->second);
				}

				pDst->state = cStateValid;
			}

			if (update_src_state)
				pSrc->state = cStateInvalid;
		}

		struct raw_node
		{
			inline raw_node()
			{
				node* p = reinterpret_cast<node*>(this);
				p->state = cStateInvalid;
			}

			// In practice, this should never be called (right?). We manage destruction ourselves.
			inline ~raw_node()
			{
				node* p = reinterpret_cast<node*>(this);
				if (p->state)
					hash_map_type::destruct_value_type(p);
			}

			inline raw_node(const raw_node& other)
			{
				node* pDst = reinterpret_cast<node*>(this);
				const node* pSrc = reinterpret_cast<const node*>(&other);

				if (pSrc->state)
				{
					hash_map_type::construct_value_type(pDst, pSrc);
					pDst->state = cStateValid;
				}
				else
					pDst->state = cStateInvalid;
			}

			inline raw_node& operator= (const raw_node& rhs)
			{
				if (this == &rhs)
					return *this;

				node* pDst = reinterpret_cast<node*>(this);
				const node* pSrc = reinterpret_cast<const node*>(&rhs);

				if (pSrc->state)
				{
					if (pDst->state)
					{
						pDst->first = pSrc->first;
						pDst->second = pSrc->second;
					}
					else
					{
						hash_map_type::construct_value_type(pDst, pSrc);
						pDst->state = cStateValid;
					}
				}
				else if (pDst->state)
				{
					hash_map_type::destruct_value_type(pDst);
					pDst->state = cStateInvalid;
				}

				return *this;
			}

			uint8_t m_bits[sizeof(node)];
		};

		typedef basisu::vector<raw_node> node_vector;

		node_vector		m_values;

		size_t			m_num_valid;
		size_t			m_grow_threshold;

		uint32_t		m_hash_shift;

		Hasher			m_hasher;
		Equals			m_equals;

		inline size_t hash_key(const Key& k) const
		{
			assert((safe_shift_left(static_cast<uint64_t>(1), (SIZE_T_BITS - m_hash_shift))) == m_values.size());

			// Fibonacci hashing
			if (SIZE_T_BITS == 32)
			{
				assert(m_hash_shift != 32);

				uint32_t hash = static_cast<uint32_t>(m_hasher(k));
				hash = (2654435769U * hash) >> m_hash_shift;

				assert(hash < m_values.size());
				return (size_t)hash;
			}
			else
			{
				assert(m_hash_shift != 64);

				uint64_t hash = static_cast<uint64_t>(m_hasher(k));
				hash = (0x9E3779B97F4A7C15ULL * hash) >> m_hash_shift;

				assert(hash < m_values.size());
				return (size_t)hash;
			}
		}

		inline const node& get_node(size_t index) const
		{
			return *reinterpret_cast<const node*>(&m_values[index]);
		}

		inline node& get_node(size_t index)
		{
			return *reinterpret_cast<node*>(&m_values[index]);
		}

		inline state get_node_state(size_t index) const
		{
			return static_cast<state>(get_node(index).state);
		}

		inline void set_node_state(size_t index, bool valid)
		{
			get_node(index).state = valid;
		}

		inline bool try_grow()
		{
			uint64_t n = m_values.size() * 2ULL;

			if (!helpers::is_power_of_2(n))
				n = helpers::next_pow2(n);

			if (!can_fit_into_size_t(n))
			{
				assert(0);
				return false;
			}

			return rehash(helpers::maximum<size_t>(cMinHashSize, (size_t)n));
		}

		// new_hash_size must be a power of 2.
		inline bool rehash(size_t new_hash_size)
		{
			if (!helpers::is_power_of_2((uint64_t)new_hash_size))
			{
				assert(0);
				return false;
			}

			if (new_hash_size < m_num_valid)
			{
				assert(0);
				return false;
			}

			if (new_hash_size == m_values.size())
				return true;

			hash_map new_map;
			if (!new_map.m_values.try_resize(new_hash_size))
				return false;

			new_map.m_hash_shift = SIZE_T_BITS - helpers::floor_log2i((uint64_t)new_hash_size);
			assert(new_hash_size == safe_shift_left(static_cast<uint64_t>(1), SIZE_T_BITS - new_map.m_hash_shift));

			new_map.m_grow_threshold = std::numeric_limits<size_t>::max();

			node* pNode = reinterpret_cast<node*>(m_values.begin());
			node* pNode_end = pNode + m_values.size();

			while (pNode != pNode_end)
			{
				if (pNode->state)
				{
					new_map.move_into(pNode);

					if (new_map.m_num_valid == m_num_valid)
						break;
				}

				pNode++;
			}

			new_map.m_grow_threshold = new_hash_size >> 1U;
			if (new_hash_size & 1)
				new_map.m_grow_threshold++;

			m_values.clear_no_destruction();
			m_hash_shift = SIZE_T_BITS;

			swap(new_map);

			return true;
		}

		inline size_t find_next(size_t index) const
		{
			index++;

			if (index >= m_values.size())
				return index;

			const node* pNode = &get_node(index);

			for (; ; )
			{
				if (pNode->state)
					break;

				if (++index >= m_values.size())
					break;

				pNode++;
			}

			return index;
		}

		inline size_t find_index(const Key& k) const
		{
			if (m_num_valid)
			{
				size_t index = hash_key(k);
				const node* pNode = &get_node(index);

				if (pNode->state)
				{
					if (m_equals(pNode->first, k))
						return index;

					const size_t orig_index = index;

					for (; ; )
					{
						if (!index)
						{
							index = m_values.size() - 1;
							pNode = &get_node(index);
						}
						else
						{
							index--;
							pNode--;
						}

						if (index == orig_index)
							break;

						if (!pNode->state)
							break;

						if (m_equals(pNode->first, k))
							return index;
					}
				}
			}

			return m_values.size();
		}

		inline bool insert_no_grow(insert_result& result, const Key& k, const Value& v)
		{
			if (!m_values.size())
				return false;

			size_t index = hash_key(k);
			node* pNode = &get_node(index);

			if (pNode->state)
			{
				if (m_equals(pNode->first, k))
				{
					result.first = iterator(*this, index);
					result.second = false;
					return true;
				}

				const size_t orig_index = index;

				for (; ; )
				{
					if (!index)
					{
						index = m_values.size() - 1;
						pNode = &get_node(index);
					}
					else
					{
						index--;
						pNode--;
					}

					if (orig_index == index)
						return false;

					if (!pNode->state)
						break;

					if (m_equals(pNode->first, k))
					{
						result.first = iterator(*this, index);
						result.second = false;
						return true;
					}
				}
			}

			if (m_num_valid >= m_grow_threshold)
				return false;

			construct_value_type(pNode, k, v);

			pNode->state = cStateValid;

			m_num_valid++;
			assert(m_num_valid <= m_values.size());

			result.first = iterator(*this, index);
			result.second = true;

			return true;
		}

		// Move user supplied key/value into a node.
		static inline void move_value_type(value_type* pDst, Key&& k, Value&& v)
		{
			// Not checking for is MOVABLE because the caller could later destruct k and/or v (what state do we set them to?)
			if (BASISU_IS_BITWISE_COPYABLE(Key))
			{
				memcpy(&pDst->first, &k, sizeof(Key));
			}
			else
			{
				new ((void*)&pDst->first) Key(std::move(k));
				// No destruction - user will do that (we don't own k).
			}

			if (BASISU_IS_BITWISE_COPYABLE(Value))
			{
				memcpy(&pDst->second, &v, sizeof(Value));
			}
			else
			{
				new ((void*)&pDst->second) Value(std::move(v));
				// No destruction - user will do that (we don't own v).
			}
		}

		// Insert user provided k/v, by moving, into the current hash table
		inline bool insert_no_grow_move(insert_result& result, Key&& k, Value&& v)
		{
			if (!m_values.size())
				return false;

			size_t index = hash_key(k);
			node* pNode = &get_node(index);

			if (pNode->state)
			{
				if (m_equals(pNode->first, k))
				{
					result.first = iterator(*this, index);
					result.second = false;
					return true;
				}

				const size_t orig_index = index;

				for (; ; )
				{
					if (!index)
					{
						index = m_values.size() - 1;
						pNode = &get_node(index);
					}
					else
					{
						index--;
						pNode--;
					}

					if (orig_index == index)
						return false;

					if (!pNode->state)
						break;

					if (m_equals(pNode->first, k))
					{
						result.first = iterator(*this, index);
						result.second = false;
						return true;
					}
				}
			}

			if (m_num_valid >= m_grow_threshold)
				return false;

			move_value_type(pNode, std::move(k), std::move(v));

			pNode->state = cStateValid;

			m_num_valid++;
			assert(m_num_valid <= m_values.size());

			result.first = iterator(*this, index);
			result.second = true;

			return true;
		}

		// Insert pNode by moving into the current hash table
		inline void move_into(node* pNode)
		{
			size_t index = hash_key(pNode->first);
			node* pDst_node = &get_node(index);

			if (pDst_node->state)
			{
				const size_t orig_index = index;

				for (; ; )
				{
					if (!index)
					{
						index = m_values.size() - 1;
						pDst_node = &get_node(index);
					}
					else
					{
						index--;
						pDst_node--;
					}

					if (index == orig_index)
					{
						assert(false);
						return;
					}

					if (!pDst_node->state)
						break;
				}
			}

			// No need to update the source node's state (it's going away)
			move_node(pDst_node, pNode, false);

			m_num_valid++;
		}
	};

	template<typename Key, typename Value, typename Hasher, typename Equals>
	struct bitwise_movable< hash_map<Key, Value, Hasher, Equals> > { enum { cFlag = true }; };

#if BASISU_HASHMAP_TEST
	extern void hash_map_test();
#endif

	// String formatting
	inline std::string string_format(const char* pFmt, ...)
	{
		char buf[2048];

		va_list args;
		va_start(args, pFmt);
#ifdef _WIN32		
		vsprintf_s(buf, sizeof(buf), pFmt, args);
#else
		vsnprintf(buf, sizeof(buf), pFmt, args);
#endif		
		va_end(args);

		return std::string(buf);
	}

	enum class variant_type
	{
		cInvalid,
		cI32, cU32,
		cI64, cU64,
		cFlt, cDbl, cBool,
		cStrPtr, cStdStr
	};

	struct fmt_variant
	{
		union
		{
			int32_t m_i32;
			uint32_t m_u32;
			int64_t m_i64;
			uint64_t m_u64;
			float m_flt;
			double m_dbl;
			bool m_bool;
			const char* m_pStr;
		};

		std::string m_str;

		variant_type m_type;

		inline fmt_variant() :
			m_u64(0),
			m_type(variant_type::cInvalid)
		{
		}

		inline fmt_variant(const fmt_variant& other) :
			m_u64(other.m_u64),
			m_str(other.m_str),
			m_type(other.m_type)
		{
		}

		inline fmt_variant(fmt_variant&& other) :
			m_u64(other.m_u64),
			m_str(std::move(other.m_str)),
			m_type(other.m_type)
		{
			other.m_type = variant_type::cInvalid;
			other.m_u64 = 0;
		}

		inline fmt_variant& operator= (fmt_variant&& other)
		{
			if (this == &other)
				return *this;

			m_type = other.m_type;
			m_u64 = other.m_u64;
			m_str = std::move(other.m_str);

			other.m_type = variant_type::cInvalid;
			other.m_u64 = 0;

			return *this;
		}

		inline fmt_variant& operator= (const fmt_variant& rhs)
		{
			if (this == &rhs)
				return *this;

			m_u64 = rhs.m_u64;
			m_type = rhs.m_type;
			m_str = rhs.m_str;

			return *this;
		}

		inline fmt_variant(int32_t v) : m_i32(v), m_type(variant_type::cI32) { }
		inline fmt_variant(uint32_t v) : m_u32(v), m_type(variant_type::cU32) { }
		inline fmt_variant(int64_t v) : m_i64(v), m_type(variant_type::cI64) { }
		inline fmt_variant(uint64_t v) : m_u64(v), m_type(variant_type::cU64) { }
#ifdef _MSC_VER
		inline fmt_variant(unsigned long v) : m_u64(v), m_type(variant_type::cU64) {}
		inline fmt_variant(long v) : m_i64(v), m_type(variant_type::cI64) {}
#endif
		inline fmt_variant(float v) : m_flt(v), m_type(variant_type::cFlt) { }
		inline fmt_variant(double v) : m_dbl(v), m_type(variant_type::cDbl) { }
		inline fmt_variant(const char* pStr) : m_pStr(pStr), m_type(variant_type::cStrPtr) { }
		inline fmt_variant(const std::string& str) : m_u64(0), m_str(str), m_type(variant_type::cStdStr) { }
		inline fmt_variant(bool val) : m_bool(val), m_type(variant_type::cBool) { }

		bool to_string(std::string& res, std::string& fmt) const;
	};

	typedef basisu::vector<fmt_variant> fmt_variant_vec;

	bool fmt_variants(std::string& res, const char* pFmt, const fmt_variant_vec& variants);

	template <typename... Args>
	inline bool fmt_string(std::string& res, const char* pFmt, Args&&... args)
	{
		return fmt_variants(res, pFmt, fmt_variant_vec{ fmt_variant(std::forward<Args>(args))... });
	}

	template <typename... Args>
	inline std::string fmt_string(const char* pFmt, Args&&... args)
	{
		std::string res;
		fmt_variants(res, pFmt, fmt_variant_vec{ fmt_variant(std::forward<Args>(args))... });
		return res;
	}

	template <typename... Args>
	inline int fmt_printf(const char* pFmt, Args&&... args)
	{
		std::string res;
		if (!fmt_variants(res, pFmt, fmt_variant_vec{ fmt_variant(std::forward<Args>(args))... }))
			return EOF;

		return fputs(res.c_str(), stdout);
	}

	template <typename... Args>
	inline int fmt_fprintf(FILE* pFile, const char* pFmt, Args&&... args)
	{
		std::string res;
		if (!fmt_variants(res, pFmt, fmt_variant_vec{ fmt_variant(std::forward<Args>(args))... }))
			return EOF;

		return fputs(res.c_str(), pFile);
	}

	// fixed_array - zero initialized by default, operator[] is always bounds checked.
    template <std::size_t N, typename T>
    class fixed_array
    {
		static_assert(N >= 1, "fixed_array size must be at least 1");

    public:
		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using const_reference = const T&;
		using pointer = T*;
		using const_pointer = const T*;
		using iterator = T*;
		using const_iterator = const T*;

        T m_data[N];

		BASISU_FORCE_INLINE fixed_array()
		{
            initialize_array();
        }

		BASISU_FORCE_INLINE fixed_array(std::initializer_list<T> list)
		{
			assert(list.size() <= N);

            std::size_t copy_size = std::min(list.size(), N);
            std::copy_n(list.begin(), copy_size, m_data);  // Copy up to min(list.size(), N)

            if (list.size() < N) 
			{
                // Initialize the rest of the array
                std::fill(m_data + copy_size, m_data + N, T{});
            }
        }

		BASISU_FORCE_INLINE T& operator[](std::size_t index)
		{
			if (index >= N)
				container_abort("fixed_array: Index out of bounds.");
			return m_data[index];
        }

        BASISU_FORCE_INLINE const T& operator[](std::size_t index) const 
		{
			if (index >= N)
				container_abort("fixed_array: Index out of bounds.");
			return m_data[index];
        }

		BASISU_FORCE_INLINE T* begin() { return m_data; }
		BASISU_FORCE_INLINE const T* begin() const { return m_data; }

		BASISU_FORCE_INLINE T* end() { return m_data + N; }
		BASISU_FORCE_INLINE const T* end() const { return m_data + N; }

		BASISU_FORCE_INLINE const T* data() const { return m_data; }
		BASISU_FORCE_INLINE T* data() { return m_data; }

		BASISU_FORCE_INLINE const T& front() const { return m_data[0]; }
		BASISU_FORCE_INLINE T& front() { return m_data[0]; }

		BASISU_FORCE_INLINE const T& back() const { return m_data[N - 1]; }
		BASISU_FORCE_INLINE T& back() { return m_data[N - 1]; }

		BASISU_FORCE_INLINE constexpr std::size_t size() const { return N; }

		BASISU_FORCE_INLINE void clear()
		{
            initialize_array();  // Reinitialize the array
        }

		BASISU_FORCE_INLINE void set_all(const T& value)
		{
            std::fill(m_data, m_data + N, value);
        }

		BASISU_FORCE_INLINE readable_span<T> get_readable_span() const
		{
			return readable_span<T>(m_data, N);
		}

		BASISU_FORCE_INLINE writable_span<T> get_writable_span()
		{
			return writable_span<T>(m_data, N);
		}
				
    private:
		BASISU_FORCE_INLINE void initialize_array()
		{
            if constexpr (std::is_integral<T>::value || std::is_floating_point<T>::value) 
                memset(m_data, 0, sizeof(m_data));
            else 
                std::fill(m_data, m_data + N, T{});
        }

		BASISU_FORCE_INLINE T& access_element(std::size_t index)
		{
            if (index >= N) 
				container_abort("fixed_array: Index out of bounds.");
            return m_data[index];
        }

		BASISU_FORCE_INLINE const T& access_element(std::size_t index) const
		{
            if (index >= N) 
				container_abort("fixed_array: Index out of bounds.");
            return m_data[index];
        }
    };

	// 2D array

	template<typename T>
	class vector2D
	{
		typedef basisu::vector<T> vec_type;

		uint32_t m_width, m_height;
		vec_type m_values;

	public:
		vector2D() :
			m_width(0),
			m_height(0)
		{
		}

		vector2D(uint32_t w, uint32_t h) :
			m_width(0),
			m_height(0)
		{
			resize(w, h);
		}

		vector2D(const vector2D& other)
		{
			*this = other;
		}

		vector2D(vector2D&& other) :
			m_width(0),
			m_height(0)
		{
			*this = std::move(other);
		}

		vector2D& operator= (const vector2D& other)
		{
			if (this != &other)
			{
				m_width = other.m_width;
				m_height = other.m_height;
				m_values = other.m_values;
			}
			return *this;
		}

		vector2D& operator= (vector2D&& other)
		{
			if (this != &other)
			{
				m_width = other.m_width;
				m_height = other.m_height;
				m_values = std::move(other.m_values);

				other.m_width = 0;
				other.m_height = 0;
			}
			return *this;
		}

		inline bool operator== (const vector2D& rhs) const
		{
			return (m_width == rhs.m_width) && (m_height == rhs.m_height) && (m_values == rhs.m_values);
		}

		inline size_t size_in_bytes() const { return m_values.size_in_bytes(); }

		inline uint32_t get_width() const { return m_width; }
		inline uint32_t get_height() const { return m_height; }

		inline const T& operator() (uint32_t x, uint32_t y) const { assert(x < m_width && y < m_height); return m_values[x + y * m_width]; }
		inline T& operator() (uint32_t x, uint32_t y) { assert(x < m_width && y < m_height); return m_values[x + y * m_width]; }

		inline size_t size() const { return m_values.size(); }

		inline const T& operator[] (uint32_t i) const { return m_values[i]; }
		inline T& operator[] (uint32_t i) { return m_values[i]; }

		inline const T& at_clamped(int x, int y) const { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }
		inline T& at_clamped(int x, int y) { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }

		void clear()
		{
			m_width = 0;
			m_height = 0;
			m_values.clear();
		}

		void set_all(const T& val)
		{
			vector_set_all(m_values, val);
		}

		inline const T* get_ptr() const { return m_values.data(); }
		inline T* get_ptr() { return m_values.data(); }

		vector2D& resize(uint32_t new_width, uint32_t new_height)
		{
			if ((m_width == new_width) && (m_height == new_height))
				return *this;

			const uint64_t total_vals = (uint64_t)new_width * new_height;

			if (!can_fit_into_size_t(total_vals))
			{
				// What can we do?
				assert(0);
				return *this;
			}

			vec_type oldVals((size_t)total_vals);
			oldVals.swap(m_values);

			const uint32_t w = minimum(m_width, new_width);
			const uint32_t h = minimum(m_height, new_height);

			if ((w) && (h))
			{
				for (uint32_t y = 0; y < h; y++)
					for (uint32_t x = 0; x < w; x++)
						m_values[x + y * new_width] = oldVals[x + y * m_width];
			}

			m_width = new_width;
			m_height = new_height;

			return *this;
		}

		bool try_resize(uint32_t new_width, uint32_t new_height)
		{
			if ((m_width == new_width) && (m_height == new_height))
				return true;

			const uint64_t total_vals = (uint64_t)new_width * new_height;

			if (!can_fit_into_size_t(total_vals))
			{
				// What can we do?
				assert(0);
				return false;
			}

			vec_type oldVals;
			if (!oldVals.try_resize((size_t)total_vals))
				return false;

			oldVals.swap(m_values);

			const uint32_t w = minimum(m_width, new_width);
			const uint32_t h = minimum(m_height, new_height);

			if ((w) && (h))
			{
				for (uint32_t y = 0; y < h; y++)
					for (uint32_t x = 0; x < w; x++)
						m_values[x + y * new_width] = oldVals[x + y * m_width];
			}

			m_width = new_width;
			m_height = new_height;

			return true;
		}

		const vector2D& extract_block_clamped(T* pDst, uint32_t src_x, uint32_t src_y, uint32_t w, uint32_t h) const
		{
			// HACK HACK
			if (((src_x + w) > m_width) || ((src_y + h) > m_height))
			{
				// Slower clamping case
				for (uint32_t y = 0; y < h; y++)
					for (uint32_t x = 0; x < w; x++)
						*pDst++ = at_clamped(src_x + x, src_y + y);
			}
			else
			{
				const T* pSrc = &m_values[src_x + src_y * m_width];

				for (uint32_t y = 0; y < h; y++)
				{
					memcpy(pDst, pSrc, w * sizeof(T));
					pSrc += m_width;
					pDst += w;
				}
			}

			return *this;
		}
	};
		
} // namespace basisu

namespace std
{
	template<typename T>
	inline void swap(basisu::vector<T>& a, basisu::vector<T>& b)
	{
		a.swap(b);
	}

	template<typename Key, typename Value, typename Hasher, typename Equals>
	inline void swap(basisu::hash_map<Key, Value, Hasher, Equals>& a, basisu::hash_map<Key, Value, Hasher, Equals>& b)
	{
		a.swap(b);
	}

} // namespace std
