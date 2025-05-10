// File: basisu_math.h
#pragma once

// TODO: Would prefer this in the basisu namespace, but to avoid collisions with the existing vec/matrix classes I'm placing this in "bu_math".
namespace bu_math
{
	// Cross-platform 1.0f/sqrtf(x) approximation. See https://en.wikipedia.org/wiki/Fast_inverse_square_root#cite_note-37.
	// Would prefer using SSE1 etc. but that would require implementing multiple versions and platform divergence (needing more testing).
	BASISU_FORCE_INLINE float inv_sqrt(float v)
	{
		union 
		{ 
			float flt; 
			uint32_t ui; 
		} un;

		un.flt = v;
		un.ui = 0x5F1FFFF9UL - (un.ui >> 1);

		return 0.703952253f * un.flt * (2.38924456f - v * (un.flt * un.flt));
	}

	inline float smoothstep(float edge0, float edge1, float x)
	{
		assert(edge1 != edge0);

		// Scale, and clamp x to 0..1 range
		x = basisu::saturate((x - edge0) / (edge1 - edge0));

		return x * x * (3.0f - 2.0f * x);
	}

	template <uint32_t N, typename T>
	class vec : public basisu::rel_ops<vec<N, T> >
	{
	public:
		typedef T scalar_type;
		enum
		{
			num_elements = N
		};

		inline vec()
		{
		}

		inline vec(basisu::eClear)
		{
			clear();
		}

		inline vec(const vec& other)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = other.m_s[i];
		}

		template <uint32_t O, typename U>
		inline vec(const vec<O, U>& other)
		{
			set(other);
		}

		template <uint32_t O, typename U>
		inline vec(const vec<O, U>& other, T w)
		{
			*this = other;
			m_s[N - 1] = w;
		}

		template <typename... Args>
		inline explicit vec(Args... args)
		{
			static_assert(sizeof...(args) <= N);
			set(args...);
		}

		inline void clear()
		{
			if (N > 4)
				memset(m_s, 0, sizeof(m_s));
			else
			{
				for (uint32_t i = 0; i < N; i++)
					m_s[i] = 0;
			}
		}

		template <uint32_t ON, typename OT>
		inline vec& set(const vec<ON, OT>& other)
		{
			if ((void*)this == (void*)&other)
				return *this;
			const uint32_t m = basisu::minimum(N, ON);
			uint32_t i;
			for (i = 0; i < m; i++)
				m_s[i] = static_cast<T>(other[i]);
			for (; i < N; i++)
				m_s[i] = 0;
			return *this;
		}

		inline vec& set_component(uint32_t index, T val)
		{
			assert(index < N);
			m_s[index] = val;
			return *this;
		}

		inline vec& set_all(T val)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = val;
			return *this;
		}

		template <typename... Args>
		inline vec& set(Args... args)
		{
			static_assert(sizeof...(args) <= N);

			// Initialize using parameter pack expansion
			T values[] = { static_cast<T>(args)... };

			// Special case if setting with a scalar
			if (sizeof...(args) == 1)
			{
				set_all(values[0]);
			}
			else
			{
				// Copy the values into the vector
				for (std::size_t i = 0; i < sizeof...(args); ++i)
				{
					m_s[i] = values[i];
				}

				// Zero-initialize the remaining elements (if any)
				if (sizeof...(args) < N)
				{
					std::fill(m_s + sizeof...(args), m_s + N, T{});
				}
			}

			return *this;
		}

		inline vec& set(const T* pValues)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = pValues[i];
			return *this;
		}

		template <uint32_t ON, typename OT>
		inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i)
		{
			return set(static_cast<T>(other[i]));
		}

		template <uint32_t ON, typename OT>
		inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j)
		{
			return set(static_cast<T>(other[i]), static_cast<T>(other[j]));
		}

		template <uint32_t ON, typename OT>
		inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j, uint32_t k)
		{
			return set(static_cast<T>(other[i]), static_cast<T>(other[j]), static_cast<T>(other[k]));
		}

		template <uint32_t ON, typename OT>
		inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j, uint32_t k, uint32_t l)
		{
			return set(static_cast<T>(other[i]), static_cast<T>(other[j]), static_cast<T>(other[k]), static_cast<T>(other[l]));
		}

		inline vec& operator=(const vec& rhs)
		{
			if (this != &rhs)
			{
				for (uint32_t i = 0; i < N; i++)
					m_s[i] = rhs.m_s[i];
			}
			return *this;
		}

		template <uint32_t O, typename U>
		inline vec& operator=(const vec<O, U>& other)
		{
			if ((void*)this == (void*)&other)
				return *this;

			uint32_t s = basisu::minimum(N, O);

			uint32_t i;
			for (i = 0; i < s; i++)
				m_s[i] = static_cast<T>(other[i]);

			for (; i < N; i++)
				m_s[i] = 0;

			return *this;
		}

		inline bool operator==(const vec& rhs) const
		{
			for (uint32_t i = 0; i < N; i++)
				if (!(m_s[i] == rhs.m_s[i]))
					return false;
			return true;
		}

		inline bool operator<(const vec& rhs) const
		{
			for (uint32_t i = 0; i < N; i++)
			{
				if (m_s[i] < rhs.m_s[i])
					return true;
				else if (!(m_s[i] == rhs.m_s[i]))
					return false;
			}

			return false;
		}

		inline T operator[](uint32_t i) const
		{
			assert(i < N);
			return m_s[i];
		}

		inline T& operator[](uint32_t i)
		{
			assert(i < N);
			return m_s[i];
		}

		template <uint32_t index>
		inline uint64_t get_component_bits_as_uint() const
		{
			static_assert(index < N);
			static_assert((sizeof(T) == sizeof(uint16_t)) || (sizeof(T) == sizeof(uint32_t)) || (sizeof(T) == sizeof(uint64_t)), "Unsupported type");

			if (sizeof(T) == sizeof(uint16_t))
				return *reinterpret_cast<const uint16_t*>(&m_s[index]);
			else if (sizeof(T) == sizeof(uint32_t))
				return *reinterpret_cast<const uint32_t*>(&m_s[index]);
			else if (sizeof(T) == sizeof(uint64_t))
				return *reinterpret_cast<const uint64_t*>(&m_s[index]);
			else
			{
				assert(0);
				return 0;
			}
		}

		inline T get_x(void) const
		{
			return m_s[0];
		}
		inline T get_y(void) const
		{
			static_assert(N >= 2);
			return m_s[1];
		}
		inline T get_z(void) const
		{
			static_assert(N >= 3);
			return m_s[2];
		}
		inline T get_w(void) const
		{
			static_assert(N >= 4);
			return m_s[3];
		}

		inline vec get_x_vector() const
		{
			return broadcast<0>();
		}
		inline vec get_y_vector() const
		{
			return broadcast<1>();
		}
		inline vec get_z_vector() const
		{
			return broadcast<2>();
		}
		inline vec get_w_vector() const
		{
			return broadcast<3>();
		}

		inline T get_component(uint32_t i) const
		{
			return (*this)[i];
		}

		inline vec& set_x(T v)
		{
			m_s[0] = v;
			return *this;
		}
		inline vec& set_y(T v)
		{
			static_assert(N >= 2);
			m_s[1] = v;
			return *this;
		}
		inline vec& set_z(T v)
		{
			static_assert(N >= 3);
			m_s[2] = v;
			return *this;
		}
		inline vec& set_w(T v)
		{
			static_assert(N >= 4);
			m_s[3] = v;
			return *this;
		}

		inline const T* get_ptr() const
		{
			return reinterpret_cast<const T*>(&m_s[0]);
		}
		inline T* get_ptr()
		{
			return reinterpret_cast<T*>(&m_s[0]);
		}

		inline vec as_point() const
		{
			vec result(*this);
			result[N - 1] = 1;
			return result;
		}

		inline vec as_dir() const
		{
			vec result(*this);
			result[N - 1] = 0;
			return result;
		}

		inline vec<2, T> select2(uint32_t i, uint32_t j) const
		{
			assert((i < N) && (j < N));
			return vec<2, T>(m_s[i], m_s[j]);
		}

		inline vec<3, T> select3(uint32_t i, uint32_t j, uint32_t k) const
		{
			assert((i < N) && (j < N) && (k < N));
			return vec<3, T>(m_s[i], m_s[j], m_s[k]);
		}

		inline vec<4, T> select4(uint32_t i, uint32_t j, uint32_t k, uint32_t l) const
		{
			assert((i < N) && (j < N) && (k < N) && (l < N));
			return vec<4, T>(m_s[i], m_s[j], m_s[k], m_s[l]);
		}

		inline bool is_dir() const
		{
			return m_s[N - 1] == 0;
		}
		inline bool is_vector() const
		{
			return is_dir();
		}
		inline bool is_point() const
		{
			return m_s[N - 1] == 1;
		}

		inline vec project() const
		{
			vec result(*this);
			if (result[N - 1])
				result /= result[N - 1];
			return result;
		}

		inline vec broadcast(unsigned i) const
		{
			return vec((*this)[i]);
		}

		template <uint32_t i>
		inline vec broadcast() const
		{
			return vec((*this)[i]);
		}

		inline vec swizzle(uint32_t i, uint32_t j) const
		{
			return vec((*this)[i], (*this)[j]);
		}

		inline vec swizzle(uint32_t i, uint32_t j, uint32_t k) const
		{
			return vec((*this)[i], (*this)[j], (*this)[k]);
		}

		inline vec swizzle(uint32_t i, uint32_t j, uint32_t k, uint32_t l) const
		{
			return vec((*this)[i], (*this)[j], (*this)[k], (*this)[l]);
		}

		inline vec operator-() const
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = -m_s[i];
			return result;
		}

		inline vec operator+() const
		{
			return *this;
		}

		inline vec& operator+=(const vec& other)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] += other.m_s[i];
			return *this;
		}

		inline vec& operator-=(const vec& other)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] -= other.m_s[i];
			return *this;
		}

		inline vec& operator*=(const vec& other)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] *= other.m_s[i];
			return *this;
		}

		inline vec& operator/=(const vec& other)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] /= other.m_s[i];
			return *this;
		}

		inline vec& operator*=(T s)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] *= s;
			return *this;
		}

		inline vec& operator/=(T s)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] /= s;
			return *this;
		}

		friend inline vec operator*(const vec& lhs, T val)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = lhs.m_s[i] * val;
			return result;
		}

		friend inline vec operator*(T val, const vec& rhs)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = val * rhs.m_s[i];
			return result;
		}

		friend inline vec operator/(const vec& lhs, const vec& rhs)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = lhs.m_s[i] / rhs.m_s[i];
			return result;
		}

		friend inline vec operator/(const vec& lhs, T val)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = lhs.m_s[i] / val;
			return result;
		}

		friend inline vec operator+(const vec& lhs, const vec& rhs)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = lhs.m_s[i] + rhs.m_s[i];
			return result;
		}

		friend inline vec operator-(const vec& lhs, const vec& rhs)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result.m_s[i] = lhs.m_s[i] - rhs.m_s[i];
			return result;
		}

		static inline vec<3, T> cross2(const vec& a, const vec& b)
		{
			static_assert(N >= 2);
			return vec<3, T>(0, 0, a[0] * b[1] - a[1] * b[0]);
		}

		inline vec<3, T> cross2(const vec& b) const
		{
			return cross2(*this, b);
		}

		static inline vec<3, T> cross3(const vec& a, const vec& b)
		{
			static_assert(N >= 3);
			return vec<3, T>(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
		}

		inline vec<3, T> cross3(const vec& b) const
		{
			return cross3(*this, b);
		}

		static inline vec<3, T> cross(const vec& a, const vec& b)
		{
			static_assert(N >= 2);

			if (N == 2)
				return cross2(a, b);
			else
				return cross3(a, b);
		}

		inline vec<3, T> cross(const vec& b) const
		{
			static_assert(N >= 2);
			return cross(*this, b);
		}

		inline T dot(const vec& rhs) const
		{
			return dot(*this, rhs);
		}

		inline vec dot_vector(const vec& rhs) const
		{
			return vec(dot(*this, rhs));
		}

		static inline T dot(const vec& lhs, const vec& rhs)
		{
			T result = lhs.m_s[0] * rhs.m_s[0];
			for (uint32_t i = 1; i < N; i++)
				result += lhs.m_s[i] * rhs.m_s[i];
			return result;
		}

		inline T dot2(const vec& rhs) const
		{
			static_assert(N >= 2);
			return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1];
		}

		inline T dot3(const vec& rhs) const
		{
			static_assert(N >= 3);
			return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1] + m_s[2] * rhs.m_s[2];
		}

		inline T dot4(const vec& rhs) const
		{
			static_assert(N >= 4);
			return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1] + m_s[2] * rhs.m_s[2] + m_s[3] * rhs.m_s[3];
		}

		inline T norm(void) const
		{
			T sum = m_s[0] * m_s[0];
			for (uint32_t i = 1; i < N; i++)
				sum += m_s[i] * m_s[i];
			return sum;
		}

		inline T length(void) const
		{
			return sqrt(norm());
		}

		inline T squared_distance(const vec& rhs) const
		{
			T dist2 = 0;
			for (uint32_t i = 0; i < N; i++)
			{
				T d = m_s[i] - rhs.m_s[i];
				dist2 += d * d;
			}
			return dist2;
		}

		inline T squared_distance(const vec& rhs, T early_out) const
		{
			T dist2 = 0;
			for (uint32_t i = 0; i < N; i++)
			{
				T d = m_s[i] - rhs.m_s[i];
				dist2 += d * d;
				if (dist2 > early_out)
					break;
			}
			return dist2;
		}

		inline T distance(const vec& rhs) const
		{
			T dist2 = 0;
			for (uint32_t i = 0; i < N; i++)
			{
				T d = m_s[i] - rhs.m_s[i];
				dist2 += d * d;
			}
			return sqrt(dist2);
		}

		inline vec inverse() const
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = m_s[i] ? (1.0f / m_s[i]) : 0;
			return result;
		}

		// returns squared length (norm)
		inline double normalize(const vec* pDefaultVec = NULL)
		{
			double n = m_s[0] * m_s[0];
			for (uint32_t i = 1; i < N; i++)
				n += m_s[i] * m_s[i];

			if (n != 0)
				*this *= static_cast<T>(1.0f / sqrt(n));
			else if (pDefaultVec)
				*this = *pDefaultVec;
			return n;
		}

		inline double normalize3(const vec* pDefaultVec = NULL)
		{
			static_assert(N >= 3);

			double n = m_s[0] * m_s[0] + m_s[1] * m_s[1] + m_s[2] * m_s[2];

			if (n != 0)
				*this *= static_cast<T>((1.0f / sqrt(n)));
			else if (pDefaultVec)
				*this = *pDefaultVec;
			return n;
		}

		inline vec& normalize_in_place(const vec* pDefaultVec = NULL)
		{
			normalize(pDefaultVec);
			return *this;
		}

		inline vec& normalize3_in_place(const vec* pDefaultVec = NULL)
		{
			normalize3(pDefaultVec);
			return *this;
		}

		inline vec get_normalized(const vec* pDefaultVec = NULL) const
		{
			vec result(*this);
			result.normalize(pDefaultVec);
			return result;
		}

		inline vec get_normalized3(const vec* pDefaultVec = NULL) const
		{
			vec result(*this);
			result.normalize3(pDefaultVec);
			return result;
		}

		inline vec& clamp(T l, T h)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = static_cast<T>(basisu::clamp(m_s[i], l, h));
			return *this;
		}

		inline vec& saturate()
		{
			return clamp(0.0f, 1.0f);
		}

		inline vec& clamp(const vec& l, const vec& h)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = static_cast<T>(basisu::clamp(m_s[i], l[i], h[i]));
			return *this;
		}

		inline bool is_within_bounds(const vec& l, const vec& h) const
		{
			for (uint32_t i = 0; i < N; i++)
				if ((m_s[i] < l[i]) || (m_s[i] > h[i]))
					return false;

			return true;
		}

		inline bool is_within_bounds(T l, T h) const
		{
			for (uint32_t i = 0; i < N; i++)
				if ((m_s[i] < l) || (m_s[i] > h))
					return false;

			return true;
		}

		inline uint32_t get_major_axis(void) const
		{
			T m = fabs(m_s[0]);
			uint32_t r = 0;
			for (uint32_t i = 1; i < N; i++)
			{
				const T c = fabs(m_s[i]);
				if (c > m)
				{
					m = c;
					r = i;
				}
			}
			return r;
		}

		inline uint32_t get_minor_axis(void) const
		{
			T m = fabs(m_s[0]);
			uint32_t r = 0;
			for (uint32_t i = 1; i < N; i++)
			{
				const T c = fabs(m_s[i]);
				if (c < m)
				{
					m = c;
					r = i;
				}
			}
			return r;
		}

		inline void get_projection_axes(uint32_t& u, uint32_t& v) const
		{
			const int axis = get_major_axis();
			if (m_s[axis] < 0.0f)
			{
				v = basisu::next_wrap<uint32_t>(axis, N);
				u = basisu::next_wrap<uint32_t>(v, N);
			}
			else
			{
				u = basisu::next_wrap<uint32_t>(axis, N);
				v = basisu::next_wrap<uint32_t>(u, N);
			}
		}

		inline T get_absolute_minimum(void) const
		{
			T result = fabs(m_s[0]);
			for (uint32_t i = 1; i < N; i++)
				result = basisu::minimum(result, fabs(m_s[i]));
			return result;
		}

		inline T get_absolute_maximum(void) const
		{
			T result = fabs(m_s[0]);
			for (uint32_t i = 1; i < N; i++)
				result = basisu::maximum(result, fabs(m_s[i]));
			return result;
		}

		inline T get_minimum(void) const
		{
			T result = m_s[0];
			for (uint32_t i = 1; i < N; i++)
				result = basisu::minimum(result, m_s[i]);
			return result;
		}

		inline T get_maximum(void) const
		{
			T result = m_s[0];
			for (uint32_t i = 1; i < N; i++)
				result = basisu::maximum(result, m_s[i]);
			return result;
		}

		inline vec& remove_unit_direction(const vec& dir)
		{
			*this -= (dot(dir) * dir);
			return *this;
		}

		inline vec get_remove_unit_direction(const vec& dir) const
		{
			return *this - (dot(dir) * dir);
		}

		inline bool all_less(const vec& b) const
		{
			for (uint32_t i = 0; i < N; i++)
				if (m_s[i] >= b.m_s[i])
					return false;
			return true;
		}

		inline bool all_less_equal(const vec& b) const
		{
			for (uint32_t i = 0; i < N; i++)
				if (m_s[i] > b.m_s[i])
					return false;
			return true;
		}

		inline bool all_greater(const vec& b) const
		{
			for (uint32_t i = 0; i < N; i++)
				if (m_s[i] <= b.m_s[i])
					return false;
			return true;
		}

		inline bool all_greater_equal(const vec& b) const
		{
			for (uint32_t i = 0; i < N; i++)
				if (m_s[i] < b.m_s[i])
					return false;
			return true;
		}

		inline vec negate_xyz() const
		{
			vec ret;

			ret[0] = -m_s[0];
			if (N >= 2)
				ret[1] = -m_s[1];
			if (N >= 3)
				ret[2] = -m_s[2];

			for (uint32_t i = 3; i < N; i++)
				ret[i] = m_s[i];

			return ret;
		}

		inline vec& invert()
		{
			for (uint32_t i = 0; i < N; i++)
				if (m_s[i] != 0.0f)
					m_s[i] = 1.0f / m_s[i];
			return *this;
		}

		inline scalar_type perp_dot(const vec& b) const
		{
			static_assert(N == 2);
			return m_s[0] * b.m_s[1] - m_s[1] * b.m_s[0];
		}

		inline vec perp() const
		{
			static_assert(N == 2);
			return vec(-m_s[1], m_s[0]);
		}

		inline vec get_floor() const
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = floor(m_s[i]);
			return result;
		}

		inline vec get_ceil() const
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = ceil(m_s[i]);
			return result;
		}

		inline T get_total() const
		{
			T res = m_s[0];
			for (uint32_t i = 1; i < N; i++)
				res += m_s[i];
			return res;
		}

		// static helper methods

		static inline vec mul_components(const vec& lhs, const vec& rhs)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = lhs.m_s[i] * rhs.m_s[i];
			return result;
		}

		static inline vec mul_add_components(const vec& a, const vec& b, const vec& c)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = a.m_s[i] * b.m_s[i] + c.m_s[i];
			return result;
		}

		static inline vec make_axis(uint32_t i)
		{
			vec result;
			result.clear();
			result[i] = 1;
			return result;
		}

		static inline vec equals_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] == b[i]);
			return ret;
		}

		static inline vec not_equals_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] != b[i]);
			return ret;
		}

		static inline vec less_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] < b[i]);
			return ret;
		}

		static inline vec less_equals_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] <= b[i]);
			return ret;
		}

		static inline vec greater_equals_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] >= b[i]);
			return ret;
		}

		static inline vec greater_mask(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret[i] = (a[i] > b[i]);
			return ret;
		}

		static inline vec component_max(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret.m_s[i] = basisu::maximum(a.m_s[i], b.m_s[i]);
			return ret;
		}

		static inline vec component_min(const vec& a, const vec& b)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret.m_s[i] = basisu::minimum(a.m_s[i], b.m_s[i]);
			return ret;
		}

		static inline vec lerp(const vec& a, const vec& b, float t)
		{
			vec ret;
			for (uint32_t i = 0; i < N; i++)
				ret.m_s[i] = a.m_s[i] + (b.m_s[i] - a.m_s[i]) * t;
			return ret;
		}

		static inline bool equal_tol(const vec& a, const vec& b, float t)
		{
			for (uint32_t i = 0; i < N; i++)
				if (!basisu::equal_tol(a.m_s[i], b.m_s[i], t))
					return false;
			return true;
		}

		inline bool equal_tol(const vec& b, float t) const
		{
			return equal_tol(*this, b, t);
		}

		static inline vec make_random(basisu::rand& r, float l, float h)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = r.frand(l, h);
			return result;
		}

		static inline vec make_random(basisu::rand& r, const vec& l, const vec& h)
		{
			vec result;
			for (uint32_t i = 0; i < N; i++)
				result[i] = r.frand(l[i], h[i]);
			return result;
		}

		void print() const
		{
			for (uint32_t c = 0; c < N; c++)
				printf("%3.3f ", (*this)[c]);
			printf("\n");
		}

	protected:
		T m_s[N];
	};

	typedef vec<1, double> vec1D;
	typedef vec<2, double> vec2D;
	typedef vec<3, double> vec3D;
	typedef vec<4, double> vec4D;

	typedef vec<1, float> vec1F;

	typedef vec<2, float> vec2F;
	typedef basisu::vector<vec2F> vec2F_array;

	typedef vec<3, float> vec3F;
	typedef basisu::vector<vec3F> vec3F_array;

	typedef vec<4, float> vec4F;
	typedef basisu::vector<vec4F> vec4F_array;

	typedef vec<2, uint32_t> vec2U;
	typedef vec<3, uint32_t> vec3U;
	typedef vec<2, int> vec2I;
	typedef vec<3, int> vec3I;
	typedef vec<4, int> vec4I;

	typedef vec<2, int16_t> vec2I16;
	typedef vec<3, int16_t> vec3I16;

	inline vec2F rotate_point_2D(const vec2F& p, float rad)
	{
		float c = cosf(rad);
		float s = sinf(rad);

		float x = p[0];
		float y = p[1];

		return vec2F(x * c - y * s, x * s + y * c);
	}

	//--------------------------------------------------------------

	// Matrix/vector cheat sheet, because confusingly, depending on how matrices are stored in memory people can use opposite definitions of "rows", "cols", etc.
	// See http://www.mindcontrol.org/~hplus/graphics/matrix-layout.html
	//
	// So in this simple row-major general matrix class:
	// matrix=[NumRows][NumCols] or [R][C], i.e. a 3x3 matrix stored in memory will appear as: R0C0, R0C1, R0C2,  R1C0, R1C1, R1C2,  etc.
	// Matrix multiplication: [R0,C0]*[R1,C1]=[R0,C1], C0 must equal R1
	//
	// In this class:
	// A "row vector" type is a vector of size # of matrix cols, 1xC. It's the vector type that is used to store the matrix rows.
	// A "col vector" type is a vector of size # of matrix rows, Rx1. It's a vector type large enough to hold each matrix column.
	//
	// Subrow/col vectors: last component is assumed to be either 0 (a "vector") or 1 (a "point")
	// "subrow vector": vector/point of size # cols-1, 1x(C-1)
	// "subcol vector": vector/point of size # rows-1, (R-1)x1
	//
	// D3D style:
	// vec*matrix, row vector on left (vec dotted against columns)
	// [1,4]*[4,4]=[1,4]
	// abcd * A B C D
	//        A B C D
	//        A B C D
	//        A B C D
	// =      e f g h
	//
	// Now confusingly, in the matrix transform method for vec*matrix below the vector's type is "col_vec", because col_vec will have the proper size for non-square matrices. But the vector on the left is written as row vector, argh.
	//
	//
	// OGL style:
	// matrix*vec, col vector on right (vec dotted against rows):
	// [4,4]*[4,1]=[4,1]
	//
	// A B C D * e = e
	// A B C D   f   f
	// A B C D   g   g
	// A B C D   h   h

	template <class X, class Y, class Z>
	Z& matrix_mul_helper(Z& result, const X& lhs, const Y& rhs)
	{
		static_assert((int)Z::num_rows == (int)X::num_rows);
		static_assert((int)Z::num_cols == (int)Y::num_cols);
		static_assert((int)X::num_cols == (int)Y::num_rows);
		assert(((void*)&result != (void*)&lhs) && ((void*)&result != (void*)&rhs));
		for (int r = 0; r < X::num_rows; r++)
			for (int c = 0; c < Y::num_cols; c++)
			{
				typename Z::scalar_type s = lhs(r, 0) * rhs(0, c);
				for (uint32_t i = 1; i < X::num_cols; i++)
					s += lhs(r, i) * rhs(i, c);
				result(r, c) = s;
			}
		return result;
	}

	template <class X, class Y, class Z>
	Z& matrix_mul_helper_transpose_lhs(Z& result, const X& lhs, const Y& rhs)
	{
		static_assert((int)Z::num_rows == (int)X::num_cols);
		static_assert((int)Z::num_cols == (int)Y::num_cols);
		static_assert((int)X::num_rows == (int)Y::num_rows);
		assert(((void*)&result != (void*)&lhs) && ((void*)&result != (void*)&rhs));
		for (int r = 0; r < X::num_cols; r++)
			for (int c = 0; c < Y::num_cols; c++)
			{
				typename Z::scalar_type s = lhs(0, r) * rhs(0, c);
				for (uint32_t i = 1; i < X::num_rows; i++)
					s += lhs(i, r) * rhs(i, c);
				result(r, c) = s;
			}
		return result;
	}

	template <class X, class Y, class Z>
	Z& matrix_mul_helper_transpose_rhs(Z& result, const X& lhs, const Y& rhs)
	{
		static_assert((int)Z::num_rows == (int)X::num_rows);
		static_assert((int)Z::num_cols == (int)Y::num_rows);
		static_assert((int)X::num_cols == (int)Y::num_cols);
		assert(((void*)&result != (void*)&lhs) && ((void*)&result != (void*)&rhs));
		for (int r = 0; r < X::num_rows; r++)
			for (int c = 0; c < Y::num_rows; c++)
			{
				typename Z::scalar_type s = lhs(r, 0) * rhs(c, 0);
				for (uint32_t i = 1; i < X::num_cols; i++)
					s += lhs(r, i) * rhs(c, i);
				result(r, c) = s;
			}
		return result;
	}
		
	template <uint32_t R, uint32_t C, typename T>
	class matrix
	{
	public:
		typedef T scalar_type;
		enum
		{
			num_rows = R,
			num_cols = C
		};

		typedef vec<R, T> col_vec;
		typedef vec < (R > 1) ? (R - 1) : 0, T > subcol_vec;

		typedef vec<C, T> row_vec;
		typedef vec < (C > 1) ? (C - 1) : 0, T > subrow_vec;

		inline matrix()
		{
		}

		inline matrix(basisu::eClear)
		{
			clear();
		}

		inline matrix(basisu::eIdentity)
		{
			set_identity_matrix();
		}

		inline matrix(const T* p)
		{
			set(p);
		}

		inline matrix(const matrix& other)
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i] = other.m_rows[i];
		}

		inline matrix& operator=(const matrix& rhs)
		{
			if (this != &rhs)
				for (uint32_t i = 0; i < R; i++)
					m_rows[i] = rhs.m_rows[i];
			return *this;
		}

		inline matrix(T val00, T val01,
			T val10, T val11)
		{
			set(val00, val01, val10, val11);
		}

		inline matrix(T val00, T val01,
			T val10, T val11,
			T val20, T val21)
		{
			set(val00, val01, val10, val11, val20, val21);
		}

		inline matrix(T val00, T val01, T val02,
			T val10, T val11, T val12,
			T val20, T val21, T val22)
		{
			set(val00, val01, val02, val10, val11, val12, val20, val21, val22);
		}

		inline matrix(T val00, T val01, T val02, T val03,
			T val10, T val11, T val12, T val13,
			T val20, T val21, T val22, T val23,
			T val30, T val31, T val32, T val33)
		{
			set(val00, val01, val02, val03, val10, val11, val12, val13, val20, val21, val22, val23, val30, val31, val32, val33);
		}

		inline matrix(T val00, T val01, T val02, T val03,
			T val10, T val11, T val12, T val13,
			T val20, T val21, T val22, T val23)
		{
			set(val00, val01, val02, val03, val10, val11, val12, val13, val20, val21, val22, val23);
		}

		inline void set(const float* p)
		{
			for (uint32_t i = 0; i < R; i++)
			{
				m_rows[i].set(p);
				p += C;
			}
		}

		inline void set(T val00, T val01,
			T val10, T val11)
		{
			m_rows[0].set(val00, val01);
			if (R >= 2)
			{
				m_rows[1].set(val10, val11);

				for (uint32_t i = 2; i < R; i++)
					m_rows[i].clear();
			}
		}

		inline void set(T val00, T val01,
			T val10, T val11,
			T val20, T val21)
		{
			m_rows[0].set(val00, val01);
			if (R >= 2)
			{
				m_rows[1].set(val10, val11);

				if (R >= 3)
				{
					m_rows[2].set(val20, val21);

					for (uint32_t i = 3; i < R; i++)
						m_rows[i].clear();
				}
			}
		}

		inline void set(T val00, T val01, T val02,
			T val10, T val11, T val12,
			T val20, T val21, T val22)
		{
			m_rows[0].set(val00, val01, val02);
			if (R >= 2)
			{
				m_rows[1].set(val10, val11, val12);
				if (R >= 3)
				{
					m_rows[2].set(val20, val21, val22);

					for (uint32_t i = 3; i < R; i++)
						m_rows[i].clear();
				}
			}
		}

		inline void set(T val00, T val01, T val02, T val03,
			T val10, T val11, T val12, T val13,
			T val20, T val21, T val22, T val23,
			T val30, T val31, T val32, T val33)
		{
			m_rows[0].set(val00, val01, val02, val03);
			if (R >= 2)
			{
				m_rows[1].set(val10, val11, val12, val13);
				if (R >= 3)
				{
					m_rows[2].set(val20, val21, val22, val23);

					if (R >= 4)
					{
						m_rows[3].set(val30, val31, val32, val33);

						for (uint32_t i = 4; i < R; i++)
							m_rows[i].clear();
					}
				}
			}
		}

		inline void set(T val00, T val01, T val02, T val03,
			T val10, T val11, T val12, T val13,
			T val20, T val21, T val22, T val23)
		{
			m_rows[0].set(val00, val01, val02, val03);
			if (R >= 2)
			{
				m_rows[1].set(val10, val11, val12, val13);
				if (R >= 3)
				{
					m_rows[2].set(val20, val21, val22, val23);

					for (uint32_t i = 3; i < R; i++)
						m_rows[i].clear();
				}
			}
		}

		inline uint32_t get_num_rows() const
		{
			return num_rows;
		}

		inline uint32_t get_num_cols() const
		{
			return num_cols;
		}

		inline uint32_t get_total_elements() const
		{
			return num_rows * num_cols;
		}

		inline T operator()(uint32_t r, uint32_t c) const
		{
			assert((r < R) && (c < C));
			return m_rows[r][c];
		}

		inline T& operator()(uint32_t r, uint32_t c)
		{
			assert((r < R) && (c < C));
			return m_rows[r][c];
		}

		inline const row_vec& operator[](uint32_t r) const
		{
			assert(r < R);
			return m_rows[r];
		}

		inline row_vec& operator[](uint32_t r)
		{
			assert(r < R);
			return m_rows[r];
		}

		inline const row_vec& get_row(uint32_t r) const
		{
			return (*this)[r];
		}

		inline row_vec& get_row(uint32_t r)
		{
			return (*this)[r];
		}

		inline void set_row(uint32_t r, const row_vec& v)
		{
			(*this)[r] = v;
		}

		inline col_vec get_col(uint32_t c) const
		{
			assert(c < C);
			col_vec result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = m_rows[i][c];
			return result;
		}

		inline void set_col(uint32_t c, const col_vec& col)
		{
			assert(c < C);
			for (uint32_t i = 0; i < R; i++)
				m_rows[i][c] = col[i];
		}

		inline void set_col(uint32_t c, const subcol_vec& col)
		{
			assert(c < C);
			for (uint32_t i = 0; i < (R - 1); i++)
				m_rows[i][c] = col[i];

			m_rows[R - 1][c] = 0.0f;
		}

		inline const row_vec& get_translate() const
		{
			return m_rows[R - 1];
		}

		inline matrix& set_translate(const row_vec& r)
		{
			m_rows[R - 1] = r;
			return *this;
		}

		inline matrix& set_translate(const subrow_vec& r)
		{
			m_rows[R - 1] = row_vec(r).as_point();
			return *this;
		}

		inline const T* get_ptr() const
		{
			return reinterpret_cast<const T*>(&m_rows[0]);
		}
		inline T* get_ptr()
		{
			return reinterpret_cast<T*>(&m_rows[0]);
		}

		inline matrix& operator+=(const matrix& other)
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i] += other.m_rows[i];
			return *this;
		}

		inline matrix& operator-=(const matrix& other)
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i] -= other.m_rows[i];
			return *this;
		}

		inline matrix& operator*=(T val)
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i] *= val;
			return *this;
		}

		inline matrix& operator/=(T val)
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i] /= val;
			return *this;
		}

		inline matrix& operator*=(const matrix& other)
		{
			matrix result;
			matrix_mul_helper(result, *this, other);
			*this = result;
			return *this;
		}

		friend inline matrix operator+(const matrix& lhs, const matrix& rhs)
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = lhs.m_rows[i] + rhs.m_rows[i];
			return result;
		}

		friend inline matrix operator-(const matrix& lhs, const matrix& rhs)
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = lhs.m_rows[i] - rhs.m_rows[i];
			return result;
		}

		friend inline matrix operator*(const matrix& lhs, T val)
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = lhs.m_rows[i] * val;
			return result;
		}

		friend inline matrix operator/(const matrix& lhs, T val)
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = lhs.m_rows[i] / val;
			return result;
		}

		friend inline matrix operator*(T val, const matrix& rhs)
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = val * rhs.m_rows[i];
			return result;
		}

#if 0
		template<uint32_t R0, uint32_t C0, uint32_t R1, uint32_t C1, typename T>
		friend inline matrix operator*(const matrix<R0, C0, T>& lhs, const matrix<R1, C1, T>& rhs)
		{
			matrix<R0, C1, T> result;
			return matrix_mul_helper(result, lhs, rhs);
		}
#endif
		friend inline matrix operator*(const matrix& lhs, const matrix& rhs)
		{
			matrix result;
			return matrix_mul_helper(result, lhs, rhs);
		}

		friend inline row_vec operator*(const col_vec& a, const matrix& b)
		{
			return transform(a, b);
		}

		inline matrix operator+() const
		{
			return *this;
		}

		inline matrix operator-() const
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				result[i] = -m_rows[i];
			return result;
		}

		inline matrix& clear()
		{
			for (uint32_t i = 0; i < R; i++)
				m_rows[i].clear();
			return *this;
		}

		inline matrix& set_zero_matrix()
		{
			clear();
			return *this;
		}

		inline matrix& set_identity_matrix()
		{
			for (uint32_t i = 0; i < R; i++)
			{
				m_rows[i].clear();
				m_rows[i][i] = 1.0f;
			}
			return *this;
		}

		inline matrix& set_scale_matrix(float s)
		{
			clear();
			for (int i = 0; i < (R - 1); i++)
				m_rows[i][i] = s;
			m_rows[R - 1][C - 1] = 1.0f;
			return *this;
		}

		inline matrix& set_scale_matrix(const row_vec& s)
		{
			clear();
			for (uint32_t i = 0; i < R; i++)
				m_rows[i][i] = s[i];
			return *this;
		}

		inline matrix& set_scale_matrix(float x, float y)
		{
			set_identity_matrix();
			m_rows[0].set_x(x);
			m_rows[1].set_y(y);
			return *this;
		}

		inline matrix& set_scale_matrix(float x, float y, float z)
		{
			set_identity_matrix();
			m_rows[0].set_x(x);
			m_rows[1].set_y(y);
			m_rows[2].set_z(z);
			return *this;
		}

		inline matrix& set_translate_matrix(const row_vec& s)
		{
			set_identity_matrix();
			set_translate(s);
			return *this;
		}

		inline matrix& set_translate_matrix(float x, float y)
		{
			set_identity_matrix();
			set_translate(row_vec(x, y).as_point());
			return *this;
		}

		inline matrix& set_translate_matrix(float x, float y, float z)
		{
			set_identity_matrix();
			set_translate(row_vec(x, y, z).as_point());
			return *this;
		}

		inline matrix get_transposed() const
		{
			static_assert(R == C);

			matrix result;
			for (uint32_t i = 0; i < R; i++)
				for (uint32_t j = 0; j < C; j++)
					result.m_rows[i][j] = m_rows[j][i];
			return result;
		}

		inline matrix<C, R, T> get_transposed_nonsquare() const
		{
			matrix<C, R, T> result;
			for (uint32_t i = 0; i < R; i++)
				for (uint32_t j = 0; j < C; j++)
					result[j][i] = m_rows[i][j];
			return result;
		}

		inline matrix& transpose_in_place()
		{
			matrix result;
			for (uint32_t i = 0; i < R; i++)
				for (uint32_t j = 0; j < C; j++)
					result.m_rows[i][j] = m_rows[j][i];
			*this = result;
			return *this;
		}

		// Frobenius Norm
		T get_norm() const
		{
			T result = 0;

			for (uint32_t i = 0; i < R; i++)
				for (uint32_t j = 0; j < C; j++)
					result += m_rows[i][j] * m_rows[i][j];

			return static_cast<T>(sqrt(result));
		}

		inline matrix get_power(T p) const
		{
			matrix result;

			for (uint32_t i = 0; i < R; i++)
				for (uint32_t j = 0; j < C; j++)
					result[i][j] = static_cast<T>(pow(m_rows[i][j], p));

			return result;
		}

		inline matrix<1, R, T> numpy_dot(const matrix<1, C, T>& b) const
		{
			matrix<1, R, T> result;

			for (uint32_t r = 0; r < R; r++)
			{
				T sum = 0;
				for (uint32_t c = 0; c < C; c++)
					sum += m_rows[r][c] * b[0][c];

				result[0][r] = static_cast<T>(sum);
			}

			return result;
		}

		bool invert(matrix& result) const
		{
			static_assert(R == C);

			result.set_identity_matrix();

			matrix mat(*this);

			for (uint32_t c = 0; c < C; c++)
			{
				uint32_t max_r = c;
				for (uint32_t r = c + 1; r < R; r++)
					if (fabs(mat[r][c]) > fabs(mat[max_r][c]))
						max_r = r;

				if (mat[max_r][c] == 0.0f)
				{
					result.set_identity_matrix();
					return false;
				}

				std::swap(mat[c], mat[max_r]);
				std::swap(result[c], result[max_r]);

				result[c] /= mat[c][c];
				mat[c] /= mat[c][c];

				for (uint32_t row = 0; row < R; row++)
				{
					if (row != c)
					{
						const row_vec temp(mat[row][c]);
						mat[row] -= row_vec::mul_components(mat[c], temp);
						result[row] -= row_vec::mul_components(result[c], temp);
					}
				}
			}

			return true;
		}

		matrix& invert_in_place()
		{
			matrix result;
			invert(result);
			*this = result;
			return *this;
		}

		matrix get_inverse() const
		{
			matrix result;
			invert(result);
			return result;
		}

		T get_det() const
		{
			static_assert(R == C);
			return det_helper(*this, R);
		}

		bool equal_tol(const matrix& b, float tol) const
		{
			for (uint32_t r = 0; r < R; r++)
				if (!row_vec::equal_tol(m_rows[r], b.m_rows[r], tol))
					return false;
			return true;
		}

		bool is_square() const
		{
			return R == C;
		}

		double get_trace() const
		{
			static_assert(is_square());

			T total = 0;
			for (uint32_t i = 0; i < R; i++)
				total += (*this)(i, i);

			return total;
		}

		void print() const
		{
			for (uint32_t r = 0; r < R; r++)
			{
				for (uint32_t c = 0; c < C; c++)
					printf("%3.7f ", (*this)(r, c));
				printf("\n");
			}
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left).
		// Confusingly, note that the data type is named "col_vec", but mathematically it's actually written as a row vector (of size equal to the # matrix rows, which is why it's called a "col_vec" in this class).
		// 1xR * RxC = 1xC
		// This dots against the matrix columns.
		static inline row_vec transform(const col_vec& a, const matrix& b)
		{
			row_vec result(b[0] * a[0]);
			for (uint32_t r = 1; r < R; r++)
				result += b[r] * a[r];
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left).
		// Last component of vec is assumed to be 1.
		static inline row_vec transform_point(const col_vec& a, const matrix& b)
		{
			row_vec result(0);
			for (int r = 0; r < (R - 1); r++)
				result += b[r] * a[r];
			result += b[R - 1];
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left).
		// Last component of vec is assumed to be 0.
		static inline row_vec transform_vector(const col_vec& a, const matrix& b)
		{
			row_vec result(0);
			for (int r = 0; r < (R - 1); r++)
				result += b[r] * a[r];
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left).
		// Last component of vec is assumed to be 1.
		static inline subcol_vec transform_point(const subcol_vec& a, const matrix& b)
		{
			subcol_vec result(0);
			for (int r = 0; r < static_cast<int>(R); r++)
			{
				const T s = (r < subcol_vec::num_elements) ? a[r] : 1.0f;
				for (int c = 0; c < static_cast<int>(C - 1); c++)
					result[c] += b[r][c] * s;
			}
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left).
		// Last component of vec is assumed to be 0.
		static inline subcol_vec transform_vector(const subcol_vec& a, const matrix& b)
		{
			subcol_vec result(0);
			for (int r = 0; r < static_cast<int>(R - 1); r++)
			{
				const T s = a[r];
				for (int c = 0; c < static_cast<int>(C - 1); c++)
					result[c] += b[r][c] * s;
			}
			return result;
		}

		// Like transform() above, but the matrix is effectively transposed before the multiply.
		static inline col_vec transform_transposed(const col_vec& a, const matrix& b)
		{
			static_assert(R == C);
			col_vec result;
			for (uint32_t r = 0; r < R; r++)
				result[r] = b[r].dot(a);
			return result;
		}

		// Like transform() above, but the matrix is effectively transposed before the multiply.
		// Last component of vec is assumed to be 0.
		static inline col_vec transform_vector_transposed(const col_vec& a, const matrix& b)
		{
			static_assert(R == C);
			col_vec result;
			for (uint32_t r = 0; r < R; r++)
			{
				T s = 0;
				for (uint32_t c = 0; c < (C - 1); c++)
					s += b[r][c] * a[c];

				result[r] = s;
			}
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left), but the matrix is effectively transposed before the multiply.
		// Last component of vec is assumed to be 1.
		static inline subcol_vec transform_point_transposed(const subcol_vec& a, const matrix& b)
		{
			static_assert(R == C);
			subcol_vec result(0);
			for (int r = 0; r < R; r++)
			{
				const T s = (r < subcol_vec::num_elements) ? a[r] : 1.0f;
				for (int c = 0; c < (C - 1); c++)
					result[c] += b[c][r] * s;
			}
			return result;
		}

		// This method transforms a vec by a matrix (D3D-style: row vector on left), but the matrix is effectively transposed before the multiply.
		// Last component of vec is assumed to be 0.
		static inline subcol_vec transform_vector_transposed(const subcol_vec& a, const matrix& b)
		{
			static_assert(R == C);
			subcol_vec result(0);
			for (int r = 0; r < static_cast<int>(R - 1); r++)
			{
				const T s = a[r];
				for (int c = 0; c < static_cast<int>(C - 1); c++)
					result[c] += b[c][r] * s;
			}
			return result;
		}

		// This method transforms a matrix by a vector (OGL style, col vector on the right).
		// Note that the data type is named "row_vec", but mathematically it's actually written as a column vector (of size equal to the # matrix cols).
		// RxC * Cx1 = Rx1
		// This dots against the matrix rows.
		static inline col_vec transform(const matrix& b, const row_vec& a)
		{
			col_vec result;
			for (int r = 0; r < static_cast<int>(R); r++)
				result[r] = b[r].dot(a);
			return result;
		}

		// This method transforms a matrix by a vector (OGL style, col vector on the right), except the matrix is effectively transposed before the multiply.
		// Note that the data type is named "row_vec", but mathematically it's actually written as a column vector (of size equal to the # matrix cols).
		// RxC * Cx1 = Rx1
		// This dots against the matrix cols.
		static inline col_vec transform_transposed(const matrix& b, const row_vec& a)
		{
			static_assert(R == C);
			row_vec result(b[0] * a[0]);
			for (int r = 1; r < static_cast<int>(R); r++)
				result += b[r] * a[r];
			return col_vec(result);
		}

		static inline matrix& mul_components(matrix& result, const matrix& lhs, const matrix& rhs)
		{
			for (uint32_t r = 0; r < R; r++)
				result[r] = row_vec::mul_components(lhs[r], rhs[r]);
			return result;
		}

		static inline matrix& concat(matrix& lhs, const matrix& rhs)
		{
			return matrix_mul_helper(lhs, matrix(lhs), rhs);
		}

		inline matrix& concat_in_place(const matrix& rhs)
		{
			return concat(*this, rhs);
		}

		static inline matrix& multiply(matrix& result, const matrix& lhs, const matrix& rhs)
		{
			matrix temp;
			matrix* pResult = ((&result == &lhs) || (&result == &rhs)) ? &temp : &result;

			matrix_mul_helper(*pResult, lhs, rhs);
			if (pResult != &result)
				result = *pResult;

			return result;
		}

		static matrix make_zero_matrix()
		{
			matrix result;
			result.clear();
			return result;
		}

		static matrix make_identity_matrix()
		{
			matrix result;
			result.set_identity_matrix();
			return result;
		}

		static matrix make_translate_matrix(const row_vec& t)
		{
			return matrix(basisu::cIdentity).set_translate(t);
		}

		static matrix make_translate_matrix(float x, float y)
		{
			return matrix(basisu::cIdentity).set_translate_matrix(x, y);
		}

		static matrix make_translate_matrix(float x, float y, float z)
		{
			return matrix(basisu::cIdentity).set_translate_matrix(x, y, z);
		}

		static inline matrix make_scale_matrix(float s)
		{
			return matrix().set_scale_matrix(s);
		}

		static inline matrix make_scale_matrix(const row_vec& s)
		{
			return matrix().set_scale_matrix(s);
		}

		static inline matrix make_scale_matrix(float x, float y)
		{
			static_assert(R >= 3 && C >= 3);
			matrix result;
			result.set_identity_matrix();
			result.m_rows[0][0] = x;
			result.m_rows[1][1] = y;
			return result;
		}

		static inline matrix make_scale_matrix(float x, float y, float z)
		{
			static_assert(R >= 4 && C >= 4);
			matrix result;
			result.set_identity_matrix();
			result.m_rows[0][0] = x;
			result.m_rows[1][1] = y;
			result.m_rows[2][2] = z;
			return result;
		}

		// Helpers derived from Graphics Gems 1 and 2 (Matrices and Transformations, Ronald N. Goldman)
		static matrix make_rotate_matrix(const vec<3, T>& axis, T ang)
		{
			static_assert(R >= 3 && C >= 3);

			vec<3, T> norm_axis(axis.get_normalized());

			double cos_a = cos(ang);
			double inv_cos_a = 1.0f - cos_a;

			double sin_a = sin(ang);

			const T x = norm_axis[0];
			const T y = norm_axis[1];
			const T z = norm_axis[2];

			const double x2 = norm_axis[0] * norm_axis[0];
			const double y2 = norm_axis[1] * norm_axis[1];
			const double z2 = norm_axis[2] * norm_axis[2];

			matrix result;
			result.set_identity_matrix();

			result[0][0] = (T)((inv_cos_a * x2) + cos_a);
			result[1][0] = (T)((inv_cos_a * x * y) + (sin_a * z));
			result[2][0] = (T)((inv_cos_a * x * z) - (sin_a * y));

			result[0][1] = (T)((inv_cos_a * x * y) - (sin_a * z));
			result[1][1] = (T)((inv_cos_a * y2) + cos_a);
			result[2][1] = (T)((inv_cos_a * y * z) + (sin_a * x));

			result[0][2] = (T)((inv_cos_a * x * z) + (sin_a * y));
			result[1][2] = (T)((inv_cos_a * y * z) - (sin_a * x));
			result[2][2] = (T)((inv_cos_a * z2) + cos_a);

			return result;
		}

		static inline matrix make_rotate_matrix(T ang)
		{
			static_assert(R >= 2 && C >= 2);

			matrix ret(basisu::cIdentity);

			const T sin_a = static_cast<T>(sin(ang));
			const T cos_a = static_cast<T>(cos(ang));

			ret[0][0] = +cos_a;
			ret[0][1] = -sin_a;
			ret[1][0] = +sin_a;
			ret[1][1] = +cos_a;

			return ret;
		}

		static inline matrix make_rotate_matrix(uint32_t axis, T ang)
		{
			vec<3, T> axis_vec;
			axis_vec.clear();
			axis_vec[axis] = 1.0f;
			return make_rotate_matrix(axis_vec, ang);
		}

		static inline matrix make_cross_product_matrix(const vec<3, scalar_type>& c)
		{
			static_assert((num_rows >= 3) && (num_cols >= 3));
			matrix ret(basisu::cClear);
			ret[0][1] = c[2];
			ret[0][2] = -c[1];
			ret[1][0] = -c[2];
			ret[1][2] = c[0];
			ret[2][0] = c[1];
			ret[2][1] = -c[0];
			return ret;
		}

		static inline matrix make_reflection_matrix(const vec<4, scalar_type>& n, const vec<4, scalar_type>& q)
		{
			static_assert((num_rows == 4) && (num_cols == 4));
			matrix ret;
			assert(n.is_vector() && q.is_vector());
			ret = make_identity_matrix() - 2.0f * make_tensor_product_matrix(n, n);
			ret.set_translate((2.0f * q.dot(n) * n).as_point());
			return ret;
		}

		static inline matrix make_tensor_product_matrix(const row_vec& v, const row_vec& w)
		{
			matrix ret;
			for (int r = 0; r < num_rows; r++)
				ret[r] = row_vec::mul_components(v.broadcast(r), w);
			return ret;
		}

		static inline matrix make_uniform_scaling_matrix(const vec<4, scalar_type>& q, scalar_type c)
		{
			static_assert((num_rows == 4) && (num_cols == 4));
			assert(q.is_vector());
			matrix ret;
			ret = c * make_identity_matrix();
			ret.set_translate(((1.0f - c) * q).as_point());
			return ret;
		}

		static inline matrix make_nonuniform_scaling_matrix(const vec<4, scalar_type>& q, scalar_type c, const vec<4, scalar_type>& w)
		{
			static_assert((num_rows == 4) && (num_cols == 4));
			assert(q.is_vector() && w.is_vector());
			matrix ret;
			ret = make_identity_matrix() - (1.0f - c) * make_tensor_product_matrix(w, w);
			ret.set_translate(((1.0f - c) * q.dot(w) * w).as_point());
			return ret;
		}

		// n = normal of plane, q = point on plane
		static inline matrix make_ortho_projection_matrix(const vec<4, scalar_type>& n, const vec<4, scalar_type>& q)
		{
			assert(n.is_vector() && q.is_vector());
			matrix ret;
			ret = make_identity_matrix() - make_tensor_product_matrix(n, n);
			ret.set_translate((q.dot(n) * n).as_point());
			return ret;
		}

		static inline matrix make_parallel_projection(const vec<4, scalar_type>& n, const vec<4, scalar_type>& q, const vec<4, scalar_type>& w)
		{
			assert(n.is_vector() && q.is_vector() && w.is_vector());
			matrix ret;
			ret = make_identity_matrix() - (make_tensor_product_matrix(n, w) / (w.dot(n)));
			ret.set_translate(((q.dot(n) / w.dot(n)) * w).as_point());
			return ret;
		}

	protected:
		row_vec m_rows[R];

		static T det_helper(const matrix& a, uint32_t n)
		{
			// Algorithm ported from Numerical Recipes in C.
			T d;
			matrix m;
			if (n == 2)
				d = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
			else
			{
				d = 0;
				for (uint32_t j1 = 1; j1 <= n; j1++)
				{
					for (uint32_t i = 2; i <= n; i++)
					{
						int j2 = 1;
						for (uint32_t j = 1; j <= n; j++)
						{
							if (j != j1)
							{
								m(i - 2, j2 - 1) = a(i - 1, j - 1);
								j2++;
							}
						}
					}
					d += (((1 + j1) & 1) ? -1.0f : 1.0f) * a(1 - 1, j1 - 1) * det_helper(m, n - 1);
				}
			}
			return d;
		}
	};

	typedef matrix<2, 2, float> matrix22F;
	typedef matrix<2, 2, double> matrix22D;

	typedef matrix<3, 3, float> matrix33F;
	typedef matrix<3, 3, double> matrix33D;

	typedef matrix<4, 4, float> matrix44F;
	typedef matrix<4, 4, double> matrix44D;

	typedef matrix<8, 8, float> matrix88F;

	// These helpers create good old D3D-style matrices.
	inline matrix44F matrix44F_make_perspective_offcenter_lh(float l, float r, float b, float t, float nz, float fz)
	{
		float two_nz = 2.0f * nz;
		float one_over_width = 1.0f / (r - l);
		float one_over_height = 1.0f / (t - b);

		matrix44F view_to_proj;
		view_to_proj[0].set(two_nz * one_over_width, 0.0f, 0.0f, 0.0f);
		view_to_proj[1].set(0.0f, two_nz * one_over_height, 0.0f, 0.0f);
		view_to_proj[2].set(-(l + r) * one_over_width, -(t + b) * one_over_height, fz / (fz - nz), 1.0f);
		view_to_proj[3].set(0.0f, 0.0f, -view_to_proj[2][2] * nz, 0.0f);
		return view_to_proj;
	}

	// fov_y: full Y field of view (radians)
	// aspect: viewspace width/height
	inline matrix44F matrix44F_make_perspective_fov_lh(float fov_y, float aspect, float nz, float fz)
	{
		double sin_fov = sin(0.5f * fov_y);
		double cos_fov = cos(0.5f * fov_y);

		float y_scale = static_cast<float>(cos_fov / sin_fov);
		float x_scale = static_cast<float>(y_scale / aspect);

		matrix44F view_to_proj;
		view_to_proj[0].set(x_scale, 0, 0, 0);
		view_to_proj[1].set(0, y_scale, 0, 0);
		view_to_proj[2].set(0, 0, fz / (fz - nz), 1);
		view_to_proj[3].set(0, 0, -nz * fz / (fz - nz), 0);
		return view_to_proj;
	}

	inline matrix44F matrix44F_make_ortho_offcenter_lh(float l, float r, float b, float t, float nz, float fz)
	{
		matrix44F view_to_proj;
		view_to_proj[0].set(2.0f / (r - l), 0.0f, 0.0f, 0.0f);
		view_to_proj[1].set(0.0f, 2.0f / (t - b), 0.0f, 0.0f);
		view_to_proj[2].set(0.0f, 0.0f, 1.0f / (fz - nz), 0.0f);
		view_to_proj[3].set((l + r) / (l - r), (t + b) / (b - t), nz / (nz - fz), 1.0f);
		return view_to_proj;
	}

	inline matrix44F matrix44F_make_ortho_lh(float w, float h, float nz, float fz)
	{
		return matrix44F_make_ortho_offcenter_lh(-w * .5f, w * .5f, -h * .5f, h * .5f, nz, fz);
	}

	inline matrix44F matrix44F_make_projection_to_screen_d3d(int x, int y, int w, int h, float min_z, float max_z)
	{
		matrix44F proj_to_screen;
		proj_to_screen[0].set(w * .5f, 0.0f, 0.0f, 0.0f);
		proj_to_screen[1].set(0, h * -.5f, 0.0f, 0.0f);
		proj_to_screen[2].set(0, 0.0f, max_z - min_z, 0.0f);
		proj_to_screen[3].set(x + w * .5f, y + h * .5f, min_z, 1.0f);
		return proj_to_screen;
	}

	inline matrix44F matrix44F_make_lookat_lh(const vec3F& camera_pos, const vec3F& look_at, const vec3F& camera_up, float camera_roll_ang_in_radians)
	{
		vec4F col2(look_at - camera_pos);
		assert(col2.is_vector());
		if (col2.normalize() == 0.0f)
			col2.set(0, 0, 1, 0);

		vec4F col1(camera_up);
		assert(col1.is_vector());
		if (!col2[0] && !col2[2])
			col1.set(-1.0f, 0.0f, 0.0f, 0.0f);

		if ((col1.dot(col2)) > .9999f)
			col1.set(0.0f, 1.0f, 0.0f, 0.0f);

		vec4F col0(vec4F::cross3(col1, col2).normalize_in_place());
		col1 = vec4F::cross3(col2, col0).normalize_in_place();

		matrix44F rotm(matrix44F::make_identity_matrix());
		rotm.set_col(0, col0);
		rotm.set_col(1, col1);
		rotm.set_col(2, col2);
		return matrix44F::make_translate_matrix(-camera_pos[0], -camera_pos[1], -camera_pos[2]) * rotm * matrix44F::make_rotate_matrix(2, camera_roll_ang_in_radians);
	}

	template<typename R> R matrix_NxN_create_DCT()
	{
		assert(R::num_rows == R::num_cols);

		const uint32_t N = R::num_cols;

		R result;
		for (uint32_t k = 0; k < N; k++)
		{
			for (uint32_t n = 0; n < N; n++)
			{
				double f;

				if (!k)
					f = 1.0f / sqrt(float(N));
				else
					f = sqrt(2.0f / float(N)) * cos((basisu::cPiD * (2.0f * float(n) + 1.0f) * float(k)) / (2.0f * float(N)));

				result(k, n) = static_cast<typename R::scalar_type>(f);
			}
		}

		return result;
	}

	template<typename R> R matrix_NxN_DCT(const R& a, const R& dct)
	{
		R temp;
		matrix_mul_helper<R, R, R>(temp, dct, a);
		R result;
		matrix_mul_helper_transpose_rhs<R, R, R>(result, temp, dct);
		return result;
	}

	template<typename R> R matrix_NxN_IDCT(const R& b, const R& dct)
	{
		R temp;
		matrix_mul_helper_transpose_lhs<R, R, R>(temp, dct, b);
		R result;
		matrix_mul_helper<R, R, R>(result, temp, dct);
		return result;
	}

	template<typename X, typename Y> matrix<X::num_rows* Y::num_rows, X::num_cols* Y::num_cols, typename X::scalar_type> matrix_kronecker_product(const X& a, const Y& b)
	{
		matrix<X::num_rows* Y::num_rows, X::num_cols* Y::num_cols, typename X::scalar_type> result;

		for (uint32_t r = 0; r < X::num_rows; r++)
		{
			for (uint32_t c = 0; c < X::num_cols; c++)
			{
				for (uint32_t i = 0; i < Y::num_rows; i++)
					for (uint32_t j = 0; j < Y::num_cols; j++)
						result(r * Y::num_rows + i, c * Y::num_cols + j) = a(r, c) * b(i, j);
			}
		}

		return result;
	}

	template<typename X, typename Y> matrix<X::num_rows + Y::num_rows, X::num_cols, typename X::scalar_type> matrix_combine_vertically(const X& a, const Y& b)
	{
		matrix<X::num_rows + Y::num_rows, X::num_cols, typename X::scalar_type> result;

		for (uint32_t r = 0; r < X::num_rows; r++)
			for (uint32_t c = 0; c < X::num_cols; c++)
				result(r, c) = a(r, c);

		for (uint32_t r = 0; r < Y::num_rows; r++)
			for (uint32_t c = 0; c < Y::num_cols; c++)
				result(r + X::num_rows, c) = b(r, c);

		return result;
	}

	inline matrix88F get_haar8()
	{
		matrix22F haar2(
			1, 1,
			1, -1);
		matrix22F i2(
			1, 0,
			0, 1);
		matrix44F i4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		matrix<1, 2, float> b0; b0(0, 0) = 1; b0(0, 1) = 1;
		matrix<1, 2, float> b1; b1(0, 0) = 1.0f; b1(0, 1) = -1.0f;

		matrix<2, 4, float> haar4_0 = matrix_kronecker_product(haar2, b0);
		matrix<2, 4, float> haar4_1 = matrix_kronecker_product(i2, b1);

		matrix<4, 4, float> haar4 = matrix_combine_vertically(haar4_0, haar4_1);

		matrix<4, 8, float> haar8_0 = matrix_kronecker_product(haar4, b0);
		matrix<4, 8, float> haar8_1 = matrix_kronecker_product(i4, b1);

		haar8_0[2] *= sqrtf(2);
		haar8_0[3] *= sqrtf(2);
		haar8_1 *= 2.0f;

		matrix<8, 8, float> haar8 = matrix_combine_vertically(haar8_0, haar8_1);

		return haar8;
	}

	inline matrix44F get_haar4()
	{
		const float sqrt2 = 1.4142135623730951f;

		return matrix44F(
			.5f * 1, .5f * 1, .5f * 1, .5f * 1,
			.5f * 1, .5f * 1, .5f * -1, .5f * -1,
			.5f * sqrt2, .5f * -sqrt2, 0, 0,
			0, 0, .5f * sqrt2, .5f * -sqrt2);
	}

	template<typename T>
	inline matrix<2, 2, T> get_inverse_2x2(const matrix<2, 2, T>& m)
	{
		double a = m[0][0];
		double b = m[0][1];
		double c = m[1][0];
		double d = m[1][1];

		double det = a * d - b * c;
		if (det != 0.0f)
			det = 1.0f / det;

		matrix<2, 2, T> result;
		result[0][0] = static_cast<T>(d * det);
		result[0][1] = static_cast<T>(-b * det);
		result[1][0] = static_cast<T>(-c * det);
		result[1][1] = static_cast<T>(a * det);
		return result;
	}

} // namespace bu_math

namespace basisu
{
	class tracked_stat
	{
	public:
		tracked_stat() { clear(); }

		inline void clear() { m_num = 0; m_total = 0; m_total2 = 0; }

		inline void update(int32_t val) { m_num++; m_total += val; m_total2 += val * val; }

		inline tracked_stat& operator += (uint32_t val) { update(val); return *this; }

		inline uint32_t get_number_of_values() { return m_num; }
		inline uint64_t get_total() const { return m_total; }
		inline uint64_t get_total2() const { return m_total2; }

		inline float get_average() const { return m_num ? (float)m_total / m_num : 0.0f; };
		inline float get_std_dev() const { return m_num ? sqrtf((float)(m_num * m_total2 - m_total * m_total)) / m_num : 0.0f; }
		inline float get_variance() const { float s = get_std_dev(); return s * s; }

	private:
		uint32_t m_num;
		int64_t m_total;
		int64_t m_total2;
	};

	class tracked_stat_dbl
	{
	public:
		tracked_stat_dbl() { clear(); }

		inline void clear() { m_num = 0; m_total = 0; m_total2 = 0; }

		inline void update(double val) { m_num++; m_total += val; m_total2 += val * val; }

		inline tracked_stat_dbl& operator += (double val) { update(val); return *this; }

		inline uint64_t get_number_of_values() { return m_num; }
		inline double get_total() const { return m_total; }
		inline double get_total2() const { return m_total2; }

		inline double get_average() const { return m_num ? m_total / (double)m_num : 0.0f; };
		inline double get_std_dev() const { return m_num ? sqrt((double)(m_num * m_total2 - m_total * m_total)) / m_num : 0.0f; }
		inline double get_variance() const { double s = get_std_dev(); return s * s; }

	private:
		uint64_t m_num;
		double m_total;
		double m_total2;
	};

	template<typename FloatType>
	struct stats
	{
		uint32_t m_n;
		FloatType m_total, m_total_sq;		// total, total of squares values
		FloatType m_avg, m_avg_sq;			// mean, mean of the squared values
		FloatType m_rms;					// sqrt(m_avg_sq)
		FloatType m_std_dev, m_var;			// population standard deviation and variance
		FloatType m_mad;					// mean absolute deviation
		FloatType m_min, m_max, m_range;	// min and max values, and max-min
		FloatType m_len;					// length of values as a vector (Euclidean norm or L2 norm)
		FloatType m_coeff_of_var;			// coefficient of variation (std_dev/mean), High CV: Indicates greater variability relative to the mean, meaning the data values are more spread out, 
											// Low CV : Indicates less variability relative to the mean, meaning the data values are more consistent.
		
		FloatType m_skewness;				// Skewness = 0: The data is perfectly symmetric around the mean, 
											// Skewness > 0: The data is positively skewed (right-skewed), 
											// Skewness < 0: The data is negatively skewed (left-skewed)
											// 0-.5 approx. symmetry, .5-1 moderate skew, >= 1 highly skewed
		
		FloatType m_kurtosis;				// Excess Kurtosis: Kurtosis = 0: The distribution has normal kurtosis (mesokurtic)
											// Kurtosis > 0: The distribution is leptokurtic, with heavy tails and a sharp peak
											// Kurtosis < 0: The distribution is platykurtic, with light tails and a flatter peak

		bool m_any_zero;

		FloatType m_median;
		uint32_t m_median_index;

		stats() 
		{ 
			clear(); 
		}

		void clear()
		{
			m_n = 0;
			m_total = 0, m_total_sq = 0;
			m_avg = 0, m_avg_sq = 0;
			m_rms = 0;
			m_std_dev = 0, m_var = 0;
			m_mad = 0;
			m_min = BIG_FLOAT_VAL, m_max = -BIG_FLOAT_VAL; m_range = 0.0f;
			m_len = 0;
			m_coeff_of_var = 0;
			m_skewness = 0;
			m_kurtosis = 0;
			m_any_zero = false;
			
			m_median = 0;
			m_median_index = 0;
		}

		template<typename T>
		void calc_median(uint32_t n, const T* pVals, uint32_t stride = 1)
		{
			m_median = 0;
			m_median_index = 0;

			if (!n)
				return;

			basisu::vector< std::pair<T, uint32_t> > vals(n);

			for (uint32_t i = 0; i < n; i++)
			{
				vals[i].first = pVals[i * stride];
				vals[i].second = i;
			}

			std::sort(vals.begin(), vals.end(), [](const std::pair<T, uint32_t>& a, const std::pair<T, uint32_t>& b) {
				return a.first < b.first;
				});

			m_median = vals[n / 2].first;
			if ((n & 1) == 0)
				m_median = (m_median + vals[(n / 2) - 1].first) * .5f;

			m_median_index = vals[n / 2].second;
		}

		template<typename T>
		void calc(uint32_t n, const T* pVals, uint32_t stride = 1, bool calc_median_flag = false)
		{
			clear();
						
			if (!n)
				return;

			if (calc_median_flag)
				calc_median(n, pVals, stride);

			m_n = n;

			for (uint32_t i = 0; i < n; i++)
			{
				FloatType v = (FloatType)pVals[i * stride];

				if (v == 0.0f)
					m_any_zero = true;
				
				m_total += v;
				m_total_sq += v * v;
				
				if (!i)
				{
					m_min = v;
					m_max = v;
				}
				else
				{
					m_min = minimum(m_min, v);
					m_max = maximum(m_max, v);
				}
			}

			m_range = m_max - m_min;

			m_len = sqrt(m_total_sq);

			const FloatType nd = (FloatType)n;

			m_avg = m_total / nd;
			m_avg_sq = m_total_sq / nd;
			m_rms = sqrt(m_avg_sq);
			
			for (uint32_t i = 0; i < n; i++)
			{
				FloatType v = (FloatType)pVals[i * stride];
				FloatType d = v - m_avg;
				
				const FloatType d2 = d * d;
				const FloatType d3 = d2 * d;
				const FloatType d4 = d3 * d;

				m_var += d2;
				m_mad += fabs(d);
				m_skewness += d3;
				m_kurtosis += d4;
			}

			m_var /= nd;
			m_mad /= nd;

			m_std_dev = sqrt(m_var);

			m_coeff_of_var = (m_avg != 0.0f) ? (m_std_dev / fabs(m_avg)) : 0.0f;

			FloatType k3 = m_std_dev * m_std_dev * m_std_dev;
			FloatType k4 = k3 * m_std_dev;
			m_skewness = (k3 != 0.0f) ? ((m_skewness / nd) / k3) : 0.0f;
			m_kurtosis = (k4 != 0.0f) ? (((m_kurtosis / nd) / k4) - 3.0f) : 0.0f;
		}

		// Only compute average, variance and standard deviation.
		template<typename T>
		void calc_simplified(uint32_t n, const T* pVals, uint32_t stride = 1)
		{
			clear();

			if (!n)
				return;

			m_n = n;

			for (uint32_t i = 0; i < n; i++)
			{
				FloatType v = (FloatType)pVals[i * stride];

				m_total += v;
			}
						
			const FloatType nd = (FloatType)n;

			m_avg = m_total / nd;

			for (uint32_t i = 0; i < n; i++)
			{
				FloatType v = (FloatType)pVals[i * stride];
				FloatType d = v - m_avg;

				const FloatType d2 = d * d;

				m_var += d2;
			}

			m_var /= nd;
			m_std_dev = sqrt(m_var);
		}
	};

	template<typename FloatType>
	struct comparative_stats
	{
		FloatType m_cov;					// covariance
		FloatType m_pearson;				// Pearson Correlation Coefficient (r) [-1,1]
		FloatType m_mse;					// mean squared error
		FloatType m_rmse;					// root mean squared error
		FloatType m_mae;					// mean abs error
		FloatType m_rmsle;					// root mean squared log error
		FloatType m_euclidean_dist;			// euclidean distance between values as vectors
		FloatType m_cosine_sim;				// normalized dot products of values as vectors
		FloatType m_min_diff, m_max_diff;	// minimum/maximum abs difference between values
				
		comparative_stats()
		{
			clear();
		}

		void clear()
		{
			m_cov = 0;
			m_pearson = 0;
			m_mse = 0;
			m_rmse = 0;
			m_mae = 0;
			m_rmsle = 0;
			m_euclidean_dist = 0;
			m_cosine_sim = 0;
			m_min_diff = 0;
			m_max_diff = 0;
		}

		template<typename T>
		void calc(uint32_t n, const T* pA, const T* pB, uint32_t a_stride = 1, uint32_t b_stride = 1, const stats<FloatType> *pA_stats = nullptr, const stats<FloatType> *pB_stats = nullptr)
		{
			clear();
			if (!n)
				return;
						
			stats<FloatType> temp_a_stats;
			if (!pA_stats)
			{
				pA_stats = &temp_a_stats;
				temp_a_stats.calc(n, pA, a_stride);
			}

			stats<FloatType> temp_b_stats;
			if (!pB_stats)
			{
				pB_stats = &temp_b_stats;
				temp_b_stats.calc(n, pB, b_stride);
			}

			for (uint32_t i = 0; i < n; i++)
			{
				const FloatType fa = (FloatType)pA[i * a_stride];
				const FloatType fb = (FloatType)pB[i * b_stride];
								
				if ((pA_stats->m_min >= 0.0f) && (pB_stats->m_min >= 0.0f))
				{
					const FloatType ld = log(fa + 1.0f) - log(fb + 1.0f);
					m_rmsle += ld * ld;
				}

				const FloatType diff = fa - fb;
				const FloatType abs_diff = fabs(diff);
				
				m_mse += diff * diff;
				m_mae += abs_diff;

				m_min_diff = i ? minimum(m_min_diff, abs_diff) : abs_diff;
				m_max_diff = maximum(m_max_diff, abs_diff);

				const FloatType da = fa - pA_stats->m_avg;
				const FloatType db = fb - pB_stats->m_avg;
				m_cov += da * db;

				m_cosine_sim += fa * fb;
			}

			const FloatType nd = (FloatType)n;
			
			m_euclidean_dist = sqrt(m_mse);

			m_mse /= nd;
			m_rmse = sqrt(m_mse);

			m_mae /= nd;

			m_cov /= nd;
			
			FloatType dv = (pA_stats->m_std_dev * pB_stats->m_std_dev);
			if (dv != 0.0f)
				m_pearson = m_cov / dv;

			if ((pA_stats->m_min >= 0.0) && (pB_stats->m_min >= 0.0f))
				m_rmsle = sqrt(m_rmsle / nd);

			FloatType c = pA_stats->m_len * pB_stats->m_len;
			if (c != 0.0f)
				m_cosine_sim /= c;
			else
				m_cosine_sim = 0.0f;
		}

		// Only computes Pearson, cov, mse, rmse, Euclidean distance
		template<typename T>
		void calc_pearson(uint32_t n, const T* pA, const T* pB, uint32_t a_stride = 1, uint32_t b_stride = 1, const stats<FloatType>* pA_stats = nullptr, const stats<FloatType>* pB_stats = nullptr)
		{
			clear();
			if (!n)
				return;

			stats<FloatType> temp_a_stats;
			if (!pA_stats)
			{
				pA_stats = &temp_a_stats;
				temp_a_stats.calc(n, pA, a_stride);
			}

			stats<FloatType> temp_b_stats;
			if (!pB_stats)
			{
				pB_stats = &temp_b_stats;
				temp_b_stats.calc(n, pB, b_stride);
			}

			for (uint32_t i = 0; i < n; i++)
			{
				const FloatType fa = (FloatType)pA[i * a_stride];
				const FloatType fb = (FloatType)pB[i * b_stride];

				const FloatType diff = fa - fb;

				m_mse += diff * diff;

				const FloatType da = fa - pA_stats->m_avg;
				const FloatType db = fb - pB_stats->m_avg;
				m_cov += da * db;
			}

			const FloatType nd = (FloatType)n;

			m_euclidean_dist = sqrt(m_mse);

			m_mse /= nd;
			m_rmse = sqrt(m_mse);

			m_cov /= nd;

			FloatType dv = (pA_stats->m_std_dev * pB_stats->m_std_dev);
			if (dv != 0.0f)
				m_pearson = m_cov / dv;
		}

		// Only computes MSE, RMSE, eclidiean distance, and covariance.
		template<typename T>
		void calc_simplified(uint32_t n, const T* pA, const T* pB, uint32_t a_stride = 1, uint32_t b_stride = 1, const stats<FloatType>* pA_stats = nullptr, const stats<FloatType>* pB_stats = nullptr)
		{
			clear();
			if (!n)
				return;

			stats<FloatType> temp_a_stats;
			if (!pA_stats)
			{
				pA_stats = &temp_a_stats;
				temp_a_stats.calc(n, pA, a_stride);
			}

			stats<FloatType> temp_b_stats;
			if (!pB_stats)
			{
				pB_stats = &temp_b_stats;
				temp_b_stats.calc(n, pB, b_stride);
			}

			for (uint32_t i = 0; i < n; i++)
			{
				const FloatType fa = (FloatType)pA[i * a_stride];
				const FloatType fb = (FloatType)pB[i * b_stride];

				const FloatType diff = fa - fb;
				
				m_mse += diff * diff;
				
				const FloatType da = fa - pA_stats->m_avg;
				const FloatType db = fb - pB_stats->m_avg;
				m_cov += da * db;
			}

			const FloatType nd = (FloatType)n;

			m_euclidean_dist = sqrt(m_mse);

			m_mse /= nd;
			m_rmse = sqrt(m_mse);
						
			m_cov /= nd;
		}

		// Only computes covariance.
		template<typename T>
		void calc_cov(uint32_t n, const T* pA, const T* pB, uint32_t a_stride = 1, uint32_t b_stride = 1, const stats<FloatType>* pA_stats = nullptr, const stats<FloatType>* pB_stats = nullptr)
		{
			clear();
			if (!n)
				return;

			stats<FloatType> temp_a_stats;
			if (!pA_stats)
			{
				pA_stats = &temp_a_stats;
				temp_a_stats.calc(n, pA, a_stride);
			}

			stats<FloatType> temp_b_stats;
			if (!pB_stats)
			{
				pB_stats = &temp_b_stats;
				temp_b_stats.calc(n, pB, b_stride);
			}

			for (uint32_t i = 0; i < n; i++)
			{
				const FloatType fa = (FloatType)pA[i * a_stride];
				const FloatType fb = (FloatType)pB[i * b_stride];

				const FloatType da = fa - pA_stats->m_avg;
				const FloatType db = fb - pB_stats->m_avg;
				m_cov += da * db;
			}

			const FloatType nd = (FloatType)n;

			m_cov /= nd;
		}
	};
		
	class stat_history
	{
	public:
		stat_history(uint32_t size)
		{
			init(size);
		}

		void init(uint32_t size)
		{
			clear();

			m_samples.reserve(size);
			m_samples.resize(0);
			m_max_samples = size;
		}

		inline void clear()
		{
			m_samples.resize(0);
			m_max_samples = 0;
		}

		inline void update(double val)
		{
			m_samples.push_back(val);

			if (m_samples.size() > m_max_samples)
				m_samples.erase_index(0);
		}

		inline size_t size()
		{
			return m_samples.size();
		}

		struct stats
		{
			double m_avg = 0;
			double m_std_dev = 0;
			double m_var = 0;
			double m_mad = 0;
			double m_min_val = 0;
			double m_max_val = 0;

			void clear()
			{
				basisu::clear_obj(*this);
			}
		};

		inline void get_stats(stats& s)
		{
			s.clear();

			if (m_samples.empty())
				return;

			double total = 0, total2 = 0;

			for (size_t i = 0; i < m_samples.size(); i++)
			{
				const double v = m_samples[i];

				total += v;
				total2 += v * v;

				if (!i)
				{
					s.m_min_val = v;
					s.m_max_val = v;
				}
				else
				{
					s.m_min_val = basisu::minimum<double>(s.m_min_val, v);
					s.m_max_val = basisu::maximum<double>(s.m_max_val, v);
				}
			}

			const double n = (double)m_samples.size();

			s.m_avg = total / n;
			s.m_std_dev = sqrt((n * total2 - total * total)) / n;
			s.m_var = (n * total2 - total * total) / (n * n);

			double sc = 0;
			for (size_t i = 0; i < m_samples.size(); i++)
			{
				const double v = m_samples[i];
				s.m_mad += fabs(v - s.m_avg);

				sc += basisu::square(v - s.m_avg);
			}
			sc = sqrt(sc / n);

			s.m_mad /= n;
		}

	private:
		uint32_t m_max_samples;
		basisu::vector<double> m_samples;
	};

	// bfloat16 helpers, see:
	// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

	typedef union
	{
		uint32_t u;
		float f;
	} float32_union;

	typedef uint16_t bfloat16;

	inline float bfloat16_to_float(bfloat16 bfloat16)
	{
		float32_union float_union;
		float_union.u = ((uint32_t)bfloat16) << 16;
		return float_union.f;
	}

	inline bfloat16 float_to_bfloat16(float input, bool round_flag = true)
	{
		float32_union float_union;
		float_union.f = input;

		uint32_t exponent = (float_union.u >> 23) & 0xFF;

		// Check if the number is denormalized in float32 (exponent == 0)
		if (exponent == 0)
		{
			// Handle denormalized float32 as zero in bfloat16
			return 0x0000;
		}

		// Extract the top 16 bits (sign, exponent, and 7 most significant bits of the mantissa)
		uint32_t upperBits = float_union.u >> 16;

		if (round_flag)
		{
			// Check the most significant bit of the lower 16 bits for rounding
			uint32_t lowerBits = float_union.u & 0xFFFF;

			// Round to nearest or even
			if ((lowerBits & 0x8000) && 
				((lowerBits > 0x8000) || ((lowerBits == 0x8000) && (upperBits & 1)))
			   )
			{
				// Round up
				upperBits += 1;        

				// Check for overflow in the exponent after rounding up
				if (((upperBits & 0x7F80) == 0x7F80) && ((upperBits & 0x007F) == 0))
				{
					// Exponent overflow (the upper bits became all 1s)
					// Set the result to infinity
					upperBits = (upperBits & 0x8000) | 0x7F80;  // Preserve the sign bit, set exponent to 0xFF, and mantissa to 0
				}
			}
		}

		return (bfloat16)upperBits;
	}

	inline int bfloat16_get_exp(bfloat16 v)
	{
		return (int)((v >> 7) & 0xFF) - 127;
	}

	inline int bfloat16_get_mantissa(bfloat16 v)
	{
		return (v & 0x7F);
	}

	inline int bfloat16_get_sign(bfloat16 v)
	{
		return (v & 0x8000) ? -1 : 1;
	}

	inline bool bfloat16_is_nan_or_inf(bfloat16 v)
	{
		return ((v >> 7) & 0xFF) == 0xFF;
	}

	inline bool bfloat16_is_zero(bfloat16 v)
	{
		return (v & 0x7FFF) == 0;
	}

	inline bfloat16 bfloat16_init(int sign, int exp, int mant)
	{
		uint16_t res = (sign < 0) ? 0x8000 : 0;

		assert((exp >= -126) && (res <= 127));
		res |= ((exp + 127) << 7);

		assert((mant >= 0) && (mant < 128));
		res |= mant;

		return res;
	}
	
	
} // namespace basisu

