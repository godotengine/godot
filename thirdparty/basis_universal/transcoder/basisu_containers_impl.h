// basisu_containers_impl.h
// Do not include directly

#include <ctype.h>
#include <exception>

#ifdef _MSC_VER
#pragma warning (disable:4127) // warning C4127: conditional expression is constant
#endif

namespace basisu
{
	// A container operation has internally panicked in an unrecoverable way.
	// Either an allocation has failed, or a range or consistency check has failed.
#ifdef _MSC_VER
	__declspec(noreturn)
#else
	[[noreturn]] 
#endif
	void container_abort(const char* pMsg, ...)
	{
		assert(0);

		va_list args;
		va_start(args, pMsg);

		char buf[1024] = {};

#ifdef _MSC_VER
		vsprintf_s(buf, sizeof(buf), pMsg, args);
#else
		vsnprintf(buf, sizeof(buf), pMsg, args);
#endif
		va_end(args);

		fputs(buf, stderr);

		std::terminate();
	}

	bool elemental_vector::increase_capacity(size_t min_new_capacity, bool grow_hint, size_t element_size, object_mover pMover, bool nofail_flag)
	{
		assert(m_size <= m_capacity);
		assert(min_new_capacity >= m_size);
		assert(element_size);
		
		// Basic sanity check min_new_capacity
		if (!can_fit_into_size_t((uint64_t)min_new_capacity * element_size))
		{
			assert(0);
			
			if (nofail_flag)
				return false;

			container_abort("elemental_vector::increase_capacity: requesting too many elements\n");
		}

		// Check for sane library limits
		if (sizeof(void*) == sizeof(uint64_t))
		{
			// 16 GB
			assert(min_new_capacity < (0x400000000ULL / element_size));
		}
		else
		{
			// ~1.99 GB
			assert(min_new_capacity < (0x7FFF0000U / element_size));
		}

		// If vector is already large enough just return.
		if (m_capacity >= min_new_capacity)
			return true;

		uint64_t new_capacity_u64 = min_new_capacity;

		if ((grow_hint) && (!helpers::is_power_of_2(new_capacity_u64)))
		{
			new_capacity_u64 = helpers::next_pow2(new_capacity_u64);

			if (!can_fit_into_size_t(new_capacity_u64))
			{
				assert(0);

				if (nofail_flag)
					return false;

				container_abort("elemental_vector::increase_capacity: vector too large\n");
			}
		}

		const uint64_t desired_size_u64 = element_size * new_capacity_u64;

		if (!can_fit_into_size_t(desired_size_u64))
		{
			assert(0);

			if (nofail_flag)
				return false;

			container_abort("elemental_vector::increase_capacity: vector too large\n");
		}

		const size_t desired_size = static_cast<size_t>(desired_size_u64);
						
		size_t actual_size = 0;
		BASISU_NOTE_UNUSED(actual_size);

		if (!pMover)
		{
			void* new_p = realloc(m_p, desired_size);
			if (!new_p)
			{
				assert(0);

				if (nofail_flag)
					return false;

				container_abort("elemental_vector::increase_capacity: realloc() failed allocating %zu bytes", desired_size);
			}

#if BASISU_VECTOR_DETERMINISTIC
			actual_size = desired_size;
#elif defined(_MSC_VER)
			actual_size = _msize(new_p);
#elif HAS_MALLOC_USABLE_SIZE
			actual_size = malloc_usable_size(new_p);
#else
			actual_size = desired_size;
#endif
			m_p = new_p;
		}
		else
		{
			void* new_p = malloc(desired_size);
			if (!new_p)
			{
				assert(0);
				if (nofail_flag)
					return false;

				container_abort("elemental_vector::increase_capacity: malloc() failed allocating %zu bytes", desired_size);
			}

#if BASISU_VECTOR_DETERMINISTIC
			actual_size = desired_size;
#elif defined(_MSC_VER)
			actual_size = _msize(new_p);
#elif HAS_MALLOC_USABLE_SIZE
			actual_size = malloc_usable_size(new_p);
#else
			actual_size = desired_size;
#endif

			(*pMover)(new_p, m_p, m_size);

			if (m_p)
				free(m_p);

			m_p = new_p;
		}

#if BASISU_VECTOR_DETERMINISTIC
		m_capacity = static_cast<size_t>(new_capacity_u64);
#else
		if (actual_size > desired_size)
			m_capacity = static_cast<size_t>(actual_size / element_size);
		else
			m_capacity = static_cast<size_t>(new_capacity_u64);
#endif

		return true;
	}

#if BASISU_HASHMAP_TEST

#define HASHMAP_TEST_VERIFY(c) do { if (!(c)) handle_hashmap_test_verify_failure(__LINE__); } while(0)

	static void handle_hashmap_test_verify_failure(int line)
	{
		container_abort("HASHMAP_TEST_VERIFY() faild on line %i\n", line);
	}

	class counted_obj
	{
	public:
		counted_obj(uint32_t v = 0) :
			m_val(v)
		{
			m_count++;
		}

		counted_obj(const counted_obj& obj) :
			m_val(obj.m_val)
		{
			if (m_val != UINT64_MAX)
				m_count++;
		}

		counted_obj(counted_obj&& obj) :
			m_val(obj.m_val)
		{
			obj.m_val = UINT64_MAX;
		}

		counted_obj& operator= (counted_obj&& rhs)
		{
			if (this != &rhs)
			{
				m_val = rhs.m_val;
				rhs.m_val = UINT64_MAX;
			}
			return *this;
		}

		~counted_obj()
		{
			if (m_val != UINT64_MAX)
			{
				assert(m_count > 0);
				m_count--;
			}
		}

		static uint32_t m_count;

		uint64_t m_val;

		operator size_t() const { return (size_t)m_val; }

		bool operator== (const counted_obj& rhs) const { return m_val == rhs.m_val; }
		bool operator== (const uint32_t rhs) const { return m_val == rhs; }

	};

	uint32_t counted_obj::m_count;

	static uint32_t urand32()
	{
		uint32_t a = rand();
		uint32_t b = rand() << 15;
		uint32_t c = rand() << (32 - 15);
		return a ^ b ^ c;
	}

	static int irand32(int l, int h)
	{
		assert(l < h);
		if (l >= h)
			return l;

		uint32_t range = static_cast<uint32_t>(h - l);

		uint32_t rnd = urand32();

		uint32_t rnd_range = static_cast<uint32_t>((((uint64_t)range) * ((uint64_t)rnd)) >> 32U);

		int result = l + rnd_range;
		assert((result >= l) && (result < h));
		return result;
	}

	void hash_map_test()
	{
		{
			basisu::hash_map<uint32_t> s;
			uint_vec k;

			for (uint32_t i = 0; i < 1000000; i++)
			{
				s.insert(i);
				k.push_back(i);
			}
						
			for (uint32_t i = 0; i < k.size(); i++)
			{
				uint32_t r = rand() ^ (rand() << 15);

				uint32_t j = i + (r % (k.size() - i));

				std::swap(k[i], k[j]);
			}

			basisu::hash_map<uint32_t> s1(s);

			for (uint32_t i = 0; i < 1000000; i++)
			{
				auto res = s.find(i);
				HASHMAP_TEST_VERIFY(res != s.end());
				HASHMAP_TEST_VERIFY(res->first == i);
				s.erase(i);
			}

			for (uint32_t it = 0; it < 1000000; it++)
			{
				uint32_t i = k[it];

				auto res = s1.find(i);
				HASHMAP_TEST_VERIFY(res != s.end());
				HASHMAP_TEST_VERIFY(res->first == i);
				s1.erase(i);
			}

			for (uint32_t i = 0; i < 1000000; i++)
			{
				auto res = s.find(i);
				HASHMAP_TEST_VERIFY(res == s.end());

				auto res1 = s1.find(i);
				HASHMAP_TEST_VERIFY(res1 == s1.end());
			}

			HASHMAP_TEST_VERIFY(s.empty());
			HASHMAP_TEST_VERIFY(s1.empty());
		}

		{
			typedef basisu::hash_map< uint32_t, basisu::vector<uint32_t> > hm;
			hm q;
			
			basisu::vector<uint32_t> a, b;
			a.push_back(1);
			b.push_back(2);
			b.push_back(3);

			basisu::vector<uint32_t> c(b);

			hm::insert_result ir;
			q.try_insert(ir, 1, std::move(a));
			q.try_insert(ir, 2, std::move(b));
			q.try_insert(ir, std::make_pair(3, c));
		}

		{
			typedef basisu::hash_map<counted_obj, counted_obj> my_hash_map;
			my_hash_map m;
			counted_obj a, b;
			m.insert(std::move(a), std::move(b));
		}

		{
			basisu::hash_map<uint64_t, uint64_t> k;
			basisu::hash_map<uint64_t, uint64_t> l;
			std::swap(k, l);

			k.begin();
			k.end();
			k.clear();
			k.empty();
			k.erase(0);
			k.insert(0, 1);
			k.find(0);
			k.get_equals();
			k.get_hasher();
			k.get_table_size();
			k.reset();
			k.reserve(1);
			k = l;
			k.set_equals(l.get_equals());
			k.set_hasher(l.get_hasher());
			k.get_table_size();
		}

		uint32_t seed = 0;
		for (; ; )
		{
			seed++;

			typedef basisu::hash_map<counted_obj, counted_obj> my_hash_map;
			my_hash_map m;

			const uint32_t n = irand32(1, 100000);

			printf("%u\n", n);

			srand(seed); // r1.seed(seed);

			basisu::vector<int> q;

			uint32_t count = 0;
			for (uint32_t i = 0; i < n; i++)
			{
				uint32_t v = urand32() & 0x7FFFFFFF;
				my_hash_map::insert_result res = m.insert(counted_obj(v), counted_obj(v ^ 0xdeadbeef));
				if (res.second)
				{
					count++;
					q.push_back(v);
				}
			}

			HASHMAP_TEST_VERIFY(m.size() == count);

			srand(seed);

			my_hash_map cm(m);
			m.clear();
			m = cm;
			cm.reset();

			for (uint32_t i = 0; i < n; i++)
			{
				uint32_t v = urand32() & 0x7FFFFFFF;
				my_hash_map::const_iterator it = m.find(counted_obj(v));
				HASHMAP_TEST_VERIFY(it != m.end());
				HASHMAP_TEST_VERIFY(it->first == v);
				HASHMAP_TEST_VERIFY(it->second == (v ^ 0xdeadbeef));
			}

			for (uint32_t t = 0; t < 2; t++)
			{
				const uint32_t nd = irand32(1, q.size_u32() + 1);
				for (uint32_t i = 0; i < nd; i++)
				{
					uint32_t p = irand32(0, q.size_u32());

					int k = q[p];
					if (k >= 0)
					{
						q[p] = -k - 1;

						bool s = m.erase(counted_obj(k));
						HASHMAP_TEST_VERIFY(s);
					}
				}

				typedef basisu::hash_map<uint32_t, empty_type> uint_hash_set;
				uint_hash_set s;

				for (uint32_t i = 0; i < q.size(); i++)
				{
					int v = q[i];

					if (v >= 0)
					{
						my_hash_map::const_iterator it = m.find(counted_obj(v));
						HASHMAP_TEST_VERIFY(it != m.end());
						HASHMAP_TEST_VERIFY(it->first == (uint32_t)v);
						HASHMAP_TEST_VERIFY(it->second == ((uint32_t)v ^ 0xdeadbeef));

						s.insert(v);
					}
					else
					{
						my_hash_map::const_iterator it = m.find(counted_obj(-v - 1));
						HASHMAP_TEST_VERIFY(it == m.end());
					}
				}

				uint32_t found_count = 0;
				for (my_hash_map::const_iterator it = m.begin(); it != m.end(); ++it)
				{
					HASHMAP_TEST_VERIFY(it->second == ((uint32_t)it->first ^ 0xdeadbeef));

					uint_hash_set::const_iterator fit(s.find((uint32_t)it->first));
					HASHMAP_TEST_VERIFY(fit != s.end());

					HASHMAP_TEST_VERIFY(fit->first == it->first);

					found_count++;
				}

				HASHMAP_TEST_VERIFY(found_count == s.size());
			}

			HASHMAP_TEST_VERIFY(counted_obj::m_count == m.size() * 2);
		}
	}

#endif // BASISU_HASHMAP_TEST

	// String formatting

	bool fmt_variant::to_string(std::string& res, std::string& fmt) const
	{
		res.resize(0);

		// Scan for allowed formatting characters.
		for (size_t i = 0; i < fmt.size(); i++)
		{
			const char c = fmt[i];

			if (isdigit(c) || (c == '.') || (c == ' ') || (c == '#') || (c == '+') || (c == '-'))
				continue;

			if (isalpha(c))
			{
				if ((i + 1) == fmt.size())
					continue;
			}

			return false;
		}

		if (fmt.size() && (fmt.back() == 'c'))
		{
			if ((m_type == variant_type::cI32) || (m_type == variant_type::cU32))
			{
				if (m_u32 > 255)
					return false;

				// Explictly allowing caller to pass in a char of 0, which is ignored.
				if (m_u32)
					res.push_back((uint8_t)m_u32);
				return true;
			}
			else
				return false;
		}

		switch (m_type)
		{
		case variant_type::cInvalid:
		{
			return false;
		}
		case variant_type::cI32:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if ((e != 'x') && (e != 'X') && (e != 'i') && (e != 'd') && (e != 'u'))
						return false;
				}
				else
				{
					fmt += "i";
				}

				res = string_format((std::string("%") + fmt).c_str(), m_i32);
			}
			else
			{
				res = string_format("%i", m_i32);
			}
			break;
		}
		case variant_type::cU32:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if ((e != 'x') && (e != 'X') && (e != 'i') && (e != 'd') && (e != 'u'))
						return false;
				}
				else
				{
					fmt += "u";
				}

				res = string_format((std::string("%") + fmt).c_str(), m_u32);
			}
			else
			{
				res = string_format("%u", m_u32);
			}
			break;
		}
		case variant_type::cI64:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if (e == 'x')
					{
						fmt.pop_back();
						fmt += PRIx64;
					}
					else if (e == 'X')
					{
						fmt.pop_back();
						fmt += PRIX64;
					}
					else
						return false;
				}
				else
				{
					fmt += PRId64;
				}

				res = string_format((std::string("%") + fmt).c_str(), m_i64);
			}
			else
			{
				res = string_format("%" PRId64, m_i64);
			}
			break;
		}
		case variant_type::cU64:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if (e == 'x')
					{
						fmt.pop_back();
						fmt += PRIx64;
					}
					else if (e == 'X')
					{
						fmt.pop_back();
						fmt += PRIX64;
					}
					else
						return false;
				}
				else
				{
					fmt += PRIu64;
				}

				res = string_format((std::string("%") + fmt).c_str(), m_u64);
			}
			else
			{
				res = string_format("%" PRIu64, m_u64);
			}
			break;
		}
		case variant_type::cFlt:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if ((e != 'f') && (e != 'g') && (e != 'e') && (e != 'E'))
						return false;
				}
				else
				{
					fmt += "f";
				}

				res = string_format((std::string("%") + fmt).c_str(), m_flt);
			}
			else
			{
				res = string_format("%f", m_flt);
			}
			break;
		}
		case variant_type::cDbl:
		{
			if (fmt.size())
			{
				int e = fmt.back();
				if (isalpha(e))
				{
					if ((e != 'f') && (e != 'g') && (e != 'e') && (e != 'E'))
						return false;
				}
				else
				{
					fmt += "f";
				}

				res = string_format((std::string("%") + fmt).c_str(), m_dbl);
			}
			else
			{
				res = string_format("%f", m_dbl);
			}
			break;
		}
		case variant_type::cStrPtr:
		{
			if (fmt.size())
				return false;
			if (!m_pStr)
				return false;
			res = m_pStr;
			break;
		}
		case variant_type::cBool:
		{
			if (fmt.size())
				return false;
			res = m_bool ? "true" : "false";
			break;
		}
		case variant_type::cStdStr:
		{
			if (fmt.size())
				return false;
			res = m_str;
			break;
		}
		default:
		{
			return false;
		}
		}

		return true;
	}

	bool fmt_variants(std::string& res, const char* pFmt, const fmt_variant_vec& variants)
	{
		res.resize(0);

		// Must specify a format string
		if (!pFmt)
		{
			assert(0);
			return false;
		}

		// Check format string's length
		const size_t fmt_len = strlen(pFmt);
		if (!fmt_len)
		{
			if (variants.size())
			{
				assert(0);
				return false;
			}
			return true;
		}

		// Wildly estimate output length
		res.reserve(fmt_len + 32);

		std::string var_fmt;
		var_fmt.reserve(16);

		std::string tmp;
		tmp.reserve(16);

		size_t variant_index = 0;
		bool inside_brackets = false;
		const char* p = pFmt;

		while (*p)
		{
			const uint8_t c = *p++;

			if (inside_brackets)
			{
				if (c == '}')
				{
					inside_brackets = false;

					if (variant_index >= variants.size())
					{
						assert(0);
						return false;
					}

					if (!variants[variant_index].to_string(tmp, var_fmt))
					{
						assert(0);
						return false;
					}

					res += tmp;

					variant_index++;
				}
				else
				{
					// Check for forbidden formatting characters.
					if ((c == '*') || (c == 'n') || (c == '%'))
					{
						assert(0);
						return false;
					}

					var_fmt.push_back(c);
				}
			}
			else if (c == '{')
			{
				// Check for escaped '{'
				if (*p == '{')
				{
					res.push_back((char)c);
					p++;
				}
				else
				{
					inside_brackets = true;
					var_fmt.resize(0);
				}
			}
			else
			{
				res.push_back((char)c);
			}
		}

		if (inside_brackets)
		{
			assert(0);
			return false;
		}

		if (variant_index != variants.size())
		{
			assert(0);
			return false;
		}

		return true;
	}

} // namespace basisu
