/*
 * mptCRC.h
 * --------
 * Purpose: generic CRC implementation
 * Notes  : (currently none)
 * Authors: Joern Heusipp
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

OPENMPT_NAMESPACE_BEGIN

namespace mpt
{

namespace checksum
{

template <typename T, T polynomial, T initial, T resultXOR, bool reverseData>
class crc
{

public:
	
	typedef crc self_type;
	typedef T value_type;
	typedef uint8 byte_type;

	static const std::size_t size_bytes = sizeof(value_type);
	static const std::size_t size_bits = sizeof(value_type) * 8;
	static const value_type top_bit = static_cast<value_type>(1) << ((sizeof(value_type) * 8) - 1);

private:
	
	template <typename Tint>
	static inline Tint reverse(Tint value)
	{
		const std::size_t bits = sizeof(Tint) * 8;
		Tint result = 0;
		for(std::size_t i = 0; i < bits; ++i)
		{
			result <<= 1;
			result |= static_cast<Tint>(value & 0x1);
			value >>= 1;
		}
		return result;
	}

	static inline value_type calculate_table_entry(byte_type pos)
	{
		value_type value = 0;
		value = (static_cast<value_type>(reverseData ? reverse(pos) : pos) << (size_bits - 8));
		for(std::size_t bit = 0; bit < 8; ++bit)
		{
			if(value & top_bit)
			{
				value = (value << 1) ^ polynomial;
			} else
			{
				value = (value << 1);
			}
		}
		value = (reverseData ? reverse(value) : value);
		return value;
	}

private:

	static value_type table[256];
	
	static inline void fill_table()
	{
		for(std::size_t i = 0; i < 256; ++i)
		{
			table[i] = calculate_table_entry(static_cast<byte_type>(i));
		}
	}
	
	struct table_filler
	{
		inline table_filler()
		{
			self_type::fill_table();
		}
	};

	static inline void init()
	{
		static table_filler table_filler;
	}

private:

	inline value_type read_table(byte_type pos) const
	{
		return table[pos];
	}

private:

	value_type value;

public:

	crc()
		: value(initial)
	{
		init();
	}

	inline void processByte(byte_type byte)
	{
		MPT_CONSTANT_IF(reverseData)
		{
			value = (value >> 8) ^ read_table(static_cast<byte_type>((value & 0xff) ^ byte));
		} else
		{
			value = (value << 8) ^ read_table(static_cast<byte_type>(((value >> (size_bits - 8)) & 0xff) ^ byte));
		}
	}

	inline value_type result() const
	{
		return (value ^ resultXOR);
	}

public:

	inline operator value_type () const
	{
		return result();
	}

	inline crc & process(char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	inline crc & process(signed char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	inline crc & process(unsigned char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	template <typename InputIt>
	crc & process(InputIt beg, InputIt end)
	{
		for(InputIt it = beg; it != end; ++it)
		{
			static_assert(sizeof(*it) == 1, "1 byte type required");
			process(*it);
		}
		return *this;
	}

	template <typename Container>
	inline crc & process(const Container &data)
	{
		operator () (data.begin(), data.end());
		return *this;
	}

	inline crc & operator () (char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	inline crc & operator () (signed char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	inline crc & operator () (unsigned char c)
	{
		processByte(static_cast<byte_type>(c));
		return *this;
	}

	template <typename InputIt>
	crc & operator () (InputIt beg, InputIt end)
	{
		for(InputIt it = beg; it != end; ++it)
		{
			static_assert(sizeof(*it) == 1, "1 byte type required");
			operator () (*it);
		}
		return *this;
	}

	template <typename Container>
	inline crc & operator () (const Container &data)
	{
		operator () (data.begin(), data.end());
		return *this;
	}

	template <typename InputIt>
	crc(InputIt beg, InputIt end)
		: value(initial)
	{
		init();
		for(InputIt it = beg; it != end; ++it)
		{
			static_assert(sizeof(*it) == 1, "1 byte type required");
			process(*it);
		}
	}

	template <typename Container>
	inline crc(const Container &data)
		: value(initial)
	{
		init();
		process(data.begin(), data.end());
	}

};

template <typename T, T polynomial, T initial, T resultXOR, bool reverseData>
typename crc<T, polynomial, initial, resultXOR, reverseData>::value_type crc<T, polynomial, initial, resultXOR, reverseData>::table[256];

typedef crc<uint16, 0x8005, 0, 0, true> crc16;
typedef crc<uint32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, true> crc32;
typedef crc<uint32, 0x04C11DB7, 0, 0, false> crc32_ogg;
typedef crc<uint32, 0x1EDC6F41, 0xFFFFFFFF, 0xFFFFFFFF, true> crc32c;
typedef crc<uint64, 0xAD93D23594C935A9ull, 0xFFFFFFFFFFFFFFFFull, 0, true> crc64_jones;

} // namespace checksum

using mpt::checksum::crc32;
using mpt::checksum::crc32_ogg;

} // namespace mpt

OPENMPT_NAMESPACE_END
