#pragma once
#include <istream>
#include <string>
#include "stdint_defs.h"
#include "pe_structures.h"

namespace pe_bliss
{
class pe_utils
{
public:
	//Returns true if string "data" with maximum length "raw_length" is null-terminated
	template<typename T>
	static bool is_null_terminated(const T* data, size_t raw_length)
	{
		raw_length /= sizeof(T);
		for(size_t l = 0; l < raw_length; l++)
		{
			if(data[l] == static_cast<T>(L'\0'))
				return true;
		}

		return false;
	}

	//Helper template function to strip nullbytes in the end of string
	template<typename T>
	static void strip_nullbytes(std::basic_string<T>& str)
	{
		while(!*(str.end() - 1) && !str.empty())
			str.erase(str.length() - 1);
	}

	//Helper function to determine if number is power of 2
	template<typename T>
	static inline bool is_power_of_2(T x)
	{
		return !(x & (x - 1));
	}

	//Helper function to align number down
	template<typename T>
	static inline T align_down(T x, uint32_t align)
	{
		return x & ~(static_cast<T>(align) - 1);
	}

	//Helper function to align number up
	template<typename T>
	static inline T align_up(T x, uint32_t align)
	{
		return (x & static_cast<T>(align - 1)) ? align_down(x, align) + static_cast<T>(align) : x;
	}

	//Returns true if sum of two unsigned integers is safe (no overflow occurs)
	static inline bool is_sum_safe(uint32_t a, uint32_t b)
	{
		return a <= static_cast<uint32_t>(-1) - b;
	}

	//Two gigabytes value in bytes
	static const uint32_t two_gb = 0x80000000;
	static const uint32_t max_dword = 0xFFFFFFFF;
	static const uint32_t max_word = 0x0000FFFF;
	static const double log_2; //instead of using M_LOG2E

	//Returns stream size
	static std::streamoff get_file_size(std::istream& file);
	
#ifndef PE_BLISS_WINDOWS
public:
	static const u16string to_ucs2(const std::wstring& str);
	static const std::wstring from_ucs2(const u16string& str);
#endif

private:
	pe_utils();
	pe_utils(pe_utils&);
	pe_utils& operator=(const pe_utils&);
};

//Windows GUID comparison
bool operator==(const pe_win::guid& guid1, const pe_win::guid& guid2);
}
