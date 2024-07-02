// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdAssert.h"


PSD_NAMESPACE_BEGIN

namespace util
{
	/// \ingroup Util
	/// \class FixedSizeString
	/// \brief A class representing a string containing a fixed number of characters.
	/// \details This class offers functionality similar to other string implementations, such as std::string.
	/// However, as the name implies, a FixedSizeString will never grow, shrink, or allocate any memory dynamically. It can
	/// often be used in situations where the maximum length of a string can be limited, such as when dealing with filenames,
	/// in the logging system, when parsing files, etc. A FixedSizeString should be preferred to other implementations
	/// in such cases.
	class FixedSizeString
	{
	public:
		/// A constant denoting the capacity of the string.
		static const size_t CAPACITY = 1024;


		/// \brief Clears the string such that GetLength() yields 0.
		/// \remark After calling Clear(), no assumptions about the characters stored in the internal array should be made.
		void Clear(void);


		/// Assigns a string.
		void Assign(const char* const str);


		/// Appends a string.
		void Append(const char* str);

		/// Appends part of another string.
		void Append(const char* str, size_t count);


		/// Returns whether the string equals a given string.
		bool IsEqual(const char* other) const;


		/// Returns the i-th character.
		inline char& operator[](size_t i)
		{
			// allow access to the null terminator
			PSD_ASSERT(i <= m_length, "Character cannot be accessed. Subscript out of range.");
			return m_string[i];
		}

		/// Returns the i-th character.
		inline const char& operator[](size_t i) const
		{
			// allow access to the null terminator
			PSD_ASSERT(i <= m_length, "Character cannot be accessed. Subscript out of range.");
			return m_string[i];
		}

		/// Returns the C-style string.
		inline const char* c_str(void) const
		{
			return m_string;
		}

		/// Returns the length of the string, not counting the terminating null.
		inline size_t GetLength(void) const
		{
			return m_length;
		}

		/// Converts all characters to lower-case characters.
		void ToLower(void);

		/// Converts all characters to upper-case characters.
		void ToUpper(void);

	private:
		char m_string[CAPACITY];
		size_t m_length;
	};
}

PSD_NAMESPACE_END
