// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdFixedSizeString.h"

#include <cstdarg>
#include <cctype>
#include <cstring>


PSD_NAMESPACE_BEGIN

namespace util
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::Assign(const char* const str)
	{
		m_length = strlen(str);
		PSD_ASSERT(m_length < CAPACITY, "String \"%s\" does not fit into FixedSizeString.", str);

		memcpy(m_string, str, m_length+1);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::Append(const char* str)
	{
		Append(str, strlen(str));
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::Append(const char* str, size_t count)
	{
		PSD_ASSERT(m_length + count < CAPACITY, "Cannot append character(s) from string \"%s\". Not enough space left.", str);
		memcpy(m_string + m_length, str, count);
		m_length += count;
		m_string[m_length] = '\0';
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::Clear(void)
	{
		m_length = 0;
		m_string[0] = '\0';
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	bool FixedSizeString::IsEqual(const char* other) const
	{
		return (strcmp(m_string, other) == 0);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::ToLower(void)
	{
		for (unsigned int i=0; i < m_length; ++i)
		{
			m_string[i] = static_cast<char>(tolower(m_string[i]));
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void FixedSizeString::ToUpper(void)
	{
		for (unsigned int i=0; i < m_length; ++i)
		{
			m_string[i] = static_cast<char>(toupper(m_string[i]));
		}
	}
}

PSD_NAMESPACE_END
