/*
 * Copyright 2015-2020 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_ERROR_HANDLING
#define SPIRV_CROSS_ERROR_HANDLING

#include <stdio.h>
#include <stdlib.h>
#include <string>
#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#include <stdexcept>
#endif

#ifdef SPIRV_CROSS_NAMESPACE_OVERRIDE
#define SPIRV_CROSS_NAMESPACE SPIRV_CROSS_NAMESPACE_OVERRIDE
#else
#define SPIRV_CROSS_NAMESPACE spirv_cross
#endif

namespace SPIRV_CROSS_NAMESPACE
{
#ifdef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#if !defined(_MSC_VER) || defined(__clang__)
[[noreturn]]
#elif defined(_MSC_VER)
__declspec(noreturn)
#endif
inline void
report_and_abort(const std::string &msg)
{
#ifdef NDEBUG
	(void)msg;
#else
	fprintf(stderr, "There was a compiler error: %s\n", msg.c_str());
#endif
	fflush(stderr);
	abort();
}

#define SPIRV_CROSS_THROW(x) report_and_abort(x)
#else
class CompilerError : public std::runtime_error
{
public:
	explicit CompilerError(const std::string &str)
	    : std::runtime_error(str)
	{
	}
};

#define SPIRV_CROSS_THROW(x) throw CompilerError(x)
#endif

// MSVC 2013 does not have noexcept. We need this for Variant to get move constructor to work correctly
// instead of copy constructor.
// MSVC 2013 ignores that move constructors cannot throw in std::vector, so just don't define it.
#if defined(_MSC_VER) && _MSC_VER < 1900
#define SPIRV_CROSS_NOEXCEPT
#else
#define SPIRV_CROSS_NOEXCEPT noexcept
#endif

#if __cplusplus >= 201402l
#define SPIRV_CROSS_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(__GNUC__)
#define SPIRV_CROSS_DEPRECATED(reason) __attribute__((deprecated))
#elif defined(_MSC_VER)
#define SPIRV_CROSS_DEPRECATED(reason) __declspec(deprecated(reason))
#else
#define SPIRV_CROSS_DEPRECATED(reason)
#endif
} // namespace SPIRV_CROSS_NAMESPACE

#endif
