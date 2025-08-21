
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_EXCEPTION_HPP_INCLUDED
#define CATCH_INTERFACES_EXCEPTION_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>

#include <string>
#include <vector>

namespace Catch {
    using exceptionTranslateFunction = std::string(*)();

    class IExceptionTranslator;
    using ExceptionTranslators = std::vector<Detail::unique_ptr<IExceptionTranslator const>>;

    class IExceptionTranslator {
    public:
        virtual ~IExceptionTranslator(); // = default
        virtual std::string translate( ExceptionTranslators::const_iterator it, ExceptionTranslators::const_iterator itEnd ) const = 0;
    };

    class IExceptionTranslatorRegistry {
    public:
        virtual ~IExceptionTranslatorRegistry(); // = default
        virtual std::string translateActiveException() const = 0;
    };

} // namespace Catch

#endif // CATCH_INTERFACES_EXCEPTION_HPP_INCLUDED
