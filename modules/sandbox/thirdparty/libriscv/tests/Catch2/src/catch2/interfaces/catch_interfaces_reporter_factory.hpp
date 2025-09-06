
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_REPORTER_FACTORY_HPP_INCLUDED
#define CATCH_INTERFACES_REPORTER_FACTORY_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <string>

namespace Catch {

    struct ReporterConfig;
    class IConfig;
    class IEventListener;
    using IEventListenerPtr = Detail::unique_ptr<IEventListener>;


    class IReporterFactory {
    public:
        virtual ~IReporterFactory(); // = default

        virtual IEventListenerPtr
        create( ReporterConfig&& config ) const = 0;
        virtual std::string getDescription() const = 0;
    };
    using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;

    class EventListenerFactory {
    public:
        virtual ~EventListenerFactory(); // = default
        virtual IEventListenerPtr create( IConfig const* config ) const = 0;
        //! Return a meaningful name for the listener, e.g. its type name
        virtual StringRef getName() const = 0;
        //! Return listener's description if available
        virtual std::string getDescription() const = 0;
    };
} // namespace Catch

#endif // CATCH_INTERFACES_REPORTER_FACTORY_HPP_INCLUDED
