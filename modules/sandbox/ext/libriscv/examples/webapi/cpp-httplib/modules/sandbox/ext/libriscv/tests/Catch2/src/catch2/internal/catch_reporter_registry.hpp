
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_REGISTRY_HPP_INCLUDED
#define CATCH_REPORTER_REGISTRY_HPP_INCLUDED

#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>

#include <map>
#include <string>
#include <vector>

namespace Catch {

    class IEventListener;
    using IEventListenerPtr = Detail::unique_ptr<IEventListener>;
    class IReporterFactory;
    using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;
    struct ReporterConfig;
    class EventListenerFactory;

    class ReporterRegistry {
        struct ReporterRegistryImpl;
        Detail::unique_ptr<ReporterRegistryImpl> m_impl;

    public:
        ReporterRegistry();
        ~ReporterRegistry(); // = default;

        IEventListenerPtr create( std::string const& name,
                                  ReporterConfig&& config ) const;

        void registerReporter( std::string const& name,
                               IReporterFactoryPtr factory );

        void
        registerListener( Detail::unique_ptr<EventListenerFactory> factory );

        std::map<std::string,
                 IReporterFactoryPtr,
                 Detail::CaseInsensitiveLess> const&
        getFactories() const;

        std::vector<Detail::unique_ptr<EventListenerFactory>> const&
        getListeners() const;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_REGISTRY_HPP_INCLUDED
