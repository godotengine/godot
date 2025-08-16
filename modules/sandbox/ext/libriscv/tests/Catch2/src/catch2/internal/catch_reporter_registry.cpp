
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/reporters/catch_reporter_automake.hpp>
#include <catch2/reporters/catch_reporter_compact.hpp>
#include <catch2/reporters/catch_reporter_console.hpp>
#include <catch2/reporters/catch_reporter_json.hpp>
#include <catch2/reporters/catch_reporter_junit.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_sonarqube.hpp>
#include <catch2/reporters/catch_reporter_tap.hpp>
#include <catch2/reporters/catch_reporter_teamcity.hpp>
#include <catch2/reporters/catch_reporter_xml.hpp>

namespace Catch {
    struct ReporterRegistry::ReporterRegistryImpl {
        std::vector<Detail::unique_ptr<EventListenerFactory>> listeners;
        std::map<std::string, IReporterFactoryPtr, Detail::CaseInsensitiveLess>
            factories;
    };

    ReporterRegistry::ReporterRegistry():
        m_impl( Detail::make_unique<ReporterRegistryImpl>() ) {
        // Because it is impossible to move out of initializer list,
        // we have to add the elements manually
        m_impl->factories["Automake"] =
            Detail::make_unique<ReporterFactory<AutomakeReporter>>();
        m_impl->factories["compact"] =
            Detail::make_unique<ReporterFactory<CompactReporter>>();
        m_impl->factories["console"] =
            Detail::make_unique<ReporterFactory<ConsoleReporter>>();
        m_impl->factories["JUnit"] =
            Detail::make_unique<ReporterFactory<JunitReporter>>();
        m_impl->factories["SonarQube"] =
            Detail::make_unique<ReporterFactory<SonarQubeReporter>>();
        m_impl->factories["TAP"] =
            Detail::make_unique<ReporterFactory<TAPReporter>>();
        m_impl->factories["TeamCity"] =
            Detail::make_unique<ReporterFactory<TeamCityReporter>>();
        m_impl->factories["XML"] =
            Detail::make_unique<ReporterFactory<XmlReporter>>();
        m_impl->factories["JSON"] =
            Detail::make_unique<ReporterFactory<JsonReporter>>();
    }

    ReporterRegistry::~ReporterRegistry() = default;

    IEventListenerPtr
    ReporterRegistry::create( std::string const& name,
                              ReporterConfig&& config ) const {
        auto it = m_impl->factories.find( name );
        if ( it == m_impl->factories.end() ) return nullptr;
        return it->second->create( CATCH_MOVE( config ) );
    }

    void ReporterRegistry::registerReporter( std::string const& name,
                                             IReporterFactoryPtr factory ) {
        CATCH_ENFORCE( name.find( "::" ) == name.npos,
                       "'::' is not allowed in reporter name: '" + name +
                           '\'' );
        auto ret = m_impl->factories.emplace( name, CATCH_MOVE( factory ) );
        CATCH_ENFORCE( ret.second,
                       "reporter using '" + name +
                           "' as name was already registered" );
    }
    void ReporterRegistry::registerListener(
        Detail::unique_ptr<EventListenerFactory> factory ) {
        m_impl->listeners.push_back( CATCH_MOVE( factory ) );
    }

    std::map<std::string,
             IReporterFactoryPtr,
             Detail::CaseInsensitiveLess> const&
    ReporterRegistry::getFactories() const {
        return m_impl->factories;
    }

    std::vector<Detail::unique_ptr<EventListenerFactory>> const&
    ReporterRegistry::getListeners() const {
        return m_impl->listeners;
    }
} // namespace Catch
