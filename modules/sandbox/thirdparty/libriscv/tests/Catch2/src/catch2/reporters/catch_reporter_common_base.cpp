
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/reporters/catch_reporter_common_base.hpp>

#include <catch2/reporters/catch_reporter_helpers.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_istream.hpp>


namespace Catch {
    ReporterBase::ReporterBase( ReporterConfig&& config ):
        IEventListener( config.fullConfig() ),
        m_wrapped_stream( CATCH_MOVE(config).takeStream() ),
        m_stream( m_wrapped_stream->stream() ),
        m_colour( makeColourImpl( config.colourMode(), m_wrapped_stream.get() ) ),
        m_customOptions( config.customOptions() )
    {}

    ReporterBase::~ReporterBase() = default;

    void ReporterBase::listReporters(
        std::vector<ReporterDescription> const& descriptions ) {
        defaultListReporters(m_stream, descriptions, m_config->verbosity());
    }

    void ReporterBase::listListeners(
        std::vector<ListenerDescription> const& descriptions ) {
        defaultListListeners( m_stream, descriptions );
    }

    void ReporterBase::listTests(std::vector<TestCaseHandle> const& tests) {
        defaultListTests(m_stream,
                         m_colour.get(),
                         tests,
                         m_config->hasTestFilters(),
                         m_config->verbosity());
    }

    void ReporterBase::listTags(std::vector<TagInfo> const& tags) {
        defaultListTags( m_stream, tags, m_config->hasTestFilters() );
    }

} // namespace Catch
