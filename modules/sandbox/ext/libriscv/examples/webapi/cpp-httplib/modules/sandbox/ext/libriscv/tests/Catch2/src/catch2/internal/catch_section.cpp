
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_section.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_uncaught_exceptions.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {

    Section::Section( SectionInfo&& info ):
        m_info( CATCH_MOVE( info ) ),
        m_sectionIncluded(
            getResultCapture().sectionStarted( m_info.name, m_info.lineInfo, m_assertions ) ) {
        // Non-"included" sections will not use the timing information
        // anyway, so don't bother with the potential syscall.
        if (m_sectionIncluded) {
            m_timer.start();
        }
    }

    Section::Section( SourceLineInfo const& _lineInfo,
                      StringRef _name,
                      const char* const ):
        m_info( { "invalid", static_cast<std::size_t>( -1 ) }, std::string{} ),
        m_sectionIncluded(
            getResultCapture().sectionStarted( _name, _lineInfo, m_assertions ) ) {
        // We delay initialization the SectionInfo member until we know
        // this section needs it, so we avoid allocating std::string for name.
        // We also delay timer start to avoid the potential syscall unless we
        // will actually use the result.
        if ( m_sectionIncluded ) {
            m_info.name = static_cast<std::string>( _name );
            m_info.lineInfo = _lineInfo;
            m_timer.start();
        }
    }

    Section::~Section() {
        if( m_sectionIncluded ) {
            SectionEndInfo endInfo{ CATCH_MOVE(m_info), m_assertions, m_timer.getElapsedSeconds() };
            if ( uncaught_exceptions() ) {
                getResultCapture().sectionEndedEarly( CATCH_MOVE(endInfo) );
            } else {
                getResultCapture().sectionEnded( CATCH_MOVE( endInfo ) );
            }
        }
    }

    // This indicates whether the section should be executed or not
    Section::operator bool() const {
        return m_sectionIncluded;
    }


} // end namespace Catch
