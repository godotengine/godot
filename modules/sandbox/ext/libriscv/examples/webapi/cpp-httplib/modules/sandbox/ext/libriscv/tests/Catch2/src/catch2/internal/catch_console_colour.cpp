
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif


#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_errno_guard.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_istream.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_debugger.hpp>
#include <catch2/internal/catch_windows_h_proxy.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <cassert>
#include <ostream>
#include <utility>

namespace Catch {

    ColourImpl::~ColourImpl() = default;

    ColourImpl::ColourGuard ColourImpl::guardColour( Colour::Code colourCode ) {
        return ColourGuard(colourCode, this );
    }

    void ColourImpl::ColourGuard::engageImpl( std::ostream& stream ) {
        assert( &stream == &m_colourImpl->m_stream->stream() &&
                "Engaging colour guard for different stream than used by the "
                "parent colour implementation" );
        static_cast<void>( stream );

        m_engaged = true;
        m_colourImpl->use( m_code );
    }

    ColourImpl::ColourGuard::ColourGuard( Colour::Code code,
                                          ColourImpl const* colour ):
        m_colourImpl( colour ), m_code( code ) {
    }
    ColourImpl::ColourGuard::ColourGuard( ColourGuard&& rhs ) noexcept:
        m_colourImpl( rhs.m_colourImpl ),
        m_code( rhs.m_code ),
        m_engaged( rhs.m_engaged ) {
        rhs.m_engaged = false;
    }
    ColourImpl::ColourGuard&
    ColourImpl::ColourGuard::operator=( ColourGuard&& rhs ) noexcept {
        using std::swap;
        swap( m_colourImpl, rhs.m_colourImpl );
        swap( m_code, rhs.m_code );
        swap( m_engaged, rhs.m_engaged );

        return *this;
    }
    ColourImpl::ColourGuard::~ColourGuard() {
        if ( m_engaged ) {
            m_colourImpl->use( Colour::None );
        }
    }

    ColourImpl::ColourGuard&
    ColourImpl::ColourGuard::engage( std::ostream& stream ) & {
        engageImpl( stream );
        return *this;
    }

    ColourImpl::ColourGuard&&
    ColourImpl::ColourGuard::engage( std::ostream& stream ) && {
        engageImpl( stream );
        return CATCH_MOVE(*this);
    }

    namespace {
        //! A do-nothing implementation of colour, used as fallback for unknown
        //! platforms, and when the user asks to deactivate all colours.
        class NoColourImpl final : public ColourImpl {
        public:
            NoColourImpl( IStream* stream ): ColourImpl( stream ) {}

        private:
            void use( Colour::Code ) const override {}
        };
    } // namespace


} // namespace Catch


#if defined ( CATCH_CONFIG_COLOUR_WIN32 ) /////////////////////////////////////////

namespace Catch {
namespace {

    class Win32ColourImpl final : public ColourImpl {
    public:
        Win32ColourImpl(IStream* stream):
            ColourImpl(stream) {
            CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
            GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ),
                                        &csbiInfo );
            originalForegroundAttributes = csbiInfo.wAttributes & ~( BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_BLUE | BACKGROUND_INTENSITY );
            originalBackgroundAttributes = csbiInfo.wAttributes & ~( FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY );
        }

        static bool useImplementationForStream(IStream const& stream) {
            // Win32 text colour APIs can only be used on console streams
            // We cannot check that the output hasn't been redirected,
            // so we just check that the original stream is console stream.
            return stream.isConsole();
        }

    private:
        void use( Colour::Code _colourCode ) const override {
            switch( _colourCode ) {
                case Colour::None:      return setTextAttribute( originalForegroundAttributes );
                case Colour::White:     return setTextAttribute( FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE );
                case Colour::Red:       return setTextAttribute( FOREGROUND_RED );
                case Colour::Green:     return setTextAttribute( FOREGROUND_GREEN );
                case Colour::Blue:      return setTextAttribute( FOREGROUND_BLUE );
                case Colour::Cyan:      return setTextAttribute( FOREGROUND_BLUE | FOREGROUND_GREEN );
                case Colour::Yellow:    return setTextAttribute( FOREGROUND_RED | FOREGROUND_GREEN );
                case Colour::Grey:      return setTextAttribute( 0 );

                case Colour::LightGrey:     return setTextAttribute( FOREGROUND_INTENSITY );
                case Colour::BrightRed:     return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_RED );
                case Colour::BrightGreen:   return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_GREEN );
                case Colour::BrightWhite:   return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE );
                case Colour::BrightYellow:  return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN );

                case Colour::Bright: CATCH_INTERNAL_ERROR( "not a colour" );

                default:
                    CATCH_ERROR( "Unknown colour requested" );
            }
        }

        void setTextAttribute( WORD _textAttribute ) const {
            SetConsoleTextAttribute( GetStdHandle( STD_OUTPUT_HANDLE ),
                                     _textAttribute |
                                         originalBackgroundAttributes );
        }
        WORD originalForegroundAttributes;
        WORD originalBackgroundAttributes;
    };

} // end anon namespace
} // end namespace Catch

#endif // Windows/ ANSI/ None


#if defined( CATCH_PLATFORM_LINUX ) || defined( CATCH_PLATFORM_MAC ) || defined( __GLIBC__ ) || defined(__FreeBSD__)
#    define CATCH_INTERNAL_HAS_ISATTY
#    include <unistd.h>
#endif

namespace Catch {
namespace {

    class ANSIColourImpl final : public ColourImpl {
    public:
        ANSIColourImpl( IStream* stream ): ColourImpl( stream ) {}

        static bool useImplementationForStream(IStream const& stream) {
            // This is kinda messy due to trying to support a bunch of
            // different platforms at once.
            // The basic idea is that if we are asked to do autodetection (as
            // opposed to being told to use posixy colours outright), then we
            // only want to use the colours if we are writing to console.
            // However, console might be redirected, so we make an attempt at
            // checking for that on platforms where we know how to do that.
            bool useColour = stream.isConsole();
#if defined( CATCH_INTERNAL_HAS_ISATTY ) && \
    !( defined( __DJGPP__ ) && defined( __STRICT_ANSI__ ) )
            ErrnoGuard _; // for isatty
            useColour = useColour && isatty( STDOUT_FILENO );
#    endif
#    if defined( CATCH_PLATFORM_MAC ) || defined( CATCH_PLATFORM_IPHONE )
            useColour = useColour && !isDebuggerActive();
#    endif

            return useColour;
        }

    private:
        void use( Colour::Code _colourCode ) const override {
            auto setColour = [&out =
                                  m_stream->stream()]( char const* escapeCode ) {
                // The escape sequence must be flushed to console, otherwise
                // if stdin and stderr are intermixed, we'd get accidentally
                // coloured output.
                out << '\033' << escapeCode << std::flush;
            };
            switch( _colourCode ) {
                case Colour::None:
                case Colour::White:     return setColour( "[0m" );
                case Colour::Red:       return setColour( "[0;31m" );
                case Colour::Green:     return setColour( "[0;32m" );
                case Colour::Blue:      return setColour( "[0;34m" );
                case Colour::Cyan:      return setColour( "[0;36m" );
                case Colour::Yellow:    return setColour( "[0;33m" );
                case Colour::Grey:      return setColour( "[1;30m" );

                case Colour::LightGrey:     return setColour( "[0;37m" );
                case Colour::BrightRed:     return setColour( "[1;31m" );
                case Colour::BrightGreen:   return setColour( "[1;32m" );
                case Colour::BrightWhite:   return setColour( "[1;37m" );
                case Colour::BrightYellow:  return setColour( "[1;33m" );

                case Colour::Bright: CATCH_INTERNAL_ERROR( "not a colour" );
                default: CATCH_INTERNAL_ERROR( "Unknown colour requested" );
            }
        }
    };

} // end anon namespace
} // end namespace Catch

namespace Catch {

    Detail::unique_ptr<ColourImpl> makeColourImpl( ColourMode colourSelection,
                                                   IStream* stream ) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
        if ( colourSelection == ColourMode::Win32 ) {
            return Detail::make_unique<Win32ColourImpl>( stream );
        }
#endif
        if ( colourSelection == ColourMode::ANSI ) {
            return Detail::make_unique<ANSIColourImpl>( stream );
        }
        if ( colourSelection == ColourMode::None ) {
            return Detail::make_unique<NoColourImpl>( stream );
        }

        if ( colourSelection == ColourMode::PlatformDefault) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
            if ( Win32ColourImpl::useImplementationForStream( *stream ) ) {
                return Detail::make_unique<Win32ColourImpl>( stream );
            }
#endif
            if ( ANSIColourImpl::useImplementationForStream( *stream ) ) {
                return Detail::make_unique<ANSIColourImpl>( stream );
            }
            return Detail::make_unique<NoColourImpl>( stream );
        }

        CATCH_ERROR( "Could not create colour impl for selection " << static_cast<int>(colourSelection) );
    }

    bool isColourImplAvailable( ColourMode colourSelection ) {
        switch ( colourSelection ) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
        case ColourMode::Win32:
#endif
        case ColourMode::ANSI:
        case ColourMode::None:
        case ColourMode::PlatformDefault:
            return true;
        default:
            return false;
        }
    }


} // end namespace Catch

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

