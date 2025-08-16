
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_MESSAGE_HPP_INCLUDED
#define CATCH_MESSAGE_HPP_INCLUDED

#include <catch2/internal/catch_config_prefix_messages.hpp>
#include <catch2/internal/catch_result_type.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_stream_end_stop.hpp>
#include <catch2/internal/catch_message_info.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>

#include <string>
#include <vector>

namespace Catch {

    struct SourceLineInfo;
    class IResultCapture;

    struct MessageStream {

        template<typename T>
        MessageStream& operator << ( T const& value ) {
            m_stream << value;
            return *this;
        }

        ReusableStringStream m_stream;
    };

    struct MessageBuilder : MessageStream {
        MessageBuilder( StringRef macroName,
                        SourceLineInfo const& lineInfo,
                        ResultWas::OfType type ):
            m_info(macroName, lineInfo, type) {}

        template<typename T>
        MessageBuilder&& operator << ( T const& value ) && {
            m_stream << value;
            return CATCH_MOVE(*this);
        }

        MessageInfo m_info;
    };

    class ScopedMessage {
    public:
        explicit ScopedMessage( MessageBuilder&& builder );
        ScopedMessage( ScopedMessage& duplicate ) = delete;
        ScopedMessage( ScopedMessage&& old ) noexcept;
        ~ScopedMessage();

        MessageInfo m_info;
        bool m_moved = false;
    };

    class Capturer {
        std::vector<MessageInfo> m_messages;
        IResultCapture& m_resultCapture;
        size_t m_captured = 0;
    public:
        Capturer( StringRef macroName, SourceLineInfo const& lineInfo, ResultWas::OfType resultType, StringRef names );

        Capturer(Capturer const&) = delete;
        Capturer& operator=(Capturer const&) = delete;

        ~Capturer();

        void captureValue( size_t index, std::string const& value );

        template<typename T>
        void captureValues( size_t index, T const& value ) {
            captureValue( index, Catch::Detail::stringify( value ) );
        }

        template<typename T, typename... Ts>
        void captureValues( size_t index, T const& value, Ts const&... values ) {
            captureValue( index, Catch::Detail::stringify(value) );
            captureValues( index+1, values... );
        }
    };

} // end namespace Catch

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_MSG( macroName, messageType, resultDisposition, ... ) \
    do { \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::StringRef(), resultDisposition ); \
        catchAssertionHandler.handleMessage( messageType, ( Catch::MessageStream() << __VA_ARGS__ + ::Catch::StreamEndStop() ).m_stream.str() ); \
        catchAssertionHandler.complete(); \
    } while( false )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_CAPTURE( varName, macroName, ... ) \
    Catch::Capturer varName( macroName##_catch_sr,        \
                             CATCH_INTERNAL_LINEINFO,     \
                             Catch::ResultWas::Info,      \
                             #__VA_ARGS__##_catch_sr );   \
    varName.captureValues( 0, __VA_ARGS__ )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_INFO( macroName, log ) \
    const Catch::ScopedMessage INTERNAL_CATCH_UNIQUE_NAME( scopedMessage )( Catch::MessageBuilder( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::ResultWas::Info ) << log )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_UNSCOPED_INFO( macroName, log ) \
    Catch::getResultCapture().emplaceUnscopedMessage( Catch::MessageBuilder( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::ResultWas::Info ) << log )


#if defined(CATCH_CONFIG_PREFIX_MESSAGES) && !defined(CATCH_CONFIG_DISABLE)

  #define CATCH_INFO( msg ) INTERNAL_CATCH_INFO( "CATCH_INFO", msg )
  #define CATCH_UNSCOPED_INFO( msg ) INTERNAL_CATCH_UNSCOPED_INFO( "CATCH_UNSCOPED_INFO", msg )
  #define CATCH_WARN( msg ) INTERNAL_CATCH_MSG( "CATCH_WARN", Catch::ResultWas::Warning, Catch::ResultDisposition::ContinueOnFailure, msg )
  #define CATCH_CAPTURE( ... ) INTERNAL_CATCH_CAPTURE( INTERNAL_CATCH_UNIQUE_NAME(capturer), "CATCH_CAPTURE", __VA_ARGS__ )

#elif defined(CATCH_CONFIG_PREFIX_MESSAGES) && defined(CATCH_CONFIG_DISABLE)

  #define CATCH_INFO( msg )          (void)(0)
  #define CATCH_UNSCOPED_INFO( msg ) (void)(0)
  #define CATCH_WARN( msg )          (void)(0)
  #define CATCH_CAPTURE( ... )       (void)(0)

#elif !defined(CATCH_CONFIG_PREFIX_MESSAGES) && !defined(CATCH_CONFIG_DISABLE)

  #define INFO( msg ) INTERNAL_CATCH_INFO( "INFO", msg )
  #define UNSCOPED_INFO( msg ) INTERNAL_CATCH_UNSCOPED_INFO( "UNSCOPED_INFO", msg )
  #define WARN( msg ) INTERNAL_CATCH_MSG( "WARN", Catch::ResultWas::Warning, Catch::ResultDisposition::ContinueOnFailure, msg )
  #define CAPTURE( ... ) INTERNAL_CATCH_CAPTURE( INTERNAL_CATCH_UNIQUE_NAME(capturer), "CAPTURE", __VA_ARGS__ )

#elif !defined(CATCH_CONFIG_PREFIX_MESSAGES) && defined(CATCH_CONFIG_DISABLE)

  #define INFO( msg )          (void)(0)
  #define UNSCOPED_INFO( msg ) (void)(0)
  #define WARN( msg )          (void)(0)
  #define CAPTURE( ... )       (void)(0)

#endif // end of user facing macro declarations




#endif // CATCH_MESSAGE_HPP_INCLUDED
