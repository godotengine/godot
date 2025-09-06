
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TRANSLATE_EXCEPTION_HPP_INCLUDED
#define CATCH_TRANSLATE_EXCEPTION_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_exception.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_unique_name.hpp>

#include <exception>

namespace Catch {
    namespace Detail {
        void registerTranslatorImpl(
            Detail::unique_ptr<IExceptionTranslator>&& translator );
    }

    class ExceptionTranslatorRegistrar {
        template<typename T>
        class ExceptionTranslator : public IExceptionTranslator {
        public:

            constexpr ExceptionTranslator( std::string(*translateFunction)( T const& ) )
            : m_translateFunction( translateFunction )
            {}

            std::string translate( ExceptionTranslators::const_iterator it, ExceptionTranslators::const_iterator itEnd ) const override {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
                try {
                    if( it == itEnd )
                        std::rethrow_exception(std::current_exception());
                    else
                        return (*it)->translate( it+1, itEnd );
                }
                catch( T const& ex ) {
                    return m_translateFunction( ex );
                }
#else
                return "You should never get here!";
#endif
            }

        protected:
            std::string(*m_translateFunction)( T const& );
        };

    public:
        template<typename T>
        ExceptionTranslatorRegistrar( std::string(*translateFunction)( T const& ) ) {
            Detail::registerTranslatorImpl(
                Detail::make_unique<ExceptionTranslator<T>>(
                    translateFunction ) );
        }
    };

} // namespace Catch

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_TRANSLATE_EXCEPTION2( translatorName, signature ) \
    static std::string translatorName( signature ); \
    CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
    CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
    namespace{ const Catch::ExceptionTranslatorRegistrar INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionRegistrar )( &translatorName ); } \
    CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
    static std::string translatorName( signature )

#define INTERNAL_CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION2( INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionTranslator ), signature )

#if defined(CATCH_CONFIG_DISABLE)
    #define INTERNAL_CATCH_TRANSLATE_EXCEPTION_NO_REG( translatorName, signature) \
            static std::string translatorName( signature )
#endif


// This macro is always prefixed
#if !defined(CATCH_CONFIG_DISABLE)
#define CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION( signature )
#else
#define CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION_NO_REG( INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionTranslator ), signature )
#endif


#endif // CATCH_TRANSLATE_EXCEPTION_HPP_INCLUDED
