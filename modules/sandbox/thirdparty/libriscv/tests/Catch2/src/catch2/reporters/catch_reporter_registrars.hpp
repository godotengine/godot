
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_REGISTRARS_HPP_INCLUDED
#define CATCH_REPORTER_REGISTRARS_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_unique_name.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_void_type.hpp>

#include <type_traits>

namespace Catch {

    namespace Detail {

        template <typename T, typename = void>
        struct has_description : std::false_type {};

        template <typename T>
        struct has_description<
            T,
            void_t<decltype( T::getDescription() )>>
            : std::true_type {};

        //! Indirection for reporter registration, so that the error handling is
        //! independent on the reporter's concrete type
        void registerReporterImpl( std::string const& name,
                                   IReporterFactoryPtr reporterPtr );
        //! Actually registers the factory, independent on listener's concrete type
        void registerListenerImpl( Detail::unique_ptr<EventListenerFactory> listenerFactory );
    } // namespace Detail

    class IEventListener;
    using IEventListenerPtr = Detail::unique_ptr<IEventListener>;

    template <typename T>
    class ReporterFactory : public IReporterFactory {

        IEventListenerPtr create( ReporterConfig&& config ) const override {
            return Detail::make_unique<T>( CATCH_MOVE(config) );
        }

        std::string getDescription() const override {
            return T::getDescription();
        }
    };


    template<typename T>
    class ReporterRegistrar {
    public:
        explicit ReporterRegistrar( std::string const& name ) {
            registerReporterImpl( name,
                                  Detail::make_unique<ReporterFactory<T>>() );
        }
    };

    template<typename T>
    class ListenerRegistrar {

        class TypedListenerFactory : public EventListenerFactory {
            StringRef m_listenerName;

            std::string getDescriptionImpl( std::true_type ) const {
                return T::getDescription();
            }

            std::string getDescriptionImpl( std::false_type ) const {
                return "(No description provided)";
            }

        public:
            TypedListenerFactory( StringRef listenerName ):
                m_listenerName( listenerName ) {}

            IEventListenerPtr create( IConfig const* config ) const override {
                return Detail::make_unique<T>( config );
            }

            StringRef getName() const override {
                return m_listenerName;
            }

            std::string getDescription() const override {
                return getDescriptionImpl( Detail::has_description<T>{} );
            }
        };

    public:
        ListenerRegistrar(StringRef listenerName) {
            registerListenerImpl( Detail::make_unique<TypedListenerFactory>(listenerName) );
        }
    };
}

#if !defined(CATCH_CONFIG_DISABLE)

#    define CATCH_REGISTER_REPORTER( name, reporterType )                  \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                          \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                           \
        namespace {                                                        \
            const Catch::ReporterRegistrar<reporterType>                   \
                INTERNAL_CATCH_UNIQUE_NAME( catch_internal_RegistrarFor )( \
                    name );                                                \
        }                                                                  \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#    define CATCH_REGISTER_LISTENER( listenerType )                        \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                          \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                           \
        namespace {                                                        \
            const Catch::ListenerRegistrar<listenerType>                   \
                INTERNAL_CATCH_UNIQUE_NAME( catch_internal_RegistrarFor )( \
                    #listenerType##_catch_sr );                            \
        }                                                                  \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#else // CATCH_CONFIG_DISABLE

#define CATCH_REGISTER_REPORTER(name, reporterType)
#define CATCH_REGISTER_LISTENER(listenerType)

#endif // CATCH_CONFIG_DISABLE

#endif // CATCH_REPORTER_REGISTRARS_HPP_INCLUDED
