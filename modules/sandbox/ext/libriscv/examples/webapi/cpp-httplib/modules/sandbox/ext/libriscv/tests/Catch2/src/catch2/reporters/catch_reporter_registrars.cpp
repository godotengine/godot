
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

namespace Catch {
    namespace Detail {

        void registerReporterImpl( std::string const& name,
                                   IReporterFactoryPtr reporterPtr ) {
            CATCH_TRY {
                getMutableRegistryHub().registerReporter(
                    name, CATCH_MOVE( reporterPtr ) );
            }
            CATCH_CATCH_ALL {
                // Do not throw when constructing global objects, instead
                // register the exception to be processed later
                getMutableRegistryHub().registerStartupException();
            }
        }

        void registerListenerImpl( Detail::unique_ptr<EventListenerFactory> listenerFactory ) {
            getMutableRegistryHub().registerListener( CATCH_MOVE(listenerFactory) );
        }


    } // namespace Detail
} // namespace Catch
