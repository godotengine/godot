
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_REGISTRY_HUB_HPP_INCLUDED
#define CATCH_INTERFACES_REGISTRY_HUB_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>

#include <string>

namespace Catch {

    class TestCaseHandle;
    struct TestCaseInfo;
    class ITestCaseRegistry;
    class IExceptionTranslatorRegistry;
    class IExceptionTranslator;
    class ReporterRegistry;
    class IReporterFactory;
    class ITagAliasRegistry;
    class ITestInvoker;
    class IMutableEnumValuesRegistry;
    struct SourceLineInfo;

    class StartupExceptionRegistry;
    class EventListenerFactory;

    using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;

    class IRegistryHub {
    public:
        virtual ~IRegistryHub(); // = default

        virtual ReporterRegistry const& getReporterRegistry() const = 0;
        virtual ITestCaseRegistry const& getTestCaseRegistry() const = 0;
        virtual ITagAliasRegistry const& getTagAliasRegistry() const = 0;
        virtual IExceptionTranslatorRegistry const& getExceptionTranslatorRegistry() const = 0;


        virtual StartupExceptionRegistry const& getStartupExceptionRegistry() const = 0;
    };

    class IMutableRegistryHub {
    public:
        virtual ~IMutableRegistryHub(); // = default
        virtual void registerReporter( std::string const& name, IReporterFactoryPtr factory ) = 0;
        virtual void registerListener( Detail::unique_ptr<EventListenerFactory> factory ) = 0;
        virtual void registerTest(Detail::unique_ptr<TestCaseInfo>&& testInfo, Detail::unique_ptr<ITestInvoker>&& invoker) = 0;
        virtual void registerTranslator( Detail::unique_ptr<IExceptionTranslator>&& translator ) = 0;
        virtual void registerTagAlias( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo ) = 0;
        virtual void registerStartupException() noexcept = 0;
        virtual IMutableEnumValuesRegistry& getMutableEnumValuesRegistry() = 0;
    };

    IRegistryHub const& getRegistryHub();
    IMutableRegistryHub& getMutableRegistryHub();
    void cleanUp();
    std::string translateActiveException();

}

#endif // CATCH_INTERFACES_REGISTRY_HUB_HPP_INCLUDED
