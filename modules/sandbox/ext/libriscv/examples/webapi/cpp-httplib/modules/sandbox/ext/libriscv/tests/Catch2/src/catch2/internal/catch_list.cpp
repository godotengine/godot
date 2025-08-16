
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_list.hpp>

#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/interfaces/catch_interfaces_reporter.hpp>
#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>
#include <catch2/internal/catch_test_case_registry_impl.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/catch_config.hpp>
#include <catch2/catch_test_case_info.hpp>

namespace Catch {
    namespace {

        void listTests(IEventListener& reporter, IConfig const& config) {
            auto const& testSpec = config.testSpec();
            auto matchedTestCases = filterTests(getAllTestCasesSorted(config), testSpec, config);
            reporter.listTests(matchedTestCases);
        }

        void listTags(IEventListener& reporter, IConfig const& config) {
            auto const& testSpec = config.testSpec();
            std::vector<TestCaseHandle> matchedTestCases = filterTests(getAllTestCasesSorted(config), testSpec, config);

            std::map<StringRef, TagInfo, Detail::CaseInsensitiveLess> tagCounts;
            for (auto const& testCase : matchedTestCases) {
                for (auto const& tagName : testCase.getTestCaseInfo().tags) {
                    auto it = tagCounts.find(tagName.original);
                    if (it == tagCounts.end())
                        it = tagCounts.insert(std::make_pair(tagName.original, TagInfo())).first;
                    it->second.add(tagName.original);
                }
            }

            std::vector<TagInfo> infos; infos.reserve(tagCounts.size());
            for (auto& tagc : tagCounts) {
                infos.push_back(CATCH_MOVE(tagc.second));
            }

            reporter.listTags(infos);
        }

        void listReporters(IEventListener& reporter) {
            std::vector<ReporterDescription> descriptions;

            auto const& factories = getRegistryHub().getReporterRegistry().getFactories();
            descriptions.reserve(factories.size());
            for (auto const& fac : factories) {
                descriptions.push_back({ fac.first, fac.second->getDescription() });
            }

            reporter.listReporters(descriptions);
        }

        void listListeners(IEventListener& reporter) {
            std::vector<ListenerDescription> descriptions;

            auto const& factories =
                getRegistryHub().getReporterRegistry().getListeners();
            descriptions.reserve( factories.size() );
            for ( auto const& fac : factories ) {
                descriptions.push_back( { fac->getName(), fac->getDescription() } );
            }

            reporter.listListeners( descriptions );
        }

    } // end anonymous namespace

    void TagInfo::add( StringRef spelling ) {
        ++count;
        spellings.insert( spelling );
    }

    std::string TagInfo::all() const {
        // 2 per tag for brackets '[' and ']'
        size_t size =  spellings.size() * 2;
        for (auto const& spelling : spellings) {
            size += spelling.size();
        }

        std::string out; out.reserve(size);
        for (auto const& spelling : spellings) {
            out += '[';
            out += spelling;
            out += ']';
        }
        return out;
    }

    bool list( IEventListener& reporter, Config const& config ) {
        bool listed = false;
        if (config.listTests()) {
            listed = true;
            listTests(reporter, config);
        }
        if (config.listTags()) {
            listed = true;
            listTags(reporter, config);
        }
        if (config.listReporters()) {
            listed = true;
            listReporters(reporter);
        }
        if ( config.listListeners() ) {
            listed = true;
            listListeners( reporter );
        }
        return listed;
    }

} // end namespace Catch
