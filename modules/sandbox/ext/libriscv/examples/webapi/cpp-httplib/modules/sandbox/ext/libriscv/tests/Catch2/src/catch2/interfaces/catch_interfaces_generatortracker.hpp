
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_GENERATORTRACKER_HPP_INCLUDED
#define CATCH_INTERFACES_GENERATORTRACKER_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <string>

namespace Catch {

    namespace Generators {
        class GeneratorUntypedBase {
            // Caches result from `toStringImpl`, assume that when it is an
            // empty string, the cache is invalidated.
            mutable std::string m_stringReprCache;

            // Counts based on `next` returning true
            std::size_t m_currentElementIndex = 0;

            /**
             * Attempts to move the generator to the next element
             *
             * Returns true iff the move succeeded (and a valid element
             * can be retrieved).
             */
            virtual bool next() = 0;

            //! Customization point for `currentElementAsString`
            virtual std::string stringifyImpl() const = 0;

        public:
            GeneratorUntypedBase() = default;
            // Generation of copy ops is deprecated (and Clang will complain)
            // if there is a user destructor defined
            GeneratorUntypedBase(GeneratorUntypedBase const&) = default;
            GeneratorUntypedBase& operator=(GeneratorUntypedBase const&) = default;

            virtual ~GeneratorUntypedBase(); // = default;

            /**
             * Attempts to move the generator to the next element
             *
             * Serves as a non-virtual interface to `next`, so that the
             * top level interface can provide sanity checking and shared
             * features.
             *
             * As with `next`, returns true iff the move succeeded and
             * the generator has new valid element to provide.
             */
            bool countedNext();

            std::size_t currentElementIndex() const { return m_currentElementIndex; }

            /**
             * Returns generator's current element as user-friendly string.
             *
             * By default returns string equivalent to calling
             * `Catch::Detail::stringify` on the current element, but generators
             * can customize their implementation as needed.
             *
             * Not thread-safe due to internal caching.
             *
             * The returned ref is valid only until the generator instance
             * is destructed, or it moves onto the next element, whichever
             * comes first.
             */
            StringRef currentElementAsString() const;
        };
        using GeneratorBasePtr = Catch::Detail::unique_ptr<GeneratorUntypedBase>;

    } // namespace Generators

    class IGeneratorTracker {
    public:
        virtual ~IGeneratorTracker(); // = default;
        virtual auto hasGenerator() const -> bool = 0;
        virtual auto getGenerator() const -> Generators::GeneratorBasePtr const& = 0;
        virtual void setGenerator( Generators::GeneratorBasePtr&& generator ) = 0;
    };

} // namespace Catch

#endif // CATCH_INTERFACES_GENERATORTRACKER_HPP_INCLUDED
