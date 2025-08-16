
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_ISTREAM_HPP_INCLUDED
#define CATCH_ISTREAM_HPP_INCLUDED

#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>

#include <iosfwd>
#include <string>

namespace Catch {

    class IStream {
    public:
        virtual ~IStream(); // = default
        virtual std::ostream& stream() = 0;
        /**
         * Best guess on whether the instance is writing to a console (e.g. via stdout/stderr)
         *
         * This is useful for e.g. Win32 colour support, because the Win32
         * API manipulates console directly, unlike POSIX escape codes,
         * that can be written anywhere.
         *
         * Due to variety of ways to change where the stdout/stderr is
         * _actually_ being written, users should always assume that
         * the answer might be wrong.
         */
        virtual bool isConsole() const { return false; }
    };

    /**
     * Creates a stream wrapper that writes to specific file.
     *
     * Also recognizes 4 special filenames
     * * `-` for stdout
     * * `%stdout` for stdout
     * * `%stderr` for stderr
     * * `%debug` for platform specific debugging output
     *
     * \throws if passed an unrecognized %-prefixed stream
     */
    auto makeStream( std::string const& filename ) -> Detail::unique_ptr<IStream>;

}

#endif // CATCH_STREAM_HPP_INCLUDED
