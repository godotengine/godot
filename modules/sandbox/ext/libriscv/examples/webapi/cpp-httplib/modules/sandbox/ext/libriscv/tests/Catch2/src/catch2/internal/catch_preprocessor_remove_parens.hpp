
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_PREPROCESSOR_REMOVE_PARENS_HPP_INCLUDED
#define CATCH_PREPROCESSOR_REMOVE_PARENS_HPP_INCLUDED

#define INTERNAL_CATCH_EXPAND1( param ) INTERNAL_CATCH_EXPAND2( param )
#define INTERNAL_CATCH_EXPAND2( ... ) INTERNAL_CATCH_NO##__VA_ARGS__
#define INTERNAL_CATCH_DEF( ... ) INTERNAL_CATCH_DEF __VA_ARGS__
#define INTERNAL_CATCH_NOINTERNAL_CATCH_DEF

#define INTERNAL_CATCH_REMOVE_PARENS( ... ) \
    INTERNAL_CATCH_EXPAND1( INTERNAL_CATCH_DEF __VA_ARGS__ )

#endif // CATCH_PREPROCESSOR_REMOVE_PARENS_HPP_INCLUDED
