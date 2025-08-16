
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_RESULT_TYPE_HPP_INCLUDED
#define CATCH_RESULT_TYPE_HPP_INCLUDED

namespace Catch {

    // ResultWas::OfType enum
    struct ResultWas { enum OfType {
        Unknown = -1,
        Ok = 0,
        Info = 1,
        Warning = 2,
        // TODO: Should explicit skip be considered "not OK" (cf. isOk)? I.e., should it have the failure bit?
        ExplicitSkip = 4,

        FailureBit = 0x10,

        ExpressionFailed = FailureBit | 1,
        ExplicitFailure = FailureBit | 2,

        Exception = 0x100 | FailureBit,

        ThrewException = Exception | 1,
        DidntThrowException = Exception | 2,

        FatalErrorCondition = 0x200 | FailureBit

    }; };

    constexpr bool isOk( ResultWas::OfType resultType ) {
        return ( resultType & ResultWas::FailureBit ) == 0;
    }
    constexpr bool isJustInfo( int flags ) { return flags == ResultWas::Info; }


    // ResultDisposition::Flags enum
    struct ResultDisposition { enum Flags {
        Normal = 0x01,

        ContinueOnFailure = 0x02,   // Failures fail test, but execution continues
        FalseTest = 0x04,           // Prefix expression with !
        SuppressFail = 0x08         // Failures are reported but do not fail the test
    }; };

    constexpr ResultDisposition::Flags operator|( ResultDisposition::Flags lhs,
                                        ResultDisposition::Flags rhs ) {
        return static_cast<ResultDisposition::Flags>( static_cast<int>( lhs ) |
                                                      static_cast<int>( rhs ) );
    }

    constexpr bool isFalseTest( int flags ) {
        return ( flags & ResultDisposition::FalseTest ) != 0;
    }
    constexpr bool shouldSuppressFailure( int flags ) {
        return ( flags & ResultDisposition::SuppressFail ) != 0;
    }

} // end namespace Catch

#endif // CATCH_RESULT_TYPE_HPP_INCLUDED
