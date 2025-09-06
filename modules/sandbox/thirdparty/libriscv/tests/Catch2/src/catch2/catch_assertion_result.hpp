
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_ASSERTION_RESULT_HPP_INCLUDED
#define CATCH_ASSERTION_RESULT_HPP_INCLUDED

#include <catch2/catch_assertion_info.hpp>
#include <catch2/internal/catch_result_type.hpp>
#include <catch2/internal/catch_source_line_info.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_lazy_expr.hpp>

#include <string>

namespace Catch {

    struct AssertionResultData
    {
        AssertionResultData() = delete;

        AssertionResultData( ResultWas::OfType _resultType, LazyExpression const& _lazyExpression );

        std::string message;
        mutable std::string reconstructedExpression;
        LazyExpression lazyExpression;
        ResultWas::OfType resultType;

        std::string reconstructExpression() const;
    };

    class AssertionResult {
    public:
        AssertionResult() = delete;
        AssertionResult( AssertionInfo const& info, AssertionResultData&& data );

        bool isOk() const;
        bool succeeded() const;
        ResultWas::OfType getResultType() const;
        bool hasExpression() const;
        bool hasMessage() const;
        std::string getExpression() const;
        std::string getExpressionInMacro() const;
        bool hasExpandedExpression() const;
        std::string getExpandedExpression() const;
        StringRef getMessage() const;
        SourceLineInfo getSourceInfo() const;
        StringRef getTestMacroName() const;

    //protected:
        AssertionInfo m_info;
        AssertionResultData m_resultData;
    };

} // end namespace Catch

#endif // CATCH_ASSERTION_RESULT_HPP_INCLUDED
