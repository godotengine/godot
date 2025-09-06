
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_assertion_result.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {

    AssertionResultData::AssertionResultData(ResultWas::OfType _resultType, LazyExpression const& _lazyExpression):
        lazyExpression(_lazyExpression),
        resultType(_resultType) {}

    std::string AssertionResultData::reconstructExpression() const {

        if( reconstructedExpression.empty() ) {
            if( lazyExpression ) {
                ReusableStringStream rss;
                rss << lazyExpression;
                reconstructedExpression = rss.str();
            }
        }
        return reconstructedExpression;
    }

    AssertionResult::AssertionResult( AssertionInfo const& info, AssertionResultData&& data )
    :   m_info( info ),
        m_resultData( CATCH_MOVE(data) )
    {}

    // Result was a success
    bool AssertionResult::succeeded() const {
        return Catch::isOk( m_resultData.resultType );
    }

    // Result was a success, or failure is suppressed
    bool AssertionResult::isOk() const {
        return Catch::isOk( m_resultData.resultType ) || shouldSuppressFailure( m_info.resultDisposition );
    }

    ResultWas::OfType AssertionResult::getResultType() const {
        return m_resultData.resultType;
    }

    bool AssertionResult::hasExpression() const {
        return !m_info.capturedExpression.empty();
    }

    bool AssertionResult::hasMessage() const {
        return !m_resultData.message.empty();
    }

    std::string AssertionResult::getExpression() const {
        // Possibly overallocating by 3 characters should be basically free
        std::string expr; expr.reserve(m_info.capturedExpression.size() + 3);
        if (isFalseTest(m_info.resultDisposition)) {
            expr += "!(";
        }
        expr += m_info.capturedExpression;
        if (isFalseTest(m_info.resultDisposition)) {
            expr += ')';
        }
        return expr;
    }

    std::string AssertionResult::getExpressionInMacro() const {
        if ( m_info.macroName.empty() ) {
            return static_cast<std::string>( m_info.capturedExpression );
        }
        std::string expr;
        expr.reserve( m_info.macroName.size() + m_info.capturedExpression.size() + 4 );
        expr += m_info.macroName;
        expr += "( ";
        expr += m_info.capturedExpression;
        expr += " )";
        return expr;
    }

    bool AssertionResult::hasExpandedExpression() const {
        return hasExpression() && getExpandedExpression() != getExpression();
    }

    std::string AssertionResult::getExpandedExpression() const {
        std::string expr = m_resultData.reconstructExpression();
        return expr.empty()
                ? getExpression()
                : expr;
    }

    StringRef AssertionResult::getMessage() const {
        return m_resultData.message;
    }
    SourceLineInfo AssertionResult::getSourceInfo() const {
        return m_info.lineInfo;
    }

    StringRef AssertionResult::getTestMacroName() const {
        return m_info.macroName;
    }

} // end namespace Catch
