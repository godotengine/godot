
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_DECOMPOSER_HPP_INCLUDED
#define CATCH_DECOMPOSER_HPP_INCLUDED

#include <catch2/catch_tostring.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_compare_traits.hpp>
#include <catch2/internal/catch_test_failure_exception.hpp>
#include <catch2/internal/catch_logical_traits.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <type_traits>
#include <iosfwd>

/** \file
 * Why does decomposing look the way it does:
 *
 * Conceptually, decomposing is simple. We change `REQUIRE( a == b )` into
 * `Decomposer{} <= a == b`, so that `Decomposer{} <= a` is evaluated first,
 * and our custom operator is used for `a == b`, because `a` is transformed
 * into `ExprLhs<T&>` and then into `BinaryExpr<T&, U&>`.
 *
 * In practice, decomposing ends up a mess, because we have to support
 * various fun things.
 *
 * 1) Types that are only comparable with literal 0, and they do this by
 *    comparing against a magic type with pointer constructor and deleted
 *    other constructors. Example: `REQUIRE((a <=> b) == 0)` in libstdc++
 *
 * 2) Types that are only comparable with literal 0, and they do this by
 *    comparing against a magic type with consteval integer constructor.
 *    Example: `REQUIRE((a <=> b) == 0)` in current MSVC STL.
 *
 * 3) Types that have no linkage, and so we cannot form a reference to
 *    them. Example: some implementations of traits.
 *
 * 4) Starting with C++20, when the compiler sees `a == b`, it also uses
 *    `b == a` when constructing the overload set. For us this means that
 *    when the compiler handles `ExprLhs<T> == b`, it also tries to resolve
 *    the overload set for `b == ExprLhs<T>`.
 *
 * To accommodate these use cases, decomposer ended up rather complex.
 *
 * 1) These types are handled by adding SFINAE overloads to our comparison
 *    operators, checking whether `T == U` are comparable with the given
 *    operator, and if not, whether T (or U) are comparable with literal 0.
 *    If yes, the overload compares T (or U) with 0 literal inline in the
 *    definition.
 *
 *    Note that for extra correctness, we check  that the other type is
 *    either an `int` (literal 0 is captured as `int` by templates), or
 *    a `long` (some platforms use 0L for `NULL` and we want to support
 *    that for pointer comparisons).
 *
 * 2) For these types, `is_foo_comparable<T, int>` is true, but letting
 *    them fall into the overload that actually does `T == int` causes
 *    compilation error. Handling them requires that the decomposition
 *    is `constexpr`, so that P2564R3 applies and the `consteval` from
 *    their accompanying magic type is propagated through the `constexpr`
 *    call stack.
 *
 *    However this is not enough to handle these types automatically,
 *    because our default is to capture types by reference, to avoid
 *    runtime copies. While these references cannot become dangling,
 *    they outlive the constexpr context and thus the default capture
 *    path cannot be actually constexpr.
 *
 *    The solution is to capture these types by value, by explicitly
 *    specializing `Catch::capture_by_value` for them. Catch2 provides
 *    specialization for `std::foo_ordering`s, but users can specialize
 *    the trait for their own types as well.
 *
 * 3) If a type has no linkage, we also cannot capture it by reference.
 *    The solution is once again to capture them by value. We handle
 *    the common cases by using `std::is_arithmetic` and `std::is_enum`
 *    as the default for `Catch::capture_by_value`, but that is only a
 *    some-effort heuristic. These combine to capture all possible bitfield
 *    bases, and also some trait-like types. As with 2), users can
 *    specialize `capture_by_value` for their own types as needed.
 *
 * 4) To support C++20 and make the SFINAE on our decomposing operators
 *    work, the SFINAE has to happen in return type, rather than in
 *    a template type. This is due to our use of logical type traits
 *    (`conjunction`/`disjunction`/`negation`), that we use to workaround
 *    an issue in older (9-) versions of GCC. I still blame C++20 for
 *    this, because without the comparison order switching, the logical
 *    traits could still be used in template type.
 *
 * There are also other side concerns, e.g. supporting both `REQUIRE(a)`
 * and `REQUIRE(a == b)`, or making `REQUIRE_THAT(a, IsEqual(b))` slot
 * nicely into the same expression handling logic, but these are rather
 * straightforward and add only a bit of complexity (e.g. common base
 * class for decomposed expressions).
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4389) // '==' : signed/unsigned mismatch
#pragma warning(disable:4018) // more "signed/unsigned mismatch"
#pragma warning(disable:4312) // Converting int to T* using reinterpret_cast (issue on x64 platform)
#pragma warning(disable:4180) // qualifier applied to function type has no meaning
#pragma warning(disable:4800) // Forcing result to true or false
#endif

#ifdef __clang__
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wsign-compare"
#  pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#elif defined __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

#if defined(CATCH_CPP20_OR_GREATER) && __has_include(<compare>)
#  include <compare>
#    if defined( __cpp_lib_three_way_comparison ) && \
            __cpp_lib_three_way_comparison >= 201907L
#      define CATCH_CONFIG_CPP20_COMPARE_OVERLOADS
#    endif
#endif

namespace Catch {

    namespace Detail {
        // This was added in C++20, but we require only C++14 for now.
        template <typename T>
        using RemoveCVRef_t = std::remove_cv_t<std::remove_reference_t<T>>;
    }

    // Note: This is about as much as we can currently reasonably support.
    //       In an ideal world, we could capture by value small trivially
    //       copyable types, but the actual `std::is_trivially_copyable`
    //       trait is a huge mess with standard-violating results on
    //       GCC and Clang, which are unlikely to be fixed soon due to ABI
    //       concerns.
    //       `std::is_scalar` also causes issues due to the `is_pointer`
    //       component, which causes ambiguity issues with (references-to)
    //       function pointer. If those are resolved, we still need to
    //       disambiguate the overload set for arrays, through explicit
    //       overload for references to sized arrays.
    template <typename T>
    struct capture_by_value
        : std::integral_constant<bool,
                                 std::is_arithmetic<T>::value ||
                                     std::is_enum<T>::value> {};

#if defined( CATCH_CONFIG_CPP20_COMPARE_OVERLOADS )
    template <>
    struct capture_by_value<std::strong_ordering> : std::true_type {};
    template <>
    struct capture_by_value<std::weak_ordering> : std::true_type {};
    template <>
    struct capture_by_value<std::partial_ordering> : std::true_type {};
#endif

    template <typename T>
    struct always_false : std::false_type {};

    class ITransientExpression {
        bool m_isBinaryExpression;
        bool m_result;

    protected:
        ~ITransientExpression() = default;

    public:
        constexpr auto isBinaryExpression() const -> bool { return m_isBinaryExpression; }
        constexpr auto getResult() const -> bool { return m_result; }
        //! This function **has** to be overridden by the derived class.
        virtual void streamReconstructedExpression( std::ostream& os ) const;

        constexpr ITransientExpression( bool isBinaryExpression, bool result )
        :   m_isBinaryExpression( isBinaryExpression ),
            m_result( result )
        {}

        constexpr ITransientExpression( ITransientExpression const& ) = default;
        constexpr ITransientExpression& operator=( ITransientExpression const& ) = default;

        friend std::ostream& operator<<(std::ostream& out, ITransientExpression const& expr) {
            expr.streamReconstructedExpression(out);
            return out;
        }
    };

    void formatReconstructedExpression( std::ostream &os, std::string const& lhs, StringRef op, std::string const& rhs );

    template<typename LhsT, typename RhsT>
    class BinaryExpr  : public ITransientExpression {
        LhsT m_lhs;
        StringRef m_op;
        RhsT m_rhs;

        void streamReconstructedExpression( std::ostream &os ) const override {
            formatReconstructedExpression
                    ( os, Catch::Detail::stringify( m_lhs ), m_op, Catch::Detail::stringify( m_rhs ) );
        }

    public:
        constexpr BinaryExpr( bool comparisonResult, LhsT lhs, StringRef op, RhsT rhs )
        :   ITransientExpression{ true, comparisonResult },
            m_lhs( lhs ),
            m_op( op ),
            m_rhs( rhs )
        {}

        template<typename T>
        auto operator && ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator || ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator == ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator != ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator > ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator < ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator >= ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename T>
        auto operator <= ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
            static_assert(always_false<T>::value,
            "chained comparisons are not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }
    };

    template<typename LhsT>
    class UnaryExpr : public ITransientExpression {
        LhsT m_lhs;

        void streamReconstructedExpression( std::ostream &os ) const override {
            os << Catch::Detail::stringify( m_lhs );
        }

    public:
        explicit constexpr UnaryExpr( LhsT lhs )
        :   ITransientExpression{ false, static_cast<bool>(lhs) },
            m_lhs( lhs )
        {}
    };


    template<typename LhsT>
    class ExprLhs {
        LhsT m_lhs;
    public:
        explicit constexpr ExprLhs( LhsT lhs ) : m_lhs( lhs ) {}

#define CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( id, op )           \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )             \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
                                Detail::negation<capture_by_value<             \
                                    Detail::RemoveCVRef_t<RhsT>>>>::value,     \
            BinaryExpr<LhsT, RhsT const&>> {                                   \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
                                capture_by_value<RhsT>>::value,                \
            BinaryExpr<LhsT, RhsT>> {                                          \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<                                               \
                Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
                Detail::is_eq_0_comparable<LhsT>,                              \
              /* We allow long because we want `ptr op NULL` to be accepted */ \
                Detail::disjunction<std::is_same<RhsT, int>,                   \
                                    std::is_same<RhsT, long>>>::value,         \
            BinaryExpr<LhsT, RhsT>> {                                          \
        if ( rhs != 0 ) { throw_test_failure_exception(); }                    \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op 0 ), lhs.m_lhs, #op##_sr, rhs };   \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<                                               \
                Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
                Detail::is_eq_0_comparable<RhsT>,                              \
              /* We allow long because we want `ptr op NULL` to be accepted */ \
                Detail::disjunction<std::is_same<LhsT, int>,                   \
                                    std::is_same<LhsT, long>>>::value,         \
            BinaryExpr<LhsT, RhsT>> {                                          \
        if ( lhs.m_lhs != 0 ) { throw_test_failure_exception(); }              \
        return { static_cast<bool>( 0 op rhs ), lhs.m_lhs, #op##_sr, rhs };    \
    }

        CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( eq, == )
        CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( ne, != )

    #undef CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR


#define CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( id, op )         \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )             \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
                                Detail::negation<capture_by_value<             \
                                    Detail::RemoveCVRef_t<RhsT>>>>::value,     \
            BinaryExpr<LhsT, RhsT const&>> {                                   \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
                                capture_by_value<RhsT>>::value,                \
            BinaryExpr<LhsT, RhsT>> {                                          \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<                                               \
                Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
                Detail::is_##id##_0_comparable<LhsT>,                          \
                std::is_same<RhsT, int>>::value,                               \
            BinaryExpr<LhsT, RhsT>> {                                          \
        if ( rhs != 0 ) { throw_test_failure_exception(); }                    \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op 0 ), lhs.m_lhs, #op##_sr, rhs };   \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<                                                   \
            Detail::conjunction<                                               \
                Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
                Detail::is_##id##_0_comparable<RhsT>,                          \
                std::is_same<LhsT, int>>::value,                               \
            BinaryExpr<LhsT, RhsT>> {                                          \
        if ( lhs.m_lhs != 0 ) { throw_test_failure_exception(); }              \
        return { static_cast<bool>( 0 op rhs ), lhs.m_lhs, #op##_sr, rhs };    \
    }

        CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( lt, < )
        CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( le, <= )
        CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( gt, > )
        CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( ge, >= )

    #undef CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR


#define CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR( op )                        \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )             \
        -> std::enable_if_t<                                                   \
            !capture_by_value<Detail::RemoveCVRef_t<RhsT>>::value,             \
            BinaryExpr<LhsT, RhsT const&>> {                                   \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }                                                                          \
    template <typename RhsT>                                                   \
    constexpr friend auto operator op( ExprLhs&& lhs, RhsT rhs )               \
        -> std::enable_if_t<capture_by_value<RhsT>::value,                     \
                            BinaryExpr<LhsT, RhsT>> {                          \
        return {                                                               \
            static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
    }

        CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(|)
        CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(&)
        CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(^)

    #undef CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR

        template<typename RhsT>
        friend auto operator && ( ExprLhs &&, RhsT && ) -> BinaryExpr<LhsT, RhsT const&> {
            static_assert(always_false<RhsT>::value,
            "operator&& is not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        template<typename RhsT>
        friend auto operator || ( ExprLhs &&, RhsT && ) -> BinaryExpr<LhsT, RhsT const&> {
            static_assert(always_false<RhsT>::value,
            "operator|| is not supported inside assertions, "
            "wrap the expression inside parentheses, or decompose it");
        }

        constexpr auto makeUnaryExpr() const -> UnaryExpr<LhsT> {
            return UnaryExpr<LhsT>{ m_lhs };
        }
    };

    struct Decomposer {
        template <typename T,
                  std::enable_if_t<!capture_by_value<Detail::RemoveCVRef_t<T>>::value,
                      int> = 0>
        constexpr friend auto operator <= ( Decomposer &&, T && lhs ) -> ExprLhs<T const&> {
            return ExprLhs<const T&>{ lhs };
        }

        template <typename T,
                  std::enable_if_t<capture_by_value<T>::value, int> = 0>
        constexpr friend auto operator <= ( Decomposer &&, T value ) -> ExprLhs<T> {
            return ExprLhs<T>{ value };
        }
    };

} // end namespace Catch

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __clang__
#  pragma clang diagnostic pop
#elif defined __GNUC__
#  pragma GCC diagnostic pop
#endif

#endif // CATCH_DECOMPOSER_HPP_INCLUDED
