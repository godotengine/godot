
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>

#include <tuple>

namespace {
    struct unique_ptr_test_helper {
        bool dummy = false;
    };
} // end unnamed namespace

TEST_CASE("unique_ptr reimplementation: basic functionality", "[internals][unique-ptr]") {
    using Catch::Detail::unique_ptr;
    SECTION("Default constructed unique_ptr is empty") {
        unique_ptr<int> ptr;
        REQUIRE_FALSE(ptr);
        REQUIRE(ptr.get() == nullptr);
    }
    SECTION("Take ownership of allocation") {
        auto naked_ptr = new int{ 0 };
        unique_ptr<int> ptr(naked_ptr);
        REQUIRE(ptr);
        REQUIRE(*ptr == 0);
        REQUIRE(ptr.get() == naked_ptr);
        SECTION("Plain reset deallocates") {
            ptr.reset(); // this makes naked_ptr dangling!
            REQUIRE_FALSE(ptr);
            REQUIRE(ptr.get() == nullptr);
        }
        SECTION("Reset replaces ownership") {
            ptr.reset(new int{ 2 });
            REQUIRE(ptr);
            REQUIRE(ptr.get() != nullptr);
            REQUIRE(*ptr == 2);
        }
    }
    SECTION("Release releases ownership") {
        auto naked_ptr = new int{ 1 };
        unique_ptr<int> ptr(naked_ptr);
        ptr.release();
        CHECK_FALSE(ptr);
        CHECK(ptr.get() == nullptr);
        delete naked_ptr;
    }
    SECTION("Move constructor") {
        unique_ptr<int> ptr1(new int{ 1 });
        auto ptr2(std::move(ptr1));
        REQUIRE_FALSE(ptr1);
        REQUIRE(ptr2);
        REQUIRE(*ptr2 == 1);
    }
    SECTION("Move assignment") {
        unique_ptr<int> ptr1(new int{ 1 }), ptr2(new int{ 2 });
        ptr1 = std::move(ptr2);
        REQUIRE_FALSE(ptr2);
        REQUIRE(ptr1);
        REQUIRE(*ptr1 == 2);
    }
    SECTION("free swap") {
        unique_ptr<int> ptr1(new int{ 1 }), ptr2(new int{ 2 });
        swap(ptr1, ptr2);
        REQUIRE(*ptr1 == 2);
        REQUIRE(*ptr2 == 1);
    }
}


namespace {
    struct base {
        int i;
        base(int i_) :i(i_) {}
    };
    struct derived : base { using base::base; };
    struct unrelated {};

} // end unnamed namespace

static_assert( std::is_constructible<Catch::Detail::unique_ptr<base>,
                                     Catch::Detail::unique_ptr<derived>>::value, "Upcasting is supported");
static_assert(!std::is_constructible<Catch::Detail::unique_ptr<derived>,
                                     Catch::Detail::unique_ptr<base>>::value, "Downcasting is not supported");
static_assert(!std::is_constructible<Catch::Detail::unique_ptr<base>,
                                     Catch::Detail::unique_ptr<unrelated>>::value, "Cannot just convert one ptr type to another");

TEST_CASE("Upcasting special member functions", "[internals][unique-ptr]") {
    using Catch::Detail::unique_ptr;

    unique_ptr<derived> dptr(new derived{3});
    SECTION("Move constructor") {
        unique_ptr<base> bptr(std::move(dptr));
        REQUIRE(bptr->i == 3);
    }
    SECTION("move assignment") {
        unique_ptr<base> bptr(new base{ 1 });
        bptr = std::move(dptr);
        REQUIRE(bptr->i == 3);
    }
}

namespace {
    struct move_detector {
        bool has_moved = false;
        move_detector() = default;
        move_detector(move_detector const& rhs) = default;
        move_detector& operator=(move_detector const& rhs) = default;

        move_detector(move_detector&& rhs) noexcept {
            rhs.has_moved = true;
        }
        move_detector& operator=(move_detector&& rhs) noexcept {
            rhs.has_moved = true;
            return *this;
        }
    };
} // end unnamed namespace

TEST_CASE("make_unique reimplementation", "[internals][unique-ptr]") {
    using Catch::Detail::make_unique;
    SECTION("From lvalue copies") {
        move_detector lval;
        auto ptr = make_unique<move_detector>(lval);
        REQUIRE_FALSE(lval.has_moved);
    }
    SECTION("From rvalue moves") {
        move_detector rval;
        auto ptr = make_unique<move_detector>(std::move(rval));
        REQUIRE(rval.has_moved);
    }
    SECTION("Variadic constructor") {
        auto ptr = make_unique<std::tuple<int, double, int>>(1, 2., 3);
        REQUIRE(*ptr == std::tuple<int, double, int>{1, 2., 3});
    }
}
