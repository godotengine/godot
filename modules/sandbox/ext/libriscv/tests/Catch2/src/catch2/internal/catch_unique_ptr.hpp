
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_UNIQUE_PTR_HPP_INCLUDED
#define CATCH_UNIQUE_PTR_HPP_INCLUDED

#include <cassert>
#include <type_traits>

#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {
namespace Detail {
    /**
     * A reimplementation of `std::unique_ptr` for improved compilation performance
     *
     * Does not support arrays nor custom deleters.
     */
    template <typename T>
    class unique_ptr {
        T* m_ptr;
    public:
        constexpr unique_ptr(std::nullptr_t = nullptr):
            m_ptr{}
        {}
        explicit constexpr unique_ptr(T* ptr):
            m_ptr(ptr)
        {}

        template <typename U, typename = std::enable_if_t<std::is_base_of<T, U>::value>>
        unique_ptr(unique_ptr<U>&& from):
            m_ptr(from.release())
        {}

        template <typename U, typename = std::enable_if_t<std::is_base_of<T, U>::value>>
        unique_ptr& operator=(unique_ptr<U>&& from) {
            reset(from.release());

            return *this;
        }

        unique_ptr(unique_ptr const&) = delete;
        unique_ptr& operator=(unique_ptr const&) = delete;

        unique_ptr(unique_ptr&& rhs) noexcept:
            m_ptr(rhs.m_ptr) {
            rhs.m_ptr = nullptr;
        }
        unique_ptr& operator=(unique_ptr&& rhs) noexcept {
            reset(rhs.release());

            return *this;
        }

        ~unique_ptr() {
            delete m_ptr;
        }

        T& operator*() {
            assert(m_ptr);
            return *m_ptr;
        }
        T const& operator*() const {
            assert(m_ptr);
            return *m_ptr;
        }
        T* operator->() noexcept {
            assert(m_ptr);
            return m_ptr;
        }
        T const* operator->() const noexcept {
            assert(m_ptr);
            return m_ptr;
        }

        T* get() { return m_ptr; }
        T const* get() const { return m_ptr; }

        void reset(T* ptr = nullptr) {
            delete m_ptr;
            m_ptr = ptr;
        }

        T* release() {
            auto temp = m_ptr;
            m_ptr = nullptr;
            return temp;
        }

        explicit operator bool() const {
            return m_ptr != nullptr;
        }

        friend void swap(unique_ptr& lhs, unique_ptr& rhs) {
            auto temp = lhs.m_ptr;
            lhs.m_ptr = rhs.m_ptr;
            rhs.m_ptr = temp;
        }
    };

    //! Specialization to cause compile-time error for arrays
    template <typename T>
    class unique_ptr<T[]>;

    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args&&... args) {
        return unique_ptr<T>(new T(CATCH_FORWARD(args)...));
    }


} // end namespace Detail
} // end namespace Catch

#endif // CATCH_UNIQUE_PTR_HPP_INCLUDED
