
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_OPTIONAL_HPP_INCLUDED
#define CATCH_OPTIONAL_HPP_INCLUDED

#include <catch2/internal/catch_move_and_forward.hpp>

#include <cassert>

namespace Catch {

    // An optional type
    template<typename T>
    class Optional {
    public:
        Optional(): nullableValue( nullptr ) {}
        ~Optional() { reset(); }

        Optional( T const& _value ):
            nullableValue( new ( storage ) T( _value ) ) {}
        Optional( T&& _value ):
            nullableValue( new ( storage ) T( CATCH_MOVE( _value ) ) ) {}

        Optional& operator=( T const& _value ) {
            reset();
            nullableValue = new ( storage ) T( _value );
            return *this;
        }
        Optional& operator=( T&& _value ) {
            reset();
            nullableValue = new ( storage ) T( CATCH_MOVE( _value ) );
            return *this;
        }

        Optional( Optional const& _other ):
            nullableValue( _other ? new ( storage ) T( *_other ) : nullptr ) {}
        Optional( Optional&& _other ):
            nullableValue( _other ? new ( storage ) T( CATCH_MOVE( *_other ) )
                                  : nullptr ) {}

        Optional& operator=( Optional const& _other ) {
            if ( &_other != this ) {
                reset();
                if ( _other ) { nullableValue = new ( storage ) T( *_other ); }
            }
            return *this;
        }
        Optional& operator=( Optional&& _other ) {
            if ( &_other != this ) {
                reset();
                if ( _other ) {
                    nullableValue = new ( storage ) T( CATCH_MOVE( *_other ) );
                }
            }
            return *this;
        }

        void reset() {
            if ( nullableValue ) { nullableValue->~T(); }
            nullableValue = nullptr;
        }

        T& operator*() {
            assert(nullableValue);
            return *nullableValue;
        }
        T const& operator*() const {
            assert(nullableValue);
            return *nullableValue;
        }
        T* operator->() {
            assert(nullableValue);
            return nullableValue;
        }
        const T* operator->() const {
            assert(nullableValue);
            return nullableValue;
        }

        T valueOr( T const& defaultValue ) const {
            return nullableValue ? *nullableValue : defaultValue;
        }

        bool some() const { return nullableValue != nullptr; }
        bool none() const { return nullableValue == nullptr; }

        bool operator !() const { return nullableValue == nullptr; }
        explicit operator bool() const {
            return some();
        }

        friend bool operator==(Optional const& a, Optional const& b) {
            if (a.none() && b.none()) {
                return true;
            } else if (a.some() && b.some()) {
                return *a == *b;
            } else {
                return false;
            }
        }
        friend bool operator!=(Optional const& a, Optional const& b) {
            return !( a == b );
        }

    private:
        T* nullableValue;
        alignas(alignof(T)) char storage[sizeof(T)];
    };

} // end namespace Catch

#endif // CATCH_OPTIONAL_HPP_INCLUDED
