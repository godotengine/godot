/*
 * Copyright (c) 2017 Ambroz Bizjak
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <cstddef>

#include <type_traits>
#include <functional>
#include <utility>
#include <new>

namespace riscv {

/**
 * @ingroup misc
 * @defgroup function Function Wrapper
 * @brief Lightweight polymorphic function wrapper and related utilities
 * 
 * The @ref Function<Ret(Args...)> "Function" class is a general-purpose lightweight
 * polymorphic function wrapper. It has intentionally very limited storage capabilities
 * in order to ensure minimal overhead and no possibility of exceptions when copying.
 * In this documentation, "function object" refers to an instance of @ref
 * Function<Ret(Args...)> "Function".
 * 
 * Function objects are most often used for asynchronous callbacks. However this is not
 * the only possible mechanism for that purpose, virtual functions being the other major
 * one as used in the stack. The choice of which to use is often not simple, but a
 * general rule is that if more than one callback function is needed then virtual
 * functions in the same class should be considered.
 * 
 * In practical use, valid function objects are created in the following ways:
 * - From a combination of non-static member function and pointer to object, using the
 *   macros @ref AIPSTACK_BIND_MEMBER and @ref AIPSTACK_BIND_MEMBER_TN (which one depends
 *   on the context). This approach should be preferred when function objects are used
 *   for asynchronous callbacks.
 * - From a lambda object using the Function(Callable) constructor. The lambda object
 *   must meet the requirements specified for this constructor which means that one
 *   is very limited in what the lambda can capture, though capturing a single pointer
 *   or reference specifically should work.
 * 
 * @{
 */

/**
 * Maximum size of callable objects that @ref Function<Ret(Args...)> "Function" can
 * store when using the Function(Callable) constructor.
 */
inline constexpr std::size_t FunctionStorageSize = sizeof(void *) * 3;

template<typename>
class Function;

/**
 * A general-purpose lightweight polymorphic function wrapper.
 * 
 * Consult the @ref function module description for an introduction.
 * 
 * A function object is always either empty or stores a callable object. The @ref
 * operator bool() "bool operator" can be used to determine which is the case.
 * 
 * @tparam Ret Return type (may be void).
 * @tparam Args Argument types.
 */
template<typename Ret, typename ...Args>
class Function<Ret(Args...)>
{
    struct Storage {
        alignas(alignof(void*)) char data[FunctionStorageSize];
    };

    using FunctionPointerType = Ret (*) (Storage, Args...);

public:
    /**
     * Default constructor, constructs an empty function object.
     */
    inline Function () noexcept :
        m_func_ptr(nullptr)
    {}

    /**
     * Constructor from nullptr, constructs an empty function object.
     */
    inline Function (std::nullptr_t) noexcept :
        Function()
    {}

    /**
     * Constructor from a callable, constructs a function object storing the
     * callable.
     * 
     * The `Callable` type must satisfy the following requirements:
     * - Its size must be less than or equal to @ref FunctionStorageSize.
     * - It must be trivially copy-constructible.
     * - It must be trivially destructible.
     * - It must be possible to "call" a const object of that type with arguments
     *   of types `Args` and convert the return value to type `Ret` (see below for
     *   details).
     * 
     * When the function object storing this callable is invoked using @ref
     * operator()() "operator()", the callable is invoked using an expression like
     * `callable(std::forward<Args>(args)...)`, where `callable` is a const
     * reference to a copy of the `Callable` object and `args` are the arguments
     * declared as `Args ...args`. The result of this expression is returned in
     * the context of a function returning type `Ret`, so it must be implicitly
     * convertible to `Ret`.
     * 
     * @tparam Callable Type of callable object to be stored (see description for
     *         requirements).
     * @param callable Callable object to be stored in the function object.
     */
    template<typename Callable>
    Function (Callable callable) noexcept
    {
        static_assert(sizeof(Callable) <= FunctionStorageSize,
                      "Callable too large (greater than FunctionStorageSize)");
        static_assert(std::is_trivially_copy_constructible_v<Callable>,
                      "Callable not trivially copy constructible");
        static_assert(std::is_trivially_destructible_v<Callable>,
                      "Callable not trivially destructible");

        m_func_ptr = &trampoline<Callable>;

        new(reinterpret_cast<Callable *>(m_storage.data)) Callable(callable);
    }

    /**
     * Determine whether the function object stores a callable.
     * 
     */
	bool operator==(std::nullptr_t) const noexcept { return m_func_ptr == nullptr; }
	bool operator!=(std::nullptr_t) const noexcept { return m_func_ptr != nullptr; }

    /**
     * Invoke the stored callable object.
     * 
     * @note The behavior is undefined if the function object is empty.
     * 
     * @param args Arguments forwarded to the stored callable object.
     * @return Value returned by the invocation of the callable object.
     */
    inline Ret operator() (Args ...args) const
    {
        return (*m_func_ptr)(m_storage, std::forward<Args>(args)...);
    }

    inline FunctionPointerType get() const noexcept
    {
        return m_func_ptr;
    }

private:
    template<typename Callable>
    static Ret trampoline (Storage storage, Args ...args)
    {
        Callable const *c = reinterpret_cast<Callable const *>(storage.data);
        return (*c)(std::forward<Args>(args)...);
    }

private:
    FunctionPointerType m_func_ptr;
    Storage m_storage;
};

/**
 * Wrap a const reference using `std::reference_wrapper`.
 * 
 * This is intended to be used together with the Function(Callable) constructor to
 * bypass the restrictions regarding object size and trivial
 * copy-construction/destruction.
 * 
 * @warning When this is used to construct a function object, the resulting function
 * object references the original callable as passed to this function and must not
 * be invoked after the callable has been destructed.
 * 
 * @tparam Callable Type of object to which a reference is to be wrapped.
 * @param callable Reference to object to be wrapped.
 * @return Wrapped reference: `std::reference_wrapper<Callable const>(callable)`.
 */
template<typename Callable>
inline std::reference_wrapper<Callable const> RefFunc (Callable const &callable) noexcept
{
    return std::reference_wrapper<Callable const>(callable);
}

namespace BindPrivate {
    template<typename Container, typename Ret, typename ...Args>
    struct BindImpl {
        template<Ret (Container::*MemberFunc)(Args...)>
        class Callable {
        public:
            inline constexpr Callable (Container *container) :
                m_container(container)
            {}

            inline Ret operator() (Args ...args) const
            {
                return (m_container->*MemberFunc)(std::forward<Args>(args)...);
            }

            inline Function<Ret(Args...)> toFunction() const
            {
                return Function<Ret(Args...)>(*this);
            }

        private:
            Container *m_container;
        };
    };

    template<typename Container, typename Ret, typename ...Args>
    struct BindImplConst {
        template<Ret (Container::*MemberFunc)(Args...) const>
        class Callable {
        public:
            inline constexpr Callable (Container const *container) :
                m_container(container)
            {}

            inline Ret operator() (Args ...args) const
            {
                return (m_container->*MemberFunc)(std::forward<Args>(args)...);
            }

            inline Function<Ret(Args...)> toFunction() const
            {
                return Function<Ret(Args...)>(*this);
            }

        private:
            Container const *m_container;
        };
    };

    template<typename Container, typename Ret, typename ...Args>
    BindImpl<Container, Ret, Args...> DeduceImpl (Ret (Container::*)(Args...));

    template<typename Container, typename Ret, typename ...Args>
    BindImplConst<Container, Ret, Args...> DeduceImpl (Ret (Container::*)(Args...) const);
}

} // riscv
