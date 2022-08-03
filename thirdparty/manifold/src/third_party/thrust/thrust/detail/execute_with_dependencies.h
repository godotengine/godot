/*
 *  Copyright 2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/type_deduction.h>
#include <thrust/type_traits/remove_cvref.h>

#include <tuple>
#include <type_traits>

THRUST_NAMESPACE_BEGIN

namespace detail
{

struct capture_as_dependency_fn
{
  template<typename Dependency>
  auto operator()(Dependency&& dependency) const
  THRUST_DECLTYPE_RETURNS(capture_as_dependency(THRUST_FWD(dependency)))
};

// Default implementation: universal forwarding.
template<typename Dependency>
auto capture_as_dependency(Dependency&& dependency)
THRUST_DECLTYPE_RETURNS(THRUST_FWD(dependency))

template<typename... Dependencies>
auto capture_as_dependency(std::tuple<Dependencies...>& dependencies)
THRUST_DECLTYPE_RETURNS(
  tuple_for_each(THRUST_FWD(dependencies), capture_as_dependency_fn{})
)

template<template<typename> class BaseSystem, typename... Dependencies>
struct execute_with_dependencies
    : BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>
{
private:
    using super_t = BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>;

    std::tuple<remove_cvref_t<Dependencies>...> dependencies;

public:
    __host__
    execute_with_dependencies(super_t const &super, Dependencies && ...dependencies)
        : super_t(super), dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(super_t const &super, UDependencies && ...deps)
        : super_t(super), dependencies(THRUST_FWD(deps)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(UDependencies && ...deps)
        : dependencies(THRUST_FWD(deps)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(super_t const &super, std::tuple<UDependencies...>&& deps)
        : super_t(super), dependencies(std::move(deps))
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(std::tuple<UDependencies...>&& deps)
        : dependencies(std::move(deps))
    {
    }

    std::tuple<remove_cvref_t<Dependencies>...>
    __host__
    extract_dependencies() 
    {
        return std::move(dependencies);
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(UDependencies&& ...udependencies) const
    {
        return { capture_as_dependency(THRUST_FWD(udependencies))... };
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>& udependencies) const
    {
        return { capture_as_dependency(udependencies) };
    }
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>&& udependencies) const
    {
        return { capture_as_dependency(std::move(udependencies)) };
    }
};

template<
    typename Allocator,
    template<typename> class BaseSystem,
    typename... Dependencies
>
struct execute_with_allocator_and_dependencies
    : BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >
{
private:
    using super_t = BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >;

    std::tuple<remove_cvref_t<Dependencies>...> dependencies;
    Allocator alloc;

public:
    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(super_t const &super, Allocator a, UDependencies && ...deps)
        : super_t(super), dependencies(THRUST_FWD(deps)...), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(Allocator a, UDependencies && ...deps)
        : dependencies(THRUST_FWD(deps)...), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(super_t const &super, Allocator a, std::tuple<UDependencies...>&& deps)
        : super_t(super), dependencies(std::move(deps)), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(Allocator a, std::tuple<UDependencies...>&& deps)
        : dependencies(std::move(deps)), alloc(a)
    {
    }

    std::tuple<remove_cvref_t<Dependencies>...>
    __host__
    extract_dependencies() 
    {
        return std::move(dependencies);
    }

    __host__
    typename std::add_lvalue_reference<Allocator>::type
    get_allocator()
    {
        return alloc;
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(UDependencies&& ...udependencies) const
    {
        return { alloc, capture_as_dependency(THRUST_FWD(udependencies))... };
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>& udependencies) const
    {
        return { alloc, capture_as_dependency(udependencies) };
    }
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>&& udependencies) const
    {
        return { alloc, capture_as_dependency(std::move(udependencies)) };
    }
};

template<template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(thrust::detail::execute_with_dependencies<BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}
template<template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(thrust::detail::execute_with_dependencies<BaseSystem, Dependencies...>& system)
{
    return std::move(system).extract_dependencies();
}

template<typename Allocator, template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}
template<typename Allocator, template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system)
{
    return std::move(system).extract_dependencies();
}

template<typename System>
__host__
std::tuple<>
extract_dependencies(System &&)
{
    return std::tuple<>{};
}

} // end detail

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011
