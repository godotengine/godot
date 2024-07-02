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

#include <tuple>

#include <thrust/detail/execute_with_dependencies.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template<template<typename> class ExecutionPolicyCRTPBase>
struct dependencies_aware_execution_policy
{
    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    after(Dependencies&& ...dependencies) const
    {
        return { capture_as_dependency(THRUST_FWD(dependencies))... };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    after(std::tuple<Dependencies...>& dependencies) const
    {
        return { capture_as_dependency(dependencies) };
    }
    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    after(std::tuple<Dependencies...>&& dependencies) const
    {
        return { capture_as_dependency(std::move(dependencies)) };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(Dependencies&& ...dependencies) const
    {
        return { capture_as_dependency(THRUST_FWD(dependencies))... };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(std::tuple<Dependencies...>& dependencies) const
    {
        return { capture_as_dependency(dependencies) };
    }
    template<typename ...Dependencies>
    __host__
    thrust::detail::execute_with_dependencies<
        ExecutionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(std::tuple<Dependencies...>&& dependencies) const
    {
        return { capture_as_dependency(std::move(dependencies)) };
    }
};

} // end detail

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

