// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <limits>

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename T>
struct numeric_limits : std::numeric_limits<T> {};

THRUST_NAMESPACE_END
