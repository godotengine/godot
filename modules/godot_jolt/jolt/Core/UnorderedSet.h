// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <unordered_set>
JPH_SUPPRESS_WARNINGS_STD_END

JPH_NAMESPACE_BEGIN

template <class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>> using UnorderedSet = std::unordered_set<Key, Hash, KeyEqual, STLAllocator<Key>>;

JPH_NAMESPACE_END
