// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <unordered_map>
JPH_SUPPRESS_WARNINGS_STD_END

JPH_NAMESPACE_BEGIN

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>> using UnorderedMap = std::unordered_map<Key, T, Hash, KeyEqual, STLAllocator<pair<const Key, T>>>;

JPH_NAMESPACE_END
