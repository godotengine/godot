// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_NAMESPACE_BEGIN

// Forward declaration of UnorderedMap (defined in UnorderedMap.h).
// This is provided because compiling UnorderedMap.h can be expensive due to its use of templates.
template <class Key, class Value, class Hash = JPH::Hash<Key>, class KeyEqual = std::equal_to<Key>>
class UnorderedMap;

JPH_NAMESPACE_END
