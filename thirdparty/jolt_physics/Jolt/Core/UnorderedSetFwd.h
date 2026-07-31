// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_NAMESPACE_BEGIN

// Forward declaration of UnorderedSet (defined in UnorderedSet.h).
// This is provided because compiling UnorderedSet.h can be expensive due to its use of templates.
template <class Key, class Hash = JPH::Hash<Key>, class KeyEqual = std::equal_to<Key>>
class UnorderedSet;

JPH_NAMESPACE_END
