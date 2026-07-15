// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/UnorderedMapFwd.h>

JPH_NAMESPACE_BEGIN

namespace StreamUtils {

template <class Type>
using ObjectToIDMap = UnorderedMap<const Type *, uint32>;

template <class Type>
using IDToObjectMap = Array<Ref<Type>>;

} // StreamUtils

JPH_NAMESPACE_END
