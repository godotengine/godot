// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/RTTI.h>

JPH_NAMESPACE_BEGIN

/// Helper functions to get the underlying RTTI type of a type (so e.g. Array<sometype> will return sometype)
template <class T>
const RTTI *GetPrimitiveTypeOfType(T *)
{
	return GetRTTIOfType((T *)nullptr);
}

template <class T>
const RTTI *GetPrimitiveTypeOfType(T **)
{
	return GetRTTIOfType((T *)nullptr);
}

template <class T>
const RTTI *GetPrimitiveTypeOfType(Ref<T> *)
{
	return GetRTTIOfType((T *)nullptr);
}

template <class T>
const RTTI *GetPrimitiveTypeOfType(RefConst<T> *)
{
	return GetRTTIOfType((T *)nullptr);
}

template <class T, class A>
const RTTI *GetPrimitiveTypeOfType(Array<T, A> *)
{
	return GetPrimitiveTypeOfType((T *)nullptr);
}

template <class T, uint N>
const RTTI *GetPrimitiveTypeOfType(StaticArray<T, N> *)
{
	return GetPrimitiveTypeOfType((T *)nullptr);
}

template <class T, uint N>
const RTTI *GetPrimitiveTypeOfType(T (*)[N])
{
	return GetPrimitiveTypeOfType((T *)nullptr);
}

JPH_NAMESPACE_END
