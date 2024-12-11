// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/ObjectStream/ObjectStream.h>

#ifdef JPH_OBJECT_STREAM

JPH_NAMESPACE_BEGIN

// Define macro to declare functions for a specific primitive type
#define JPH_DECLARE_PRIMITIVE(name)																\
	bool	OSIsType(name *, int inArrayDepth, EOSDataType inDataType, const char *inClassName) \
	{																							\
		return inArrayDepth == 0 && inDataType == EOSDataType::T_##name;						\
	}																							\
	bool	OSReadData(IObjectStreamIn &ioStream, name &outPrimitive)							\
	{																							\
		return ioStream.ReadPrimitiveData(outPrimitive);										\
	}																							\
	void	OSWriteDataType(IObjectStreamOut &ioStream, name *)									\
	{																							\
		ioStream.WriteDataType(EOSDataType::T_##name);											\
	}																							\
	void	OSWriteData(IObjectStreamOut &ioStream, const name &inPrimitive)					\
	{																							\
		ioStream.HintNextItem();																\
		ioStream.WritePrimitiveData(inPrimitive);												\
	}

// This file uses the JPH_DECLARE_PRIMITIVE macro to define all types
#include <Jolt/ObjectStream/ObjectStreamTypes.h>

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
