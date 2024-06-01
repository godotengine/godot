// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#ifdef JPH_OBJECT_STREAM

#include <Jolt/ObjectStream/ObjectStreamBinaryOut.h>
#include <Jolt/Core/StringTools.h>

JPH_NAMESPACE_BEGIN

ObjectStreamBinaryOut::ObjectStreamBinaryOut(ostream &inStream) :
	ObjectStreamOut(inStream)
{
	String header;
	header = StringFormat("BOS%2d.%02d", ObjectStream::sVersion, ObjectStream::sRevision);
	mStream.write(header.c_str(), header.size());
}

void ObjectStreamBinaryOut::WriteDataType(EOSDataType inType)
{
	mStream.write((const char *)&inType, sizeof(inType));
}

void ObjectStreamBinaryOut::WriteName(const char *inName)
{
	WritePrimitiveData(String(inName));
}

void ObjectStreamBinaryOut::WriteIdentifier(Identifier inIdentifier)
{
	mStream.write((const char *)&inIdentifier, sizeof(inIdentifier));
}

void ObjectStreamBinaryOut::WriteCount(uint32 inCount)
{
	mStream.write((const char *)&inCount, sizeof(inCount));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const uint8 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const uint16 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const int &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const uint32 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const uint64 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const float &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const double &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const bool &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const String &inPrimitive)
{
	// Empty strings are trivial
	if (inPrimitive.empty())
	{
		WritePrimitiveData((uint32)0);
		return;
	}

	// Check if we've already written this string
	StringTable::iterator i = mStringTable.find(inPrimitive);
	if (i != mStringTable.end())
	{
		WritePrimitiveData(i->second);
		return;
	}

	// Insert string in table
	mStringTable.try_emplace(inPrimitive, mNextStringID);
	mNextStringID++;

	// Write string
	uint32 len = min((uint32)inPrimitive.size(), (uint32)0x7fffffff);
	WritePrimitiveData(len);
	mStream.write(inPrimitive.c_str(), len);
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Float3 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(Float3));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Double3 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(Double3));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Vec3 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, 3 * sizeof(float));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const DVec3 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, 3 * sizeof(double));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Vec4 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Quat &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const Mat44 &inPrimitive)
{
	mStream.write((const char *)&inPrimitive, sizeof(inPrimitive));
}

void ObjectStreamBinaryOut::WritePrimitiveData(const DMat44 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetColumn4(0));
	WritePrimitiveData(inPrimitive.GetColumn4(1));
	WritePrimitiveData(inPrimitive.GetColumn4(2));
	WritePrimitiveData(inPrimitive.GetTranslation());
}

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM

