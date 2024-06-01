// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#ifdef JPH_OBJECT_STREAM

#include <Jolt/ObjectStream/ObjectStreamBinaryIn.h>

JPH_NAMESPACE_BEGIN

ObjectStreamBinaryIn::ObjectStreamBinaryIn(istream &inStream) :
	ObjectStreamIn(inStream)
{
}

bool ObjectStreamBinaryIn::ReadDataType(EOSDataType &outType)
{
	uint32 type;
	mStream.read((char *)&type, sizeof(type));
	if (mStream.fail()) return false;
	outType = (EOSDataType)type;
	return true;
}

bool ObjectStreamBinaryIn::ReadName(String &outName)
{
	return ReadPrimitiveData(outName);
}

bool ObjectStreamBinaryIn::ReadIdentifier(Identifier &outIdentifier)
{
	Identifier id;
	mStream.read((char *)&id, sizeof(id));
	if (mStream.fail()) return false;
	outIdentifier = id;
	return true;
}

bool ObjectStreamBinaryIn::ReadCount(uint32 &outCount)
{
	uint32 count;
	mStream.read((char *)&count, sizeof(count));
	if (mStream.fail()) return false;
	outCount = count;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(uint8 &outPrimitive)
{
	uint8 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(uint16 &outPrimitive)
{
	uint16 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(int &outPrimitive)
{
	int primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(uint32 &outPrimitive)
{
	uint32 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(uint64 &outPrimitive)
{
	uint64 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(float &outPrimitive)
{
	float primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(double &outPrimitive)
{
	double primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(bool &outPrimitive)
{
	bool primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(String &outPrimitive)
{
	// Read length or ID of string
	uint32 len;
	if (!ReadPrimitiveData(len))
		return false;

	// Check empty string
	if (len == 0)
	{
		outPrimitive.clear();
		return true;
	}

	// Check if it is an ID in the string table
	if (len & 0x80000000)
	{
		StringTable::iterator i = mStringTable.find(len);
		if (i == mStringTable.end())
			return false;
		outPrimitive = i->second;
		return true;
	}

	// Read the string
	char *data = (char *)JPH_STACK_ALLOC(len + 1);
	mStream.read(data, len);
	if (mStream.fail()) return false;
	data[len] = 0;
	outPrimitive = data;

	// Insert string in table
	mStringTable.try_emplace(mNextStringID, outPrimitive);
	mNextStringID++;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Float3 &outPrimitive)
{
	Float3 primitive;
	mStream.read((char *)&primitive, sizeof(Float3));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Double3 &outPrimitive)
{
	Double3 primitive;
	mStream.read((char *)&primitive, sizeof(Double3));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Vec3 &outPrimitive)
{
	Float3 primitive;
	mStream.read((char *)&primitive, sizeof(Float3));
	if (mStream.fail()) return false;
	outPrimitive = Vec3(primitive); // Use Float3 constructor so that we initialize W too
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(DVec3 &outPrimitive)
{
	Double3 primitive;
	mStream.read((char *)&primitive, sizeof(Double3));
	if (mStream.fail()) return false;
	outPrimitive = DVec3(primitive); // Use Float3 constructor so that we initialize W too
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Vec4 &outPrimitive)
{
	Vec4 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Quat &outPrimitive)
{
	Quat primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(Mat44 &outPrimitive)
{
	Mat44 primitive;
	mStream.read((char *)&primitive, sizeof(primitive));
	if (mStream.fail()) return false;
	outPrimitive = primitive;
	return true;
}

bool ObjectStreamBinaryIn::ReadPrimitiveData(DMat44 &outPrimitive)
{
	Vec4 c0, c1, c2;
	DVec3 c3;
	if (!ReadPrimitiveData(c0) || !ReadPrimitiveData(c1) || !ReadPrimitiveData(c2) || !ReadPrimitiveData(c3))
		return false;
	outPrimitive = DMat44(c0, c1, c2, c3);
	return true;
}

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
