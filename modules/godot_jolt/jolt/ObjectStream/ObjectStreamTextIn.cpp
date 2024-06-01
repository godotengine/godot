// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#ifdef JPH_OBJECT_STREAM

#include <Jolt/ObjectStream/ObjectStreamTextIn.h>

JPH_NAMESPACE_BEGIN

ObjectStreamTextIn::ObjectStreamTextIn(istream &inStream) :
	ObjectStreamIn(inStream)
{
}

bool ObjectStreamTextIn::ReadDataType(EOSDataType &outType)
{
	String token;
	if (ReadWord(token))
	{
		transform(token.begin(), token.end(), token.begin(), [](char inValue) { return (char)tolower(inValue); });
		if (token == "declare")
			outType = EOSDataType::Declare;
		else if (token == "object")
			outType = EOSDataType::Object;
		else if (token == "instance")
			outType = EOSDataType::Instance;
		else if (token == "pointer")
			outType = EOSDataType::Pointer;
		else if (token == "array")
			outType  = EOSDataType::Array;
		else if (token == "uint8")
			outType  = EOSDataType::T_uint8;
		else if (token == "uint16")
			outType  = EOSDataType::T_uint16;
		else if (token == "int")
			outType  = EOSDataType::T_int;
		else if (token == "uint32")
			outType  = EOSDataType::T_uint32;
		else if (token == "uint64")
			outType  = EOSDataType::T_uint64;
		else if (token == "float")
			outType  = EOSDataType::T_float;
		else if (token == "double")
			outType  = EOSDataType::T_double;
		else if (token == "bool")
			outType  = EOSDataType::T_bool;
		else if (token == "string")
			outType  = EOSDataType::T_String;
		else if (token == "float3")
			outType  = EOSDataType::T_Float3;
		else if (token == "double3")
			outType  = EOSDataType::T_Double3;
		else if (token == "vec3")
			outType  = EOSDataType::T_Vec3;
		else if (token == "dvec3")
			outType  = EOSDataType::T_DVec3;
		else if (token == "vec4")
			outType  = EOSDataType::T_Vec4;
		else if (token == "quat")
			outType  = EOSDataType::T_Quat;
		else if (token == "mat44")
			outType  = EOSDataType::T_Mat44;
		else if (token == "dmat44")
			outType  = EOSDataType::T_DMat44;
		else
		{
			Trace("ObjectStreamTextIn: Found unknown data type.");
			return false;
		}
		return true;
	}
	return false;
}

bool ObjectStreamTextIn::ReadName(String &outName)
{
	return ReadWord(outName);
}

bool ObjectStreamTextIn::ReadIdentifier(Identifier &outIdentifier)
{
	String token;
	if (!ReadWord(token))
		return false;
	outIdentifier = (uint32)std::strtoul(token.c_str(), nullptr, 16);
	if (errno == ERANGE)
	{
		outIdentifier = sNullIdentifier;
		return false;
	}
	return true;
}

bool ObjectStreamTextIn::ReadCount(uint32 &outCount)
{
	return ReadPrimitiveData(outCount);
}

bool ObjectStreamTextIn::ReadPrimitiveData(uint8 &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	uint32 temporary;
	IStringStream stream(token);
	stream >> temporary;
	if (!stream.fail())
	{
		outPrimitive = (uint8)temporary;
		return true;
	}
	return false;
}

bool ObjectStreamTextIn::ReadPrimitiveData(uint16 &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	uint32 temporary;
	IStringStream stream(token);
	stream >> temporary;
	if (!stream.fail())
	{
		outPrimitive = (uint16)temporary;
		return true;
	}
	return false;
}

bool ObjectStreamTextIn::ReadPrimitiveData(int &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	IStringStream stream(token);
	stream >> outPrimitive;
	return !stream.fail();
}

bool ObjectStreamTextIn::ReadPrimitiveData(uint32 &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	IStringStream stream(token);
	stream >> outPrimitive;
	return !stream.fail();
}

bool ObjectStreamTextIn::ReadPrimitiveData(uint64 &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	IStringStream stream(token);
	stream >> outPrimitive;
	return !stream.fail();
}

bool ObjectStreamTextIn::ReadPrimitiveData(float &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	IStringStream stream(token);
	stream >> outPrimitive;
	return !stream.fail();
}

bool ObjectStreamTextIn::ReadPrimitiveData(double &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	IStringStream stream(token);
	stream >> outPrimitive;
	return !stream.fail();
}

bool ObjectStreamTextIn::ReadPrimitiveData(bool &outPrimitive)
{
	String token;
	if (!ReadWord(token))
		return false;
	transform(token.begin(), token.end(), token.begin(), [](char inValue) { return (char)tolower(inValue); });
	outPrimitive = token == "true";
	return outPrimitive || token == "false";
}

bool ObjectStreamTextIn::ReadPrimitiveData(String &outPrimitive)
{
	outPrimitive.clear();

	char c;

	// Skip whitespace
	for (;;)
	{
		if (!ReadChar(c))
			return false;

		if (!isspace(c))
			break;
	}

	// Check if it is a opening quote
	if (c != '\"')
		return false;

	// Read string and interpret special characters
	String result;
	bool escaped = false;
	for (;;)
	{
		if (!ReadChar(c))
			break;

		switch (c)
		{
		case '\n':
		case '\t':
			break;

		case '\\':
			if (escaped)
			{
				result += '\\';
				escaped = false;
			}
			else
				escaped = true;
			break;

		case 'n':
			if (escaped)
			{
				result += '\n';
				escaped = false;
			}
			else
				result += 'n';
			break;

		case 't':
			if (escaped)
			{
				result += '\t';
				escaped = false;
			}
			else
				result += 't';
			break;

		case '\"':
			if (escaped)
			{
				result += '\"';
				escaped = false;
			}
			else
			{
				// Found closing double quote
				outPrimitive = result;
				return true;
			}
			break;

		default:
			if (escaped)
				escaped = false;
			else
				result += c;
			break;
		}
	}

	return false;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Float3 &outPrimitive)
{
	float x, y, z;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z))
		return false;
	outPrimitive = Float3(x, y, z);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Double3 &outPrimitive)
{
	double x, y, z;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z))
		return false;
	outPrimitive = Double3(x, y, z);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Vec3 &outPrimitive)
{
	float x, y, z;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z))
		return false;
	outPrimitive = Vec3(x, y, z);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(DVec3 &outPrimitive)
{
	double x, y, z;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z))
		return false;
	outPrimitive = DVec3(x, y, z);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Vec4 &outPrimitive)
{
	float x, y, z, w;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z) || !ReadPrimitiveData(w))
		return false;
	outPrimitive = Vec4(x, y, z, w);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Quat &outPrimitive)
{
	float x, y, z, w;
	if (!ReadPrimitiveData(x) || !ReadPrimitiveData(y) || !ReadPrimitiveData(z) || !ReadPrimitiveData(w))
		return false;
	outPrimitive = Quat(x, y, z, w);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(Mat44 &outPrimitive)
{
	Vec4 c0, c1, c2, c3;
	if (!ReadPrimitiveData(c0) || !ReadPrimitiveData(c1) || !ReadPrimitiveData(c2) || !ReadPrimitiveData(c3))
		return false;
	outPrimitive = Mat44(c0, c1, c2, c3);
	return true;
}

bool ObjectStreamTextIn::ReadPrimitiveData(DMat44 &outPrimitive)
{
	Vec4 c0, c1, c2;
	DVec3 c3;
	if (!ReadPrimitiveData(c0) || !ReadPrimitiveData(c1) || !ReadPrimitiveData(c2) || !ReadPrimitiveData(c3))
		return false;
	outPrimitive = DMat44(c0, c1, c2, c3);
	return true;
}

bool ObjectStreamTextIn::ReadChar(char &outChar)
{
	mStream.get(outChar);
	return !mStream.eof();
}

bool ObjectStreamTextIn::ReadWord(String &outWord)
{
	outWord.clear();

	char c;

	// Skip whitespace
	for (;;)
	{
		if (!ReadChar(c))
			return false;

		if (!isspace(c))
			break;
	}

	// Read word
	for (;;)
	{
		outWord += c;

		if (!ReadChar(c))
			break;

		if (isspace(c))
			break;
	}

	return !outWord.empty();
}

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
