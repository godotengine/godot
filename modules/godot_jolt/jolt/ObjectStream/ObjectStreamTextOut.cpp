// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#ifdef JPH_OBJECT_STREAM

#include <Jolt/ObjectStream/ObjectStreamTextOut.h>
#include <Jolt/Core/StringTools.h>

JPH_NAMESPACE_BEGIN

ObjectStreamTextOut::ObjectStreamTextOut(ostream &inStream) :
	ObjectStreamOut(inStream)
{
	WriteWord(StringFormat("TOS%2d.%02d", ObjectStream::sVersion, ObjectStream::sRevision));
}

void ObjectStreamTextOut::WriteDataType(EOSDataType inType)
{
	switch (inType)
	{
	case EOSDataType::Declare:		WriteWord("declare ");		break;
	case EOSDataType::Object:		WriteWord("object ");		break;
	case EOSDataType::Instance:		WriteWord("instance ");		break;
	case EOSDataType::Pointer:		WriteWord("pointer ");		break;
	case EOSDataType::Array:		WriteWord("array ");		break;
	case EOSDataType::T_uint8:		WriteWord("uint8");			break;
	case EOSDataType::T_uint16:		WriteWord("uint16");		break;
	case EOSDataType::T_int:		WriteWord("int");			break;
	case EOSDataType::T_uint32:		WriteWord("uint32");		break;
	case EOSDataType::T_uint64:		WriteWord("uint64");		break;
	case EOSDataType::T_float:		WriteWord("float");			break;
	case EOSDataType::T_double:		WriteWord("double");		break;
	case EOSDataType::T_bool:		WriteWord("bool");			break;
	case EOSDataType::T_String:		WriteWord("string");		break;
	case EOSDataType::T_Float3:		WriteWord("float3");		break;
	case EOSDataType::T_Double3:	WriteWord("double3");		break;
	case EOSDataType::T_Vec3:		WriteWord("vec3");			break;
	case EOSDataType::T_DVec3:		WriteWord("dvec3");			break;
	case EOSDataType::T_Vec4:		WriteWord("vec4");			break;
	case EOSDataType::T_Quat:		WriteWord("quat");			break;
	case EOSDataType::T_Mat44:		WriteWord("mat44");			break;
	case EOSDataType::T_DMat44:		WriteWord("dmat44");		break;
	case EOSDataType::Invalid:
	default:						JPH_ASSERT(false);			break;
	}
}

void ObjectStreamTextOut::WriteName(const char *inName)
{
	WriteWord(String(inName) + " ");
}

void ObjectStreamTextOut::WriteIdentifier(Identifier inIdentifier)
{
	WriteWord(StringFormat("%08X", inIdentifier));
}

void ObjectStreamTextOut::WriteCount(uint32 inCount)
{
	WriteWord(std::to_string(inCount));
}

void ObjectStreamTextOut::WritePrimitiveData(const uint8 &inPrimitive)
{
	WriteWord(std::to_string(inPrimitive));
}

void ObjectStreamTextOut::WritePrimitiveData(const uint16 &inPrimitive)
{
	WriteWord(std::to_string(inPrimitive));
}

void ObjectStreamTextOut::WritePrimitiveData(const int &inPrimitive)
{
	WriteWord(std::to_string(inPrimitive));
}

void ObjectStreamTextOut::WritePrimitiveData(const uint32 &inPrimitive)
{
	WriteWord(std::to_string(inPrimitive));
}

void ObjectStreamTextOut::WritePrimitiveData(const uint64 &inPrimitive)
{
	WriteWord(std::to_string(inPrimitive));
}

void ObjectStreamTextOut::WritePrimitiveData(const float &inPrimitive)
{
	std::ostringstream stream;
	stream.precision(9);
	stream << inPrimitive;
	WriteWord(stream.str());
}

void ObjectStreamTextOut::WritePrimitiveData(const double &inPrimitive)
{
	std::ostringstream stream;
	stream.precision(17);
	stream << inPrimitive;
	WriteWord(stream.str());
}

void ObjectStreamTextOut::WritePrimitiveData(const bool &inPrimitive)
{
	WriteWord(inPrimitive? "true" : "false");
}

void ObjectStreamTextOut::WritePrimitiveData(const Float3 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.x);
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.y);
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.z);
}

void ObjectStreamTextOut::WritePrimitiveData(const Double3 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.x);
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.y);
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.z);
}

void ObjectStreamTextOut::WritePrimitiveData(const Vec3 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetX());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetY());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetZ());
}

void ObjectStreamTextOut::WritePrimitiveData(const DVec3 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetX());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetY());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetZ());
}

void ObjectStreamTextOut::WritePrimitiveData(const Vec4 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetX());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetY());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetZ());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetW());
}

void ObjectStreamTextOut::WritePrimitiveData(const Quat &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetX());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetY());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetZ());
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetW());
}

void ObjectStreamTextOut::WritePrimitiveData(const Mat44 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetColumn4(0));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetColumn4(1));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetColumn4(2));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetColumn4(3));
}

void ObjectStreamTextOut::WritePrimitiveData(const DMat44 &inPrimitive)
{
	WritePrimitiveData(inPrimitive.GetColumn4(0));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetColumn4(1));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetColumn4(2));
	WriteChar(' ');
	WritePrimitiveData(inPrimitive.GetTranslation());
}

void ObjectStreamTextOut::WritePrimitiveData(const String &inPrimitive)
{
	String temporary(inPrimitive);
	StringReplace(temporary, "\\", "\\\\");
	StringReplace(temporary, "\n", "\\n");
	StringReplace(temporary, "\t", "\\t");
	StringReplace(temporary, "\"", "\\\"");
	WriteWord(String("\"") + temporary + String("\""));
}

void ObjectStreamTextOut::HintNextItem()
{
	WriteWord("\r\n");
	for (int i = 0; i < mIndentation; ++i)
		WriteWord("  ");
}

void ObjectStreamTextOut::HintIndentUp()
{
	++mIndentation;
}

void ObjectStreamTextOut::HintIndentDown()
{
	--mIndentation;
}

void ObjectStreamTextOut::WriteChar(char inChar)
{
	mStream.put(inChar);
}

void ObjectStreamTextOut::WriteWord(const string_view &inWord)
{
	mStream << inWord;
}

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
