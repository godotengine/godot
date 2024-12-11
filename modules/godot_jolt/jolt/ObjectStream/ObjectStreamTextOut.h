// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/ObjectStream/ObjectStreamOut.h>

#ifdef JPH_OBJECT_STREAM

JPH_NAMESPACE_BEGIN

/// Implementation of ObjectStream text output stream.
class JPH_EXPORT ObjectStreamTextOut : public ObjectStreamOut
{
public:
	JPH_OVERRIDE_NEW_DELETE

	/// Constructor and destructor
	explicit					ObjectStreamTextOut(ostream &inStream);

	///@name Output type specific operations
	virtual void				WriteDataType(EOSDataType inType) override;
	virtual void				WriteName(const char *inName) override;
	virtual void				WriteIdentifier(Identifier inIdentifier) override;
	virtual void				WriteCount(uint32 inCount) override;

	virtual void				WritePrimitiveData(const uint8 &inPrimitive) override;
	virtual void				WritePrimitiveData(const uint16 &inPrimitive) override;
	virtual void				WritePrimitiveData(const int &inPrimitive) override;
	virtual void				WritePrimitiveData(const uint32 &inPrimitive) override;
	virtual void				WritePrimitiveData(const uint64 &inPrimitive) override;
	virtual void				WritePrimitiveData(const float &inPrimitive) override;
	virtual void				WritePrimitiveData(const double &inPrimitive) override;
	virtual void				WritePrimitiveData(const bool &inPrimitive) override;
	virtual void				WritePrimitiveData(const String &inPrimitive) override;
	virtual void				WritePrimitiveData(const Float3 &inPrimitive) override;
	virtual void				WritePrimitiveData(const Double3 &inPrimitive) override;
	virtual void				WritePrimitiveData(const Vec3 &inPrimitive) override;
	virtual void				WritePrimitiveData(const DVec3 &inPrimitive) override;
	virtual void				WritePrimitiveData(const Vec4 &inPrimitive) override;
	virtual void				WritePrimitiveData(const Quat &inPrimitive) override;
	virtual void				WritePrimitiveData(const Mat44 &inPrimitive) override;
	virtual void				WritePrimitiveData(const DMat44 &inPrimitive) override;

	///@name Layout hints (for text output)
	virtual void				HintNextItem() override;
	virtual void				HintIndentUp() override;
	virtual void				HintIndentDown() override;

private:
	void						WriteChar(char inChar);
	void						WriteWord(const string_view &inWord);

	int							mIndentation = 0;
};

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
