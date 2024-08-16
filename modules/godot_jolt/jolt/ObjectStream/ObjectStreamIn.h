// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/ObjectStream/ObjectStream.h>
#include <Jolt/Core/Reference.h>
#include <Jolt/Core/RTTI.h>
#include <Jolt/Core/UnorderedMap.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <fstream>
JPH_SUPPRESS_WARNINGS_STD_END

#ifdef JPH_OBJECT_STREAM

JPH_NAMESPACE_BEGIN

/// ObjectStreamIn contains all logic for reading an object from disk. It is the base
/// class for the text and binary input streams (ObjectStreamTextIn and ObjectStreamBinaryIn).
class JPH_EXPORT ObjectStreamIn : public IObjectStreamIn
{
private:
	struct ClassDescription;

public:
	/// Main function to read an object from a stream
	template <class T>
	static bool sReadObject(istream &inStream, T *&outObject)
	{
		// Create the input stream
		bool result = false;
		ObjectStreamIn *stream = ObjectStreamIn::Open(inStream);
		if (stream)
		{
			// Read the object
			outObject = (T *)stream->Read(JPH_RTTI(T));
			result = (outObject != nullptr);
			delete stream;
		}
		return result;
	}

	/// Main function to read an object from a stream (reference counting pointer version)
	template <class T>
	static bool sReadObject(istream &inStream, Ref<T> &outObject)
	{
		T *object = nullptr;
		bool result = sReadObject(inStream, object);
		outObject = object;
		return result;
	}

	/// Main function to read an object from a file
	template <class T>
	static bool sReadObject(const char *inFileName, T *&outObject)
	{
		std::ifstream stream;
		stream.open(inFileName, std::ifstream::in | std::ifstream::binary);
		if (!stream.is_open())
			return false;
		return sReadObject(stream, outObject);
	}

	/// Main function to read an object from a file (reference counting pointer version)
	template <class T>
	static bool sReadObject(const char *inFileName, Ref<T> &outObject)
	{
		T *object = nullptr;
		bool result = sReadObject(inFileName, object);
		outObject = object;
		return result;
	}

	//////////////////////////////////////////////////////
	// EVERYTHING BELOW THIS SHOULD NOT DIRECTLY BE CALLED
	//////////////////////////////////////////////////////

	///@name Serialization operations
	void *						Read(const RTTI *inRTTI);
	void *						ReadObject(const RTTI *& outRTTI);
	bool						ReadRTTI();
	virtual bool				ReadClassData(const char *inClassName, void *inInstance) override;
	bool						ReadClassData(const ClassDescription &inClassDesc, void *inInstance);
	virtual bool				ReadPointerData(const RTTI *inRTTI, void **inPointer, int inRefCountOffset = -1) override;
	bool						SkipAttributeData(int inArrayDepth, EOSDataType inDataType, const char *inClassName);

protected:
	/// Constructor
	explicit					ObjectStreamIn(istream &inStream);

	/// Determine the type and version of an object stream
	static bool					GetInfo(istream &inStream, EStreamType &outType, int &outVersion, int &outRevision);

	/// Static constructor
	static ObjectStreamIn *		Open(istream &inStream);

	istream &					mStream;

private:
	/// Class descriptions
	struct AttributeDescription
	{
		int						mArrayDepth = 0;
		EOSDataType				mSourceType = EOSDataType::Invalid;
		EOSDataType				mDestinationType = EOSDataType::Invalid;
		String					mClassName;
		int						mIndex = -1;
	};

	struct ClassDescription
	{
								ClassDescription() = default;
		explicit				ClassDescription(const RTTI *inRTTI)					: mRTTI(inRTTI) { }

		const RTTI *			mRTTI = nullptr;
		Array<AttributeDescription>	mAttributes;
	};

	struct ObjectInfo
	{
								ObjectInfo() = default;
								ObjectInfo(void *inInstance, const RTTI *inRTTI)		: mInstance(inInstance), mRTTI(inRTTI) { }

		void *					mInstance = nullptr;
		const RTTI *			mRTTI = nullptr;
	};

	struct Link
	{
		void **					mPointer;
		int						mRefCountOffset;
		Identifier				mIdentifier;
		const RTTI *			mRTTI;
	};

	using IdentifierMap = UnorderedMap<Identifier, ObjectInfo>;
	using ClassDescriptionMap = UnorderedMap<String, ClassDescription>;

	ClassDescriptionMap			mClassDescriptionMap;
	IdentifierMap				mIdentifierMap;											///< Links identifier to an object pointer
	Array<Link>					mUnresolvedLinks;										///< All pointers (links) are resolved after reading the entire file, e.g. when all object exist
};

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
