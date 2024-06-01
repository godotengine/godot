// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/ObjectStream/ObjectStream.h>
#include <Jolt/Core/RTTI.h>
#include <Jolt/Core/UnorderedMap.h>
#include <Jolt/Core/UnorderedSet.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <queue>
#include <fstream>
JPH_SUPPRESS_WARNINGS_STD_END

#ifdef JPH_OBJECT_STREAM

JPH_NAMESPACE_BEGIN

template <class T> using Queue = std::queue<T, std::deque<T, STLAllocator<T>>>;

/// ObjectStreamOut contains all logic for writing an object to disk. It is the base
/// class for the text and binary output streams (ObjectStreamTextOut and ObjectStreamBinaryOut).
class JPH_EXPORT ObjectStreamOut : public IObjectStreamOut
{
private:
	struct ObjectInfo;

public:
	/// Main function to write an object to a stream
	template <class T>
	static bool	sWriteObject(ostream &inStream, ObjectStream::EStreamType inType, const T &inObject)
	{
		// Create the output stream
		bool result = false;
		ObjectStreamOut *stream = ObjectStreamOut::Open(inType, inStream);
		if (stream)
		{
			// Write the object to the stream
			result = stream->Write((void *)&inObject, GetRTTI(&inObject));
			delete stream;
		}

		return result;
	}

	/// Main function to write an object to a file
	template <class T>
	static bool	sWriteObject(const char *inFileName, ObjectStream::EStreamType inType, const T &inObject)
	{
		std::ofstream stream;
		stream.open(inFileName, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!stream.is_open())
			return false;
		return sWriteObject(stream, inType, inObject);
	}

	//////////////////////////////////////////////////////
	// EVERYTHING BELOW THIS SHOULD NOT DIRECTLY BE CALLED
	//////////////////////////////////////////////////////

	///@name Serialization operations
	bool						Write(const void *inObject, const RTTI *inRTTI);
	void						WriteObject(const void *inObject);
	void						QueueRTTI(const RTTI *inRTTI);
	void						WriteRTTI(const RTTI *inRTTI);
	virtual void				WriteClassData(const RTTI *inRTTI, const void *inInstance) override;
	virtual void				WritePointerData(const RTTI *inRTTI, const void *inPointer) override;

protected:
	/// Static constructor
	static ObjectStreamOut *	Open(EStreamType inType, ostream &inStream);

	/// Constructor
	explicit 					ObjectStreamOut(ostream &inStream);

	ostream &					mStream;

private:
	struct ObjectInfo
	{
								ObjectInfo()												: mIdentifier(0), mRTTI(nullptr) { }
								ObjectInfo(Identifier inIdentifier, const RTTI *inRTTI)		: mIdentifier(inIdentifier), mRTTI(inRTTI) { }

		Identifier				mIdentifier;
		const RTTI *			mRTTI;
	};

	using IdentifierMap = UnorderedMap<const void *, ObjectInfo>;
	using ClassSet = UnorderedSet<const RTTI *>;
	using ObjectQueue = Queue<const void *>;
	using ClassQueue = Queue<const RTTI *>;

	Identifier					mNextIdentifier = sNullIdentifier + 1;						///< Next free identifier for this stream
	IdentifierMap				mIdentifierMap;												///< Links object pointer to an identifier
	ObjectQueue					mObjectQueue;												///< Queue of objects to be written
	ClassSet					mClassSet;													///< List of classes already written
	ClassQueue					mClassQueue;												///< List of classes waiting to be written
};

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
