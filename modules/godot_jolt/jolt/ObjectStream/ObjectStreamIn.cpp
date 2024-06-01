// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#ifdef JPH_OBJECT_STREAM

#include <Jolt/ObjectStream/ObjectStreamIn.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/UnorderedSet.h>
#include <Jolt/ObjectStream/ObjectStreamTextIn.h>
#include <Jolt/ObjectStream/ObjectStreamBinaryIn.h>
#include <Jolt/ObjectStream/SerializableObject.h>

JPH_NAMESPACE_BEGIN

ObjectStreamIn::ObjectStreamIn(istream &inStream) :
	mStream(inStream)
{
}

bool ObjectStreamIn::GetInfo(istream &inStream, EStreamType &outType, int &outVersion, int &outRevision)
{
	// Read header and check if it is the correct format, e.g. "TOS 1.00"
	char header[9];
	memset(header, 0, 9);
	inStream.read(header, 8);
	if ((header[0] == 'B' || header[0] == 'T') && header[1] == 'O' && header[2] == 'S'
		&& (header[3] == ' ' || isdigit(header[3])) && isdigit(header[4])
		&& header[5] == '.' && isdigit(header[6]) && isdigit(header[7]))
	{
		// Check if this is a binary or text objectfile
		switch (header[0])
		{
		case 'T':	outType = ObjectStream::EStreamType::Text;		break;
		case 'B':	outType = ObjectStream::EStreamType::Binary;	break;
		default:	JPH_ASSERT(false);								break;
		}

		// Extract version and revision
		header[5] = '\0';
		outVersion = atoi(&header[3]);
		outRevision = atoi(&header[6]);

		return true;
	}

	Trace("ObjectStreamIn: Not a valid object stream.");
	return false;
}

ObjectStreamIn *ObjectStreamIn::Open(istream &inStream)
{
	// Check if file is an ObjectStream of the correct version and revision
	EStreamType	type;
	int version;
	int revision;
	if (GetInfo(inStream, type, version, revision))
	{
		if (version == sVersion && revision == sRevision)
		{
			// Create an input stream of the correct type
			switch (type)
			{
			case EStreamType::Text:		return new ObjectStreamTextIn(inStream);
			case EStreamType::Binary:	return new ObjectStreamBinaryIn(inStream);
			default:					JPH_ASSERT(false);
			}
		}
		else
		{
			Trace("ObjectStreamIn: Different version stream (%d.%02d, expected %d.%02d).", version, revision, sVersion, sRevision);
		}
	}

	return nullptr;
}

void *ObjectStreamIn::Read(const RTTI *inRTTI)
{
	using ObjectSet = UnorderedSet<void *>;

	// Read all information on the stream
	void *main_object = nullptr;
	bool continue_reading = true;
	for (;;)
	{
		// Get type of next operation
		EOSDataType data_type;
		if (!ReadDataType(data_type))
			break;

		if (data_type == EOSDataType::Declare)
		{
			// Read type declaration
			if (!ReadRTTI())
			{
				Trace("ObjectStreamIn: Fatal error while reading class description for class %s.", inRTTI->GetName());
				continue_reading = false;
				break;
			}
		}
		else if (data_type == EOSDataType::Object)
		{
			const RTTI *rtti;
			void *object = ReadObject(rtti);
			if (!main_object && object)
			{
				// This is the first and thus main object of the file.
				if (rtti->IsKindOf(inRTTI))
				{
					// Object is of correct type
					main_object = object;
				}
				else
				{
					Trace("ObjectStreamIn: Main object of different type. Expected %s, but found %s.", inRTTI->GetName(), rtti->GetName());
					continue_reading = false;
					break;
				}
			}
		}
		else
		{
			// Invalid or out of place token found
			Trace("ObjectStreamIn: Invalid or out of place token found.");
			continue_reading = false;
			break;
		}
	}

	// Resolve links (pointer, references)
	if (continue_reading)
	{
		// Resolve links
		ObjectSet referenced_objects;
		for (Link &link : mUnresolvedLinks)
		{
			IdentifierMap::const_iterator j = mIdentifierMap.find(link.mIdentifier);
			if (j != mIdentifierMap.end() && j->second.mRTTI->IsKindOf(link.mRTTI))
			{
				const ObjectInfo &obj_info = j->second;

				// Set pointer
				*link.mPointer = obj_info.mInstance;

				// Increment refcount if it was a referencing pointer
				if (link.mRefCountOffset != -1)
					++(*(uint32 *)(((uint8 *)obj_info.mInstance) + link.mRefCountOffset));

				// Add referenced object to the list
				if (referenced_objects.find(obj_info.mInstance) == referenced_objects.end())
					referenced_objects.insert(obj_info.mInstance);
			}
			else
			{
				// Referenced object not found, set pointer to nullptr
				Trace("ObjectStreamIn: Setting incorrect pointer to class of type %s to nullptr.", link.mRTTI->GetName());
				*link.mPointer = nullptr;
			}
		}

		// Release unreferenced objects except the main object
		for (const IdentifierMap::value_type &j : mIdentifierMap)
		{
			const ObjectInfo &obj_info = j.second;

			if (obj_info.mInstance != main_object)
			{
				ObjectSet::const_iterator k = referenced_objects.find(obj_info.mInstance);
				if (k == referenced_objects.end())
				{
					Trace("ObjectStreamIn: Releasing unreferenced object of type %s.", obj_info.mRTTI->GetName());
					obj_info.mRTTI->DestructObject(obj_info.mInstance);
				}
			}
		}

		return main_object;
	}
	else
	{
		// Release all objects if a fatal error occurred
		for (const IdentifierMap::value_type &i : mIdentifierMap)
		{
			const ObjectInfo &obj_info = i.second;
			obj_info.mRTTI->DestructObject(obj_info.mInstance);
		}

		return nullptr;
	}
}

void *ObjectStreamIn::ReadObject(const RTTI *& outRTTI)
{
	// Read the object class
	void *object = nullptr;
	String class_name;
	if (ReadName(class_name))
	{
		// Get class description
		ClassDescriptionMap::iterator i = mClassDescriptionMap.find(class_name);
		if (i != mClassDescriptionMap.end())
		{
			const ClassDescription &class_desc = i->second;

			// Read object identifier
			Identifier identifier;
			if (ReadIdentifier(identifier))
			{
				// Check if this object can be read or must be skipped
				if (identifier != sNullIdentifier
					&& class_desc.mRTTI
					&& !class_desc.mRTTI->IsAbstract())
				{
					// Create object instance
					outRTTI = class_desc.mRTTI;
					object = outRTTI->CreateObject();

					// Read object attributes
					if (ReadClassData(class_desc, object))
					{
						// Add object to identifier map
						mIdentifierMap.try_emplace(identifier, object, outRTTI);
					}
					else
					{
						// Fatal error while reading attributes, release object
						outRTTI->DestructObject(object);
						object = nullptr;
					}
				}
				else
				{
					// Skip this object
					// TODO: This operation can fail, but there is no check yet
					Trace("ObjectStreamIn: Found uncreatable object %s.", class_name.c_str());
					ReadClassData(class_desc, nullptr);
				}
			}
		}
		else
		{
			// TODO: This is a fatal error, but this function has no way of indicating this
			Trace("ObjectStreamIn: Found object of unknown class %s.", class_name.c_str());
		}
	}

	return object;
}

bool ObjectStreamIn::ReadRTTI()
{
	// Read class name and find it's attribute info
	String class_name;
	if (!ReadName(class_name))
		return false;

	// Find class
	const RTTI *rtti = Factory::sInstance->Find(class_name.c_str());
	if (rtti == nullptr)
		Trace("ObjectStreamIn: Unknown class: \"%s\".", class_name.c_str());

	// Insert class description
	ClassDescription &class_desc = mClassDescriptionMap.try_emplace(class_name, rtti).first->second;

	// Read the number of entries in the description
	uint32 count;
	if (!ReadCount(count))
		return false;

	// Read the entries
	for (uint32 i = 0; i < count; ++i)
	{
		AttributeDescription attribute;

		// Read name
		String attribute_name;
		if (!ReadName(attribute_name))
			return false;

		// Read type
		if (!ReadDataType(attribute.mSourceType))
			return false;

		// Read array depth
		while (attribute.mSourceType == EOSDataType::Array)
		{
			++attribute.mArrayDepth;
			if (!ReadDataType(attribute.mSourceType))
				return false;
		}

		// Read instance/pointer class name
		if ((attribute.mSourceType == EOSDataType::Instance || attribute.mSourceType == EOSDataType::Pointer)
			&& !ReadName(attribute.mClassName))
			return false;

		// Find attribute in rtti
		if (rtti)
		{
			// Find attribute index
			for (int idx = 0; idx < rtti->GetAttributeCount(); ++idx)
			{
				const SerializableAttribute &attr = rtti->GetAttribute(idx);
				if (strcmp(attr.GetName(), attribute_name.c_str()) == 0)
				{
					attribute.mIndex = idx;
					break;
				}
			}

			// Check if attribute is of expected type
			if (attribute.mIndex >= 0)
			{
				const SerializableAttribute &attr = rtti->GetAttribute(attribute.mIndex);
				if (attr.IsType(attribute.mArrayDepth, attribute.mSourceType, attribute.mClassName.c_str()))
				{
					// No conversion needed
					attribute.mDestinationType = attribute.mSourceType;
				}
				else if (attribute.mArrayDepth == 0 && attribute.mClassName.empty())
				{
					// Try to apply type conversions
					if (attribute.mSourceType == EOSDataType::T_Vec3 && attr.IsType(0, EOSDataType::T_DVec3, ""))
						attribute.mDestinationType = EOSDataType::T_DVec3;
					else if (attribute.mSourceType == EOSDataType::T_DVec3 && attr.IsType(0, EOSDataType::T_Vec3, ""))
						attribute.mDestinationType = EOSDataType::T_Vec3;
					else
						attribute.mIndex = -1;
				}
				else
				{
					// No conversion exists
					attribute.mIndex = -1;
				}
			}
		}

		// Add attribute to the class description
		class_desc.mAttributes.push_back(attribute);
	}

	return true;
}

bool ObjectStreamIn::ReadClassData(const char *inClassName, void *inInstance)
{
	// Find the class description
	ClassDescriptionMap::iterator i = mClassDescriptionMap.find(inClassName);
	if (i != mClassDescriptionMap.end())
		return ReadClassData(i->second, inInstance);

	return false;
}

bool ObjectStreamIn::ReadClassData(const ClassDescription &inClassDesc, void *inInstance)
{
	// Read data for this class
	bool continue_reading = true;

	for (const AttributeDescription &attr_desc : inClassDesc.mAttributes)
	{
		// Read or skip the attribute data
		if (attr_desc.mIndex >= 0 && inInstance)
		{
			const SerializableAttribute &attr = inClassDesc.mRTTI->GetAttribute(attr_desc.mIndex);
			if (attr_desc.mSourceType ==  attr_desc.mDestinationType)
			{
				continue_reading = attr.ReadData(*this, inInstance);
			}
			else if (attr_desc.mSourceType == EOSDataType::T_Vec3 && attr_desc.mDestinationType == EOSDataType::T_DVec3)
			{
				// Vec3 to DVec3
				Vec3 tmp;
				continue_reading = ReadPrimitiveData(tmp);
				if (continue_reading)
					*attr.GetMemberPointer<DVec3>(inInstance) = DVec3(tmp);
			}
			else if (attr_desc.mSourceType == EOSDataType::T_DVec3 && attr_desc.mDestinationType == EOSDataType::T_Vec3)
			{
				// DVec3 to Vec3
				DVec3 tmp;
				continue_reading = ReadPrimitiveData(tmp);
				if (continue_reading)
					*attr.GetMemberPointer<Vec3>(inInstance) = Vec3(tmp);
			}
			else
			{
				JPH_ASSERT(false); // Unknown conversion
				continue_reading = SkipAttributeData(attr_desc.mArrayDepth, attr_desc.mSourceType, attr_desc.mClassName.c_str());
			}
		}
		else
			continue_reading = SkipAttributeData(attr_desc.mArrayDepth, attr_desc.mSourceType, attr_desc.mClassName.c_str());

		if (!continue_reading)
			break;
	}

	return continue_reading;
}

bool ObjectStreamIn::ReadPointerData(const RTTI *inRTTI, void **inPointer, int inRefCountOffset)
{
	Identifier identifier;
	if (ReadIdentifier(identifier))
	{
		if (identifier == sNullIdentifier)
		{
			// Set nullptr pointer
			inPointer = nullptr;
		}
		else
		{
			// Put pointer on the list to be resolved later on
			Link &link = mUnresolvedLinks.emplace_back();
			link.mPointer = inPointer;
			link.mRefCountOffset = inRefCountOffset;
			link.mIdentifier = identifier;
			link.mRTTI = inRTTI;
		}

		return true;
	}

	return false;
}

bool ObjectStreamIn::SkipAttributeData(int inArrayDepth, EOSDataType inDataType, const char *inClassName)
{
	bool continue_reading = true;

	// Get number of items to read
	uint32 count = 1;
	for (; inArrayDepth > 0; --inArrayDepth)
	{
		uint32 temporary;
		if (ReadCount(temporary))
		{
			// Multiply for multi dimensional arrays
			count *= temporary;
		}
		else
		{
			// Fatal error while reading array size
			continue_reading = false;
			break;
		}
	}

	// Read data for all items
	if (continue_reading)
	{
		if (inDataType == EOSDataType::Instance)
		{
			// Get the class description
			ClassDescriptionMap::iterator i = mClassDescriptionMap.find(inClassName);
			if (i != mClassDescriptionMap.end())
			{
				for (; count > 0 && continue_reading; --count)
					continue_reading = ReadClassData(i->second, nullptr);
			}
			else
			{
				continue_reading = false;
				Trace("ObjectStreamIn: Found instance of unknown class %s.", inClassName);
			}
		}
		else
		{
			for (; count > 0 && continue_reading; --count)
			{
				switch (inDataType)
				{
				case EOSDataType::Pointer:
					{
						Identifier temporary;
						continue_reading = ReadIdentifier(temporary);
						break;
					}

				case EOSDataType::T_uint8:
					{
						uint8 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_uint16:
					{
						uint16 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_int:
					{
						int temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_uint32:
					{
						uint32 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_uint64:
					{
						uint64 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_float:
					{
						float temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_double:
					{
						double temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_bool:
					{
						bool temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_String:
					{
						String temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Float3:
					{
						Float3 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Double3:
					{
						Double3 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Vec3:
					{
						Vec3 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_DVec3:
					{
						DVec3 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Vec4:
					{
						Vec4 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Quat:
					{
						Quat temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_Mat44:
					{
						Mat44 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::T_DMat44:
					{
						DMat44 temporary;
						continue_reading = ReadPrimitiveData(temporary);
						break;
					}

				case EOSDataType::Array:
				case EOSDataType::Object:
				case EOSDataType::Declare:
				case EOSDataType::Instance:
				case EOSDataType::Invalid:
				default:
					continue_reading = false;
					break;
				}
			}
		}
	}

	return continue_reading;
}

JPH_NAMESPACE_END

#endif // JPH_OBJECT_STREAM
