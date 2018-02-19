/*
 * SampleFormatMediaSoundation.cpp
 * -------------------------------
 * Purpose: MediaFoundation sample import.
 * Notes  :
 * Authors: Joern Heusipp
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif
#include "../common/misc_util.h"
#include "Tagging.h"
#include "Loaders.h"
#include "ChunkReader.h"
#include "modsmp_ctrl.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"
#include "../soundlib/ModSampleCopy.h"
#include "../common/ComponentManager.h"
#if defined(MPT_WITH_MEDIAFOUNDATION)
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <Propvarutil.h>
#endif // MPT_WITH_MEDIAFOUNDATION


OPENMPT_NAMESPACE_BEGIN


#if defined(MPT_WITH_MEDIAFOUNDATION)

template <typename T>
static void mptMFSafeRelease(T **ppT)
{
	if(*ppT)
	{
		(*ppT)->Release();
		*ppT = NULL;
	}
}

#define MPT_MF_CHECKED(x) MPT_DO { \
	HRESULT hr = (x); \
	if(!SUCCEEDED(hr)) \
	{ \
		goto fail; \
	} \
} MPT_WHILE_0

// Implementing IMFByteStream is apparently not enough to stream raw bytes
// data to MediaFoundation.
// Additionally, one has to also implement a custom IMFAsyncResult for the
// BeginRead/EndRead interface which allows transferring the number of read
// bytes around.
// To make things even worse, MediaFoundation fails to detect some AAC and MPEG
// files if a non-file-based or read-only stream is used for opening.
// The only sane option which remains if we do not have an on-disk filename
// available:
//  1 - write a temporary file
//  2 - close it
//  3 - open it using MediaFoundation.
// We use FILE_ATTRIBUTE_TEMPORARY which will try to keep the file data in
// memory just like regular allocated memory and reduce the overhead basically
// to memcpy.

static FileTags ReadMFMetadata(IMFMediaSource *mediaSource)
{

	FileTags tags;

	IMFPresentationDescriptor *presentationDescriptor = NULL;
	DWORD streams = 0;
	IMFMetadataProvider *metadataProvider = NULL;
	IMFMetadata *metadata = NULL;
	PROPVARIANT varPropNames;
	PropVariantInit(&varPropNames);

	MPT_MF_CHECKED(mediaSource->CreatePresentationDescriptor(&presentationDescriptor));
	MPT_MF_CHECKED(presentationDescriptor->GetStreamDescriptorCount(&streams));
	MPT_MF_CHECKED(MFGetService(mediaSource, MF_METADATA_PROVIDER_SERVICE, IID_IMFMetadataProvider, (void**)&metadataProvider));
	MPT_MF_CHECKED(metadataProvider->GetMFMetadata(presentationDescriptor, 0, 0, &metadata));

	MPT_MF_CHECKED(metadata->GetAllPropertyNames(&varPropNames));
	for(DWORD propIndex = 0; propIndex < varPropNames.calpwstr.cElems; ++propIndex)
	{

		PROPVARIANT propVal;
		PropVariantInit(&propVal);

		LPWSTR propName = varPropNames.calpwstr.pElems[propIndex];
		if(S_OK != metadata->GetProperty(propName, &propVal))
		{
			PropVariantClear(&propVal);
			break;
		}

		std::wstring stringVal;
		std::vector<WCHAR> wcharVal(256);
#if !MPT_OS_WINDOWS_WINRT
		// WTF, no PropVariantToString() in WinRT 
		for(;;)
		{
			HRESULT hrToString = PropVariantToString(propVal, wcharVal.data(), mpt::saturate_cast<UINT>(wcharVal.size()));
			if(hrToString == S_OK)
			{
				stringVal = wcharVal.data();
				break;
			} else if(hrToString == ERROR_INSUFFICIENT_BUFFER)
			{
				wcharVal.resize(wcharVal.size() * 2);
			} else
			{
				break;
			}
		}
#endif // !MPT_OS_WINDOWS_WINRT

		PropVariantClear(&propVal);

		if(stringVal.length() > 0)
		{
			if(propName == std::wstring(L"Author")) tags.artist = mpt::ToUnicode(stringVal);
			if(propName == std::wstring(L"Title")) tags.title = mpt::ToUnicode(stringVal);
			if(propName == std::wstring(L"WM/AlbumTitle")) tags.album = mpt::ToUnicode(stringVal);
			if(propName == std::wstring(L"WM/Track")) tags.trackno = mpt::ToUnicode(stringVal);
			if(propName == std::wstring(L"WM/Year")) tags.year = mpt::ToUnicode(stringVal);
			if(propName == std::wstring(L"WM/Genre")) tags.genre = mpt::ToUnicode(stringVal);
		}
	}

fail:

	PropVariantClear(&varPropNames);
	mptMFSafeRelease(&metadata);
	mptMFSafeRelease(&metadataProvider);
	mptMFSafeRelease(&presentationDescriptor);

	return tags;

}


class ComponentMediaFoundation : public ComponentLibrary
{
	MPT_DECLARE_COMPONENT_MEMBERS
public:
	ComponentMediaFoundation()
		: ComponentLibrary(ComponentTypeSystem)
	{
		return;
	}
	virtual bool DoInitialize()
	{
		if(!mpt::Windows::Version::Current().IsAtLeast(mpt::Windows::Version::Win7))
		{
			return false;
		}
#if !MPT_OS_WINDOWS_WINRT
		if(!(true
			&& AddLibrary("mf", mpt::LibraryPath::System(MPT_PATHSTRING("mf")))
			&& AddLibrary("mfplat", mpt::LibraryPath::System(MPT_PATHSTRING("mfplat")))
			&& AddLibrary("mfreadwrite", mpt::LibraryPath::System(MPT_PATHSTRING("mfreadwrite")))
			&& AddLibrary("propsys", mpt::LibraryPath::System(MPT_PATHSTRING("propsys")))
			))
		{
			return false;
		}
#endif // !MPT_OS_WINDOWS_WINRT
		if(!SUCCEEDED(MFStartup(MF_VERSION)))
		{
			return false;
		}
		return true;
	}
	virtual ~ComponentMediaFoundation()
	{
		if(IsAvailable())
		{
			MFShutdown();
		}
	}
};
MPT_REGISTERED_COMPONENT(ComponentMediaFoundation, "MediaFoundation")

#endif // MPT_WITH_MEDIAFOUNDATION


#ifdef MODPLUG_TRACKER
std::vector<FileType> CSoundFile::GetMediaFoundationFileTypes()
{
	std::vector<FileType> result;

#if defined(MPT_WITH_MEDIAFOUNDATION)

	ComponentHandle<ComponentMediaFoundation> mf;
	if(!IsComponentAvailable(mf))
	{
		return result;
	}

	std::map<std::wstring, FileType> guidMap;

	HKEY hkHandlers = NULL;
	LSTATUS regResult = RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows Media Foundation\\ByteStreamHandlers", 0, KEY_READ, &hkHandlers);
	if(regResult != ERROR_SUCCESS)
	{
		return result;
	}

	for(DWORD handlerIndex = 0; ; ++handlerIndex)
	{

		WCHAR handlerTypeBuf[256];
		MemsetZero(handlerTypeBuf);
		regResult = RegEnumKeyW(hkHandlers, handlerIndex, handlerTypeBuf, 256);
		if(regResult != ERROR_SUCCESS)
		{
			break;
		}

		std::wstring handlerType = handlerTypeBuf;

		if(handlerType.length() < 1)
		{
			continue;
		}

		HKEY hkHandler = NULL;
		regResult = RegOpenKeyExW(hkHandlers, handlerTypeBuf, 0, KEY_READ, &hkHandler);
		if(regResult != ERROR_SUCCESS)
		{
			continue;
		}

		for(DWORD valueIndex = 0; ; ++valueIndex)
		{
			WCHAR valueNameBuf[16384];
			MemsetZero(valueNameBuf);
			DWORD valueNameBufLen = 16384;
			DWORD valueType = 0;
			BYTE valueData[16384];
			MemsetZero(valueData);
			DWORD valueDataLen = 16384;
			regResult = RegEnumValueW(hkHandler, valueIndex, valueNameBuf, &valueNameBufLen, NULL, &valueType, valueData, &valueDataLen);
			if(regResult != ERROR_SUCCESS)
			{
				break;
			}
			if(valueNameBufLen <= 0 || valueType != REG_SZ || valueDataLen <= 0)
			{
				continue;
			}

			std::wstring guid = std::wstring(valueNameBuf);

			mpt::ustring description = mpt::ToUnicode(std::wstring(reinterpret_cast<WCHAR*>(valueData)));
			description = mpt::String::Replace(description, MPT_USTRING("Byte Stream Handler"), MPT_USTRING("Files"));
			description = mpt::String::Replace(description, MPT_USTRING("ByteStreamHandler"), MPT_USTRING("Files"));

			guidMap[guid]
				.ShortName(MPT_USTRING("mf"))
				.Description(description)
				;

			if(handlerType[0] == L'.')
			{
				guidMap[guid].AddExtension(mpt::PathString::FromWide(handlerType.substr(1)));
			} else
			{
				guidMap[guid].AddMimeType(mpt::ToCharset(mpt::CharsetASCII, handlerType));
			}

		}

		RegCloseKey(hkHandler);
		hkHandler = NULL;

	}

	RegCloseKey(hkHandlers);
	hkHandlers = NULL;

	for(const auto &it : guidMap)
	{
		result.push_back(it.second);
	}

#endif // MPT_WITH_MEDIAFOUNDATION

	return result;
}
#endif // MODPLUG_TRACKER


bool CSoundFile::ReadMediaFoundationSample(SAMPLEINDEX sample, FileReader &file, bool mo3Decode)
{

#if !defined(MPT_WITH_MEDIAFOUNDATION)

	MPT_UNREFERENCED_PARAMETER(sample);
	MPT_UNREFERENCED_PARAMETER(file);
	MPT_UNREFERENCED_PARAMETER(mo3Decode);
	return false;

#else

	ComponentHandle<ComponentMediaFoundation> mf;
	if(!IsComponentAvailable(mf))
	{
		return false;
	}

	file.Rewind();
	// When using MF to decode MP3 samples in MO3 files, we need the mp3 file extension
	// for some of them or otherwise MF refuses to recognize them.
	mpt::PathString tmpfileExtension = (mo3Decode ? MPT_PATHSTRING("mp3") : MPT_PATHSTRING("tmp"));
	OnDiskFileWrapper diskfile(file, tmpfileExtension);
	if(!diskfile.IsValid())
	{
		return false;
	}

	bool result = false;

	std::vector<char> rawData;
	FileTags tags;
	std::string sampleName;

	IMFSourceResolver *sourceResolver = NULL;
	MF_OBJECT_TYPE objectType = MF_OBJECT_INVALID;
	IUnknown *unknownMediaSource = NULL;
	IMFMediaSource *mediaSource = NULL;
	IMFSourceReader *sourceReader = NULL;
	IMFMediaType *uncompressedAudioType = NULL;
	IMFMediaType *partialType = NULL;
	UINT32 numChannels = 0;
	UINT32 samplesPerSecond = 0;
	UINT32 bitsPerSample = 0;

	IMFSample *mfSample = NULL;
	DWORD mfSampleFlags = 0;
	IMFMediaBuffer *buffer = NULL;

	SmpLength length = 0;

	MPT_MF_CHECKED(MFCreateSourceResolver(&sourceResolver));
	MPT_MF_CHECKED(sourceResolver->CreateObjectFromURL(diskfile.GetFilename().AsNative().c_str(), MF_RESOLUTION_MEDIASOURCE | MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE | MF_RESOLUTION_READ, NULL, &objectType, &unknownMediaSource));
	if(objectType != MF_OBJECT_MEDIASOURCE) goto fail;
	MPT_MF_CHECKED(unknownMediaSource->QueryInterface(&mediaSource));

	tags = ReadMFMetadata(mediaSource);

	MPT_MF_CHECKED(MFCreateSourceReaderFromMediaSource(mediaSource, NULL, &sourceReader));
	MPT_MF_CHECKED(MFCreateMediaType(&partialType));
	MPT_MF_CHECKED(partialType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio));
	MPT_MF_CHECKED(partialType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM));
	MPT_MF_CHECKED(sourceReader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, NULL, partialType));
	MPT_MF_CHECKED(sourceReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &uncompressedAudioType));
	MPT_MF_CHECKED(sourceReader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE));
	MPT_MF_CHECKED(uncompressedAudioType->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &numChannels));
	MPT_MF_CHECKED(uncompressedAudioType->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &samplesPerSecond));
	MPT_MF_CHECKED(uncompressedAudioType->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bitsPerSample));
	if(numChannels <= 0 || numChannels > 2) goto fail;
	if(samplesPerSecond <= 0) goto fail;
	if(bitsPerSample != 8 && bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32) goto fail;

	for(;;)
	{
		mfSampleFlags = 0;
		MPT_MF_CHECKED(sourceReader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, NULL, &mfSampleFlags, NULL, &mfSample));
		if(mfSampleFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
		{
			break;
		}
		if(mfSampleFlags & MF_SOURCE_READERF_ENDOFSTREAM)
		{
			break;
		}
		MPT_MF_CHECKED(mfSample->ConvertToContiguousBuffer(&buffer));
		{
			BYTE *data = NULL;
			DWORD dataSize = 0;
			MPT_MF_CHECKED(buffer->Lock(&data, NULL, &dataSize));
			rawData.insert(rawData.end(), mpt::byte_cast<char*>(data), mpt::byte_cast<char*>(data + dataSize));
			MPT_MF_CHECKED(buffer->Unlock());
		}
		mptMFSafeRelease(&buffer);
		mptMFSafeRelease(&mfSample);
	}

	mptMFSafeRelease(&uncompressedAudioType);
	mptMFSafeRelease(&partialType);
	mptMFSafeRelease(&sourceReader);

	sampleName = mpt::ToCharset(GetCharsetInternal(), GetSampleNameFromTags(tags));

	if(rawData.size() / numChannels / (bitsPerSample / 8) > MAX_SAMPLE_LENGTH)
	{
		result = false;
		goto fail;
	}

	length = static_cast<SmpLength>(rawData.size() / numChannels / (bitsPerSample/8));

	DestroySampleThreadsafe(sample);
	if(!mo3Decode)
	{
		mpt::String::Copy(m_szNames[sample], sampleName);
		Samples[sample].Initialize();
		Samples[sample].nC5Speed = samplesPerSecond;
	}
	Samples[sample].nLength = length;
	Samples[sample].uFlags.set(CHN_16BIT, bitsPerSample >= 16);
	Samples[sample].uFlags.set(CHN_STEREO, numChannels == 2);
	Samples[sample].AllocateSample();
	if(Samples[sample].pSample == nullptr)
	{
		result = false;
		goto fail;
	}

	if(bitsPerSample == 24)
	{
		if(numChannels == 2)
		{
			CopyStereoInterleavedSample<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt24<0, littleEndian24> > >(Samples[sample], rawData.data(), rawData.size());
		} else
		{
			CopyMonoSample<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt24<0, littleEndian24> > >(Samples[sample], rawData.data(), rawData.size());
		}
	} else if(bitsPerSample == 32)
	{
		if(numChannels == 2)
		{
			CopyStereoInterleavedSample<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt32<0, littleEndian32> > >(Samples[sample], rawData.data(), rawData.size());
		} else
		{
			CopyMonoSample<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt32<0, littleEndian32> > >(Samples[sample], rawData.data(), rawData.size());
		}
	} else
	{
		// just copy
		std::copy(rawData.data(), rawData.data() + rawData.size(), mpt::void_cast<char*>(Samples[sample].pSample));
	}

	result = true;

fail:

	mptMFSafeRelease(&buffer);
	mptMFSafeRelease(&mfSample);
	mptMFSafeRelease(&uncompressedAudioType);
	mptMFSafeRelease(&partialType);
	mptMFSafeRelease(&sourceReader);
	mptMFSafeRelease(&mediaSource);
	mptMFSafeRelease(&unknownMediaSource);
	mptMFSafeRelease(&sourceResolver);

	return result;

#endif

}


bool CSoundFile::CanReadMediaFoundation()
{
	bool result = false;
	#if defined(MPT_WITH_MEDIAFOUNDATION)
		if(!result)
		{
			ComponentHandle<ComponentMediaFoundation> mf;
			if(IsComponentAvailable(mf))
			{
				result = true;
			}
		}
	#endif
	return result;
}


OPENMPT_NAMESPACE_END
