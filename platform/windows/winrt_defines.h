/**************************************************************************/
/*  winrt_defines.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/typedefs.h"

#include <activation.h>
#include <inspectable.h>
#include <shlwapi.h>
#include <winstring.h>
#include <wrl/client.h>

using namespace Microsoft::WRL;

class HStringWrapper : public RefCounted {
	HSTRING string = nullptr;

public:
	String get_string() const {
		String ret;
		if (!string) {
			return ret;
		}
		uint32_t size = 0;
		PCWSTR utf16 = WindowsGetStringRawBuffer(string, &size);
		if (utf16 && size > 0) {
			ret = String::utf16((const char16_t *)utf16, size);
		}

		return ret;
	}

	void set_string(const String &p_string) {
		if (string) {
			WindowsDeleteString(string);
			string = nullptr;
		}
		Char16String cs = p_string.utf16();
		HRESULT res = WindowsCreateString((PCNZWCH)cs.get_data(), cs.length(), &string);
		if (FAILED(res)) {
			string = nullptr;
			ERR_PRINT(vformat("Failed to create HSTRING with error 0x%08ux.", (uint64_t)res));
		}
	}

	HSTRING get_ptr() { return string; }
	HSTRING *get_ptrw() { return &string; }

	HStringWrapper() {}
	HStringWrapper(const String p_string) { set_string(p_string); }

	~HStringWrapper() {
		if (string) {
			WindowsDeleteString(string);
			string = nullptr;
		}
	}
};

// Common code for ROTypedEventHandler implementations.
#define TYPED_EVENT_HANDLER_CLASS(m_this, m_iface) \
	LONG ref_count = 1; \
\
public: \
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID p_riid, void **p_ppv) { \
		static const QITAB qit[] = { \
			{ &__uuidof(m_iface), static_cast<decltype(qit[0].dwOffset)>(OFFSETOFCLASS(m_iface, m_this)) }, \
			{ nullptr, 0 }, \
		}; \
		return QISearch(this, qit, p_riid, p_ppv); \
	} \
	ULONG STDMETHODCALLTYPE AddRef() { \
		return InterlockedIncrement(&ref_count); \
	} \
	ULONG STDMETHODCALLTYPE Release() { \
		long ref = InterlockedDecrement(&ref_count); \
		if (!ref) { \
			delete this; \
		} \
		return ref; \
	} \
	virtual ~m_this() {}

struct ROEventToken {
	int64_t value = 0;
	operator bool() const {
		return value != 0;
	}
	void reset() {
		value = 0;
	}
};
enum class ROAdvancedColorKind : int32_t {
	StandardDynamicRange = 0,
	WideColorGamut = 1,
	HighDynamicRange = 2,
};

enum class ROAsyncStatus : int32_t {
	Canceled = 2,
	Completed = 1,
	Error = 3,
	Started = 0,
};

enum class ROCoreInputViewKind : int32_t {
	Default = 0,
	Keyboard = 1,
	Handwriting = 2,
	Emoji = 3,
	Symbols = 4,
	Clipboard = 5,
	Dictation = 6,
	Gamepad = 7,
};

enum class ROTimedMetadataKind : int32_t {
	Caption = 0,
	Chapter = 1,
	Custom = 2,
	Data = 3,
	Description = 4,
	Subtitle = 5,
	ImageSubtitle = 6,
	Speech = 7,
};

enum class RoTimedMetadataTrackPresentationMode : int32_t {
	Disabled = 0,
	Hidden = 1,
	ApplicationPresented = 2,
	PlatformPresented = 3,
};

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wctor-dtor-privacy")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wstrict-aliasing")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")

// clang-format off

MIDL_INTERFACE("997439FE-F681-4A11-B416-C13A47E8BA36")
ROApiInformationStatics : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE IsTypePresent(void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsMethodPresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsMethodPresentWithArity(void *, void *, uint32_t, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsEventPresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsPropertyPresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsReadOnlyPropertyPresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsWriteablePropertyPresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsEnumNamedValuePresent(void *, void *, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsApiContractPresentByMajor(void *, uint16_t, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsApiContractPresentByMajorAndMinor(void *, uint16_t, uint16_t, bool *) = 0;
};

MIDL_INTERFACE("7449121C-382B-4705-8DA7-A795BA482013")
RODisplayInformationStaticsInterop : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetForWindow(HWND, REFIID, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetForMonitor(HMONITOR, REFIID, void **) = 0;
};

MIDL_INTERFACE("22F34E66-50DB-4E36-A98D-61C01B384D20")
RODispatcherQueueController : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_DispatcherQueue(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE ShutdownQueueAsync(void **) = 0;
};

MIDL_INTERFACE("548CEFBD-BC8A-5FA0-8DF2-957440FC8BF4")
ROReference_1_Int32 : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Value(int32_t *) = 0;
};

MIDL_INTERFACE("5A648006-843A-4DA9-865B-9D26E5DFAD7B")
ROAsyncAction : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE put_Completed(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Completed(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetResults() = 0;
};

MIDL_INTERFACE("00000036-0000-0000-C000-000000000046")
ROAsyncInfo : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Id(uint32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Status(int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_ErrorCode(HRESULT *) = 0;
	virtual HRESULT STDMETHODCALLTYPE Cancel() = 0;
	virtual HRESULT STDMETHODCALLTYPE Close() = 0;
};

MIDL_INTERFACE("9FC2B0BB-E446-44E2-AA61-9CAB8F636AF2")
ROAsyncOperation : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE put_Completed(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Completed(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetResults(void **) = 0;
};

MIDL_INTERFACE("3A5442DC-2CDE-4A8D-80D1-21DC5ADCC1AA")
RODisplayInformation5 : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetAdvancedColorInfo(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_AdvancedColorInfoChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_AdvancedColorInfoChanged(ROEventToken) = 0;
};

MIDL_INTERFACE("8797DCFB-B229-4081-AE9A-2CC85E34AD6A")
ROAdvancedColorInfo : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_CurrentAdvancedColorKind(int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_RedPrimary(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_GreenPrimary(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_BluePrimary(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_WhitePoint(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_MaxLuminanceInNits(float *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_MinLuminanceInNits(float *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_MaxAverageFullFrameLuminanceInNits(float *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_SdrWhiteLevelInNits(float *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsHdrMetadataFormatCurrentlySupported(int32_t, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IsAdvancedColorKindAvailable(int32_t, bool *) = 0;
};

MIDL_INTERFACE("9DE1C534-6AE1-11E0-84E1-18A905BCC53F")
ROTypedEventHandler : public IUnknown {
public:
	virtual HRESULT STDMETHODCALLTYPE Invoke(void *, IInspectable *) = 0;
};

MIDL_INTERFACE("2F13C006-A03A-5F69-B090-75A43E33423E")
ROVectorView_HSTRING : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetAt(uint32_t, HSTRING *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Size(uint32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IndexOf(HSTRING, uint32_t *, BOOLEAN *) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetMany(uint32_t, uint32_t, HSTRING *, uint32_t *) = 0;
};

MIDL_INTERFACE("A6487363-B074-5C60-AB16-866DCE4EE54D")
ROVectorView_IInspectable : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetAt(uint32_t, IInspectable **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Size(uint32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IndexOf(IInspectable *, uint32_t *, BOOLEAN *) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetMany(uint32_t, uint32_t, IInspectable **, uint32_t *) = 0;
};

MIDL_INTERFACE("01BF4326-ED37-4E96-B0E9-C1340D1EA158")
ROGlobalizationPreferencesStatics : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Calendars(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Clocks(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Currencies(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Languages(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_HomeGeographicRegion(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_WeekStartsOn(int32_t *) = 0;
};

MIDL_INTERFACE("7D9B97CD-EDBE-49CF-A54F-337DE052907F")
ROCoreInputViewStatics : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetForCurrentView(void **) = 0;
};

MIDL_INTERFACE("BC941653-3AB9-4849-8F58-46E7F0353CFC")
ROCoreInputView3 : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE TryShow(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE TryShowWithKind(int32_t, bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE TryHide(bool *) = 0;
};

MIDL_INTERFACE("7D526ECC-7533-4C3F-85BE-888C2BAEEBDC")
ROInstalledVoicesStatic : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_AllVoices(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_DefaultVoice(void **) = 0;
};

MIDL_INTERFACE("B127D6A4-1291-4604-AA9C-83134083352C")
ROVoiceInformation : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_DisplayName(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Id(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Language(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Description(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Gender(int32_t *) = 0;
};

MIDL_INTERFACE("CE9F7C76-97F4-4CED-AD68-D51C458E45C6")
ROSpeechSynthesizer : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE SynthesizeTextToStreamAsync(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE SynthesizeSsmlToStreamAsync(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Voice(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Voice(void **) = 0;
};

MIDL_INTERFACE("A7C5ECB2-4339-4D6A-BBF8-C7A4F1544C2E")
ROSpeechSynthesizer2 : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Options(void **) = 0;
};

MIDL_INTERFACE("A0E23871-CC3D-43C9-91B1-EE185324D83D")
ROSpeechSynthesizerOptions : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_IncludeWordBoundaryMetadata(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_IncludeWordBoundaryMetadata(bool) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_IncludeSentenceBoundaryMetadata(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_IncludeSentenceBoundaryMetadata(bool) = 0;
};

MIDL_INTERFACE("1CBEF60E-119C-4BED-B118-D250C3A25793")
ROSpeechSynthesizerOptions2 : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_AudioVolume(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_AudioVolume(double) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_SpeakingRate(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_SpeakingRate(double) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_AudioPitch(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_AudioPitch(double) = 0;
};

MIDL_INTERFACE("905A0FE1-BC53-11DF-8C49-001E4FC686DA")
RORandomAccessStream : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Size(uint64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Size(uint64_t) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetInputStreamAt(uint64_t, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetOutputStreamAt(uint64_t, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Position(uint64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE Seek(uint64_t) = 0;
	virtual HRESULT STDMETHODCALLTYPE CloneStream(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_CanRead(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_CanWrite(bool *) = 0;
};

MIDL_INTERFACE("83E46E93-244C-4622-BA0B-6229C4D0D65D")
ROSpeechSynthesisStream : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Markers(void **) = 0;
};

MIDL_INTERFACE("381A83CB-6FFF-499B-8D64-2885DFC1249E")
ROMediaPlayer : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_AutoPlay(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_AutoPlay(bool) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_NaturalDuration(int64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Position(int64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Position(int64_t) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_BufferingProgress(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_CurrentState(int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_CanSeek(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_CanPause(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_IsLoopingEnabled(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_IsLoopingEnabled(bool) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_IsProtected(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_IsMuted(bool *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_IsMuted(bool) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_PlaybackRate(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_PlaybackRate(double) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Volume(double *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Volume(double) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_PlaybackMediaMarkers(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_MediaOpened(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_MediaOpened(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_MediaEnded(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_MediaEnded(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_MediaFailed(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_MediaFailed(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_CurrentStateChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_CurrentStateChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_PlaybackMediaMarkerReached(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_PlaybackMediaMarkerReached(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_MediaPlayerRateChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_MediaPlayerRateChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_VolumeChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_VolumeChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_SeekCompleted(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_SeekCompleted(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_BufferingStarted(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_BufferingStarted(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_BufferingEnded(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_BufferingEnded(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE Play() = 0;
	virtual HRESULT STDMETHODCALLTYPE Pause() = 0;
	virtual HRESULT STDMETHODCALLTYPE SetUriSource(void *) = 0;
};

MIDL_INTERFACE("E7BFB599-A09D-4C21-BCDF-20AF4F86B3D9")
ROMediaSource : public IInspectable {
};

MIDL_INTERFACE("82449B9F-7322-4C0B-B03B-3E69A48260C5")
ROMediaPlayerSource2 : public IInspectable {
	virtual HRESULT STDMETHODCALLTYPE get_Source(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Source(void *) = 0;
};

MIDL_INTERFACE("F77D6FA4-4652-410E-B1D8-E9A5E245A45C")
ROMediaSourceStatics : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE CreateFromAdaptiveMediaSource(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromMediaStreamSource(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromMseStreamSource(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromIMediaSource(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromStorageFile(void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromStream(void *, void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromStreamReference(void *, void *, void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE CreateFromUri(void *, void **) = 0;
};

MIDL_INTERFACE("047097D2-E4AF-48AB-B283-6929E674ECE2")
ROMediaPlaybackItem : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE add_AudioTracksChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_AudioTracksChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_VideoTracksChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_VideoTracksChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_TimedMetadataTracksChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_TimedMetadataTracksChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Source(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_AudioTracks(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_VideoTracks(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_TimedMetadataTracks(void **) = 0;
};

MIDL_INTERFACE("7133FCE1-1769-4FF9-A7C1-38D2C4D42360")
ROMediaPlaybackItemFactory : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE Create(void *, void **) = 0;
};

MIDL_INTERFACE("72B41319-BBFB-46A3-9372-9C9C744B9438")
ROMediaPlaybackTimedMetadataTrackList : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE add_PresentationModeChanged(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_PresentationModeChanged(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetPresentationMode(uint32_t, int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE SetPresentationMode(uint32_t, int32_t) = 0;
};

MIDL_INTERFACE("9E6AED9E-F67A-49A9-B330-CF03B0E9CF07")
ROTimedMetadataTrack : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE add_CueEntered(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_CueEntered(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_CueExited(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_CueExited(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE add_TrackFailed(void *, ROEventToken *) = 0;
	virtual HRESULT STDMETHODCALLTYPE remove_TrackFailed(ROEventToken) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Cues(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_ActiveCues(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_TimedMetadataKind(int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_DispatchType(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE AddCue(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE RemoveCue(void *) = 0;
};

MIDL_INTERFACE("0313AE7A-2803-5D45-B5A1-A0FC5CD55E7C")
ROVectorView_TimedMetadataTrack : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetAt(uint32_t, ROTimedMetadataTrack **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Size(uint32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE IndexOf(ROTimedMetadataTrack *, uint32_t *, BOOLEAN *) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetMany(uint32_t, uint32_t, ROTimedMetadataTrack **, uint32_t *) = 0;
};

MIDL_INTERFACE("578CD1B9-90E2-4E60-ABC4-8740B01F6196")
ROPlaybackMediaMarkerReachedEventArgs : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_PlaybackMediaMarker(void **) = 0;
};

MIDL_INTERFACE("C4D22F5C-3C1C-4444-B6B9-778B0422D41A")
ROPlaybackMediaMarker : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Time(int64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_MediaMarkerType(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Text(void **) = 0;
};

MIDL_INTERFACE("D12F47F7-5FA4-4E68-9FE5-32160DCEE57E")
ROMediaCueEventArgs : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Cue(void **) = 0;
};

MIDL_INTERFACE("2744E9B9-A7E3-4F16-BAC4-7914EBC08301")
ROMediaPlayerFailedEventArgs : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Error(int32_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_ExtendedErrorCode(HRESULT *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_ErrorMessage(void **) = 0;
};

MIDL_INTERFACE("362C45A7-3A0A-5E27-99CE-CFF6D1B770E1")
ROTypedEventHandler_MediaPlayer_MediaPlayerFailedEventArgs : public IUnknown {
public:
	virtual HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *, ROMediaPlayerFailedEventArgs *) = 0;
};

MIDL_INTERFACE("F1A6A51E-D078-5C40-BA3F-348870BA5C87")
ROTypedEventHandler_MediaPlayer_IInspectable : public IUnknown {
public:
	virtual HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *, IInspectable *) = 0;
};

MIDL_INTERFACE("67A4F43C-C254-57F0-A39D-A475A342D21D")
ROTypedEventHandler_MediaPlayer_PlaybackMediaMarkerReachedEventArgs : public IUnknown {
public:
	virtual HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *, ROPlaybackMediaMarkerReachedEventArgs *) = 0;
};

MIDL_INTERFACE("4AAC9411-C355-5C95-8C78-5A0F5CA1A54D")
ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs : public IUnknown {
public:
	virtual HRESULT STDMETHODCALLTYPE Invoke(ROTimedMetadataTrack *, ROMediaCueEventArgs *) = 0;
};

MIDL_INTERFACE("C7D15E5D-59DC-431F-A0EE-27744323B36D")
ROMediaCue : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE put_StartTime(int64_t) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_StartTime(int64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Duration(int64_t) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Duration(int64_t *) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Id(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_Id(void **) = 0;
};

MIDL_INTERFACE("AEE254DC-1725-4BAD-8043-A98499B017A2")
ROSpeechCue : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_Text(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_Text(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_StartPositionInInput(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_StartPositionInInput(void *) = 0;
	virtual HRESULT STDMETHODCALLTYPE get_EndPositionInInput(void **) = 0;
	virtual HRESULT STDMETHODCALLTYPE put_EndPositionInInput(void *) = 0;
};

MIDL_INTERFACE("97D098A5-3B99-4DE9-88A5-E11D2F50C795")
ROContentTypeProvider : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE get_ContentType(void **) = 0;
};

// clang-format on

#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ROApiInformationStatics, 0x997439FE, 0xF681, 0x4A11, 0xB4, 0x16, 0xC1, 0x3A, 0x47, 0xE8, 0xBA, 0x36)
__CRT_UUID_DECL(RODisplayInformationStaticsInterop, 0x7449121C, 0x382B, 0x4705, 0x8D, 0xA7, 0xA7, 0x95, 0xBA, 0x48, 0x20, 0x13)
__CRT_UUID_DECL(RODispatcherQueueController, 0x22F34E66, 0x50DB, 0x4E36, 0xA9, 0x8D, 0x61, 0xC0, 0x1B, 0x38, 0x4D, 0x20)
__CRT_UUID_DECL(ROReference_1_Int32, 0x548CEFBD, 0xBC8A, 0x5FA0, 0x8D, 0xF2, 0x95, 0x74, 0x40, 0xFC, 0x8B, 0xF4)
__CRT_UUID_DECL(ROAsyncOperation, 0x9FC2B0BB, 0xE446, 0x44E2, 0xAA, 0x61, 0x9C, 0xAB, 0x8F, 0x63, 0x6A, 0xF2)
__CRT_UUID_DECL(ROAsyncAction, 0x5A648006, 0x843A, 0x4DA9, 0x86, 0x5B, 0x9D, 0x26, 0xE5, 0xDF, 0xAD, 0x7B)
__CRT_UUID_DECL(ROAsyncInfo, 0x00000036, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46)
__CRT_UUID_DECL(RODisplayInformation5, 0x3A5442DC, 0x2CDE, 0x4A8D, 0x80, 0xD1, 0x21, 0xDC, 0x5A, 0xDC, 0xC1, 0xAA)
__CRT_UUID_DECL(ROAdvancedColorInfo, 0x8797DCFB, 0xB229, 0x4081, 0xAE, 0x9A, 0x2C, 0xC8, 0x5E, 0x34, 0xAD, 0x6A)
__CRT_UUID_DECL(ROTypedEventHandler, 0x9DE1C534, 0x6AE1, 0x11E0, 0x84, 0xE1, 0x18, 0xA9, 0x05, 0xBC, 0xC5, 0x3F)
__CRT_UUID_DECL(ROTypedEventHandler_MediaPlayer_MediaPlayerFailedEventArgs, 0x362C45A7, 0x3A0A, 0x5E27, 0x99, 0xCE, 0xCF, 0xF6, 0xD1, 0xB7, 0x70, 0xE1)
__CRT_UUID_DECL(ROTypedEventHandler_MediaPlayer_IInspectable, 0xF1A6A51E, 0xD078, 0x5C40, 0xBA, 0x3F, 0x34, 0x88, 0x70, 0xBA, 0x5C, 0x87)
__CRT_UUID_DECL(ROTypedEventHandler_MediaPlayer_PlaybackMediaMarkerReachedEventArgs, 0x67A4F43C, 0xC254, 0x57F0, 0xA3, 0x9D, 0xA4, 0x75, 0xA3, 0x42, 0xD2, 0x1D)
__CRT_UUID_DECL(ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs, 0x4AAC9411, 0xC355, 0x5C95, 0x8C, 0x78, 0x5A, 0x0F, 0x5C, 0xA1, 0xA5, 0x4D)
__CRT_UUID_DECL(ROVectorView_HSTRING, 0x2F13C006, 0xA03A, 0x5F69, 0xB0, 0x90, 0x75, 0xA4, 0x3E, 0x33, 0x42, 0x3E)
__CRT_UUID_DECL(ROVectorView_IInspectable, 0xA6487363, 0xB074, 0x5C60, 0xAB, 0x16, 0x86, 0x6D, 0xCE, 0x4E, 0xE5, 0x4D)
__CRT_UUID_DECL(ROVectorView_TimedMetadataTrack, 0x0313AE7A, 0x2803, 0x5D45, 0xB5, 0xA1, 0xA0, 0xFC, 0x5C, 0xD5, 0x5E, 0x7C)
__CRT_UUID_DECL(ROGlobalizationPreferencesStatics, 0x01BF4326, 0xED37, 0x4E96, 0xB0, 0xE9, 0xC1, 0x34, 0x0D, 0x1E, 0xA1, 0x58)
__CRT_UUID_DECL(ROCoreInputViewStatics, 0x7D9B97CD, 0xEDBE, 0x49CF, 0xA5, 0x4F, 0x33, 0x7D, 0xE0, 0x52, 0x90, 0x7F)
__CRT_UUID_DECL(ROCoreInputView3, 0xBC941653, 0x3AB9, 0x4849, 0x8F, 0x58, 0x46, 0xE7, 0xF0, 0x35, 0x3C, 0xFC)
__CRT_UUID_DECL(ROInstalledVoicesStatic, 0x7D526ECC, 0x7533, 0x4C3F, 0x85, 0xBE, 0x88, 0x8C, 0x2B, 0xAE, 0xEB, 0xDC)
__CRT_UUID_DECL(ROVoiceInformation, 0xB127D6A4, 0x1291, 0x4604, 0xAA, 0x9C, 0x83, 0x13, 0x40, 0x83, 0x35, 0x2C)
__CRT_UUID_DECL(ROSpeechSynthesizer, 0xCE9F7C76, 0x97F4, 0x4CED, 0xAD, 0x68, 0xD5, 0x1C, 0x45, 0x8E, 0x45, 0xC6)
__CRT_UUID_DECL(ROSpeechSynthesizer2, 0xA7C5ECB2, 0x4339, 0x4D6A, 0xBB, 0xF8, 0xC7, 0xA4, 0xF1, 0x54, 0x4C, 0x2E)
__CRT_UUID_DECL(ROSpeechSynthesizerOptions, 0xA0E23871, 0xCC3D, 0x43C9, 0x91, 0xB1, 0xEE, 0x18, 0x53, 0x24, 0xD8, 0x3D)
__CRT_UUID_DECL(ROSpeechSynthesizerOptions2, 0x1CBEF60E, 0x119C, 0x4BED, 0xB1, 0x18, 0xD2, 0x50, 0xC3, 0xA2, 0x57, 0x93)
__CRT_UUID_DECL(ROSpeechSynthesisStream, 0x83E46E93, 0x244C, 0x4622, 0xBA, 0x0B, 0x62, 0x29, 0xC4, 0xD0, 0xD6, 0x5D)
__CRT_UUID_DECL(RORandomAccessStream, 0x905A0FE1, 0xBC53, 0x11DF, 0x8C, 0x49, 0x00, 0x1E, 0x4F, 0xC6, 0x86, 0xDA)
__CRT_UUID_DECL(ROMediaPlayer, 0x381A83CB, 0x6FFF, 0x499B, 0x8D, 0x64, 0x28, 0x85, 0xDF, 0xC1, 0x24, 0x9E)
__CRT_UUID_DECL(ROMediaSource, 0xE7BFB599, 0xA09D, 0x4C21, 0xBC, 0xDF, 0x20, 0xAF, 0x4F, 0x86, 0xB3, 0xD9)
__CRT_UUID_DECL(ROMediaPlayerSource2, 0x82449B9F, 0x7322, 0x4C0B, 0xB0, 0x3B, 0x3E, 0x69, 0xA4, 0x82, 0x60, 0xC5)
__CRT_UUID_DECL(ROMediaSourceStatics, 0xF77D6FA4, 0x4652, 0x410E, 0xB1, 0xD8, 0xE9, 0xA5, 0xE2, 0x45, 0xA4, 0x5C)
__CRT_UUID_DECL(ROMediaPlaybackItem, 0x047097D2, 0xE4AF, 0x48AB, 0xB2, 0x83, 0x69, 0x29, 0xE6, 0x74, 0xEC, 0xE2)
__CRT_UUID_DECL(ROMediaPlaybackItemFactory, 0x7133FCE1, 0x1769, 0x4FF9, 0xA7, 0xC1, 0x38, 0xD2, 0xC4, 0xD4, 0x23, 0x60)
__CRT_UUID_DECL(ROMediaPlaybackTimedMetadataTrackList, 0x72B41319, 0xBBFB, 0x46A3, 0x93, 0x72, 0x9C, 0x9C, 0x74, 0x4B, 0x94, 0x38)
__CRT_UUID_DECL(ROTimedMetadataTrack, 0x9E6AED9E, 0xF67A, 0x49A9, 0xB3, 0x30, 0xCF, 0x03, 0xB0, 0xE9, 0xCF, 0x07)
__CRT_UUID_DECL(ROPlaybackMediaMarkerReachedEventArgs, 0x578CD1B9, 0x90E2, 0x4E60, 0xAB, 0xC4, 0x87, 0x40, 0xB0, 0x1F, 0x61, 0x96)
__CRT_UUID_DECL(ROPlaybackMediaMarker, 0xC4D22F5C, 0x3C1C, 0x4444, 0xB6, 0xB9, 0x77, 0x8B, 0x04, 0x22, 0xD4, 0x1A)
__CRT_UUID_DECL(ROMediaCueEventArgs, 0xD12F47F7, 0x5FA4, 0x4E68, 0x9F, 0xE5, 0x32, 0x16, 0x0D, 0xCE, 0xE5, 0x7E)
__CRT_UUID_DECL(ROMediaCue, 0xC7D15E5D, 0x59DC, 0x431F, 0xA0, 0xEE, 0x27, 0x74, 0x43, 0x23, 0xB3, 0x6D)
__CRT_UUID_DECL(ROSpeechCue, 0xAEE254DC, 0x1725, 0x4BAD, 0x80, 0x43, 0xA9, 0x84, 0x99, 0xB0, 0x17, 0xA2)
__CRT_UUID_DECL(ROMediaPlayerFailedEventArgs, 0x2744E9B9, 0xA7E3, 0x4F16, 0xBA, 0xC4, 0x79, 0x14, 0xEB, 0xC0, 0x83, 0x01)
__CRT_UUID_DECL(ROContentTypeProvider, 0x97D098A5, 0x3B99, 0x4DE9, 0x88, 0xA5, 0xE1, 0x1D, 0x2F, 0x50, 0xC7, 0x95)
#endif // __CRT_UUID_DECL

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP
