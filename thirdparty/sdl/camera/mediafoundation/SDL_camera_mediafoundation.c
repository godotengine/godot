/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// the Windows Media Foundation API

#ifdef SDL_CAMERA_DRIVER_MEDIAFOUNDATION

#define COBJMACROS

// this seems to be a bug in mfidl.h, just define this to avoid the problem section.
#define __IMFVideoProcessorControl3_INTERFACE_DEFINED__

#include "../../core/windows/SDL_windows.h"

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include "../SDL_syscamera.h"
#include "../SDL_camera_c.h"

static const IID SDL_IID_IMFMediaSource = { 0x279a808d, 0xaec7, 0x40c8, { 0x9c, 0x6b, 0xa6, 0xb4, 0x92, 0xc7, 0x8a, 0x66 } };
static const IID SDL_IID_IMF2DBuffer = { 0x7dc9d5f9, 0x9ed9, 0x44ec, { 0x9b, 0xbf, 0x06, 0x00, 0xbb, 0x58, 0x9f, 0xbb } };
static const IID SDL_IID_IMF2DBuffer2 = { 0x33ae5ea6, 0x4316, 0x436f, { 0x8d, 0xdd, 0xd7, 0x3d, 0x22, 0xf8, 0x29, 0xec } };
static const GUID SDL_MF_MT_DEFAULT_STRIDE = { 0x644b4e48, 0x1e02, 0x4516, { 0xb0, 0xeb, 0xc0, 0x1c, 0xa9, 0xd4, 0x9a, 0xc6 } };
static const GUID SDL_MF_MT_MAJOR_TYPE = { 0x48eba18e, 0xf8c9, 0x4687, { 0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f } };
static const GUID SDL_MF_MT_SUBTYPE = { 0xf7e34c9a, 0x42e8, 0x4714, { 0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5 } };
static const GUID SDL_MF_MT_VIDEO_NOMINAL_RANGE = { 0xc21b8ee5, 0xb956, 0x4071, { 0x8d, 0xaf, 0x32, 0x5e, 0xdf, 0x5c, 0xab, 0x11 } };
static const GUID SDL_MF_MT_VIDEO_PRIMARIES = { 0xdbfbe4d7, 0x0740, 0x4ee0, { 0x81, 0x92, 0x85, 0x0a, 0xb0, 0xe2, 0x19, 0x35 } };
static const GUID SDL_MF_MT_TRANSFER_FUNCTION = { 0x5fb0fce9, 0xbe5c, 0x4935, { 0xa8, 0x11, 0xec, 0x83, 0x8f, 0x8e, 0xed, 0x93 } };
static const GUID SDL_MF_MT_YUV_MATRIX = { 0x3e23d450, 0x2c75, 0x4d25, { 0xa0, 0x0e, 0xb9, 0x16, 0x70, 0xd1, 0x23, 0x27 } };
static const GUID SDL_MF_MT_VIDEO_CHROMA_SITING = { 0x65df2370, 0xc773, 0x4c33, { 0xaa, 0x64, 0x84, 0x3e, 0x06, 0x8e, 0xfb, 0x0c } };
static const GUID SDL_MF_MT_FRAME_SIZE = { 0x1652c33d, 0xd6b2, 0x4012, { 0xb8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xa3, 0x7d } };
static const GUID SDL_MF_MT_FRAME_RATE = { 0xc459a2e8, 0x3d2c, 0x4e44, { 0xb1, 0x32, 0xfe, 0xe5, 0x15, 0x6c, 0x7b, 0xb0 } };
static const GUID SDL_MFMediaType_Video = { 0x73646976, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71 } };
static const IID SDL_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = { 0x60d0e559, 0x52f8, 0x4fa2, { 0xbb, 0xce, 0xac, 0xdb, 0x34, 0xa8, 0xec, 0x1 } };
static const IID SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = { 0xc60ac5fe, 0x252a, 0x478f, { 0xa0, 0xef, 0xbc, 0x8f, 0xa5, 0xf7, 0xca, 0xd3 } };
static const IID SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK = { 0x58f0aad8, 0x22bf, 0x4f8a, { 0xbb, 0x3d, 0xd2, 0xc4, 0x97, 0x8c, 0x6e, 0x2f } };
static const IID SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = { 0x8ac3587a, 0x4ae7, 0x42d8, { 0x99, 0xe0, 0x0a, 0x60, 0x13, 0xee, 0xf9, 0x0f } };

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

#define SDL_DEFINE_MEDIATYPE_GUID(name, fmt) static const GUID SDL_##name = { fmt, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } }
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_RGB555, 24);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_RGB565, 23);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_RGB24, 20);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_RGB32, 22);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_ARGB32, 21);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_A2R10G10B10, 31);
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_YV12, FCC('YV12'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_IYUV, FCC('IYUV'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_YUY2, FCC('YUY2'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_UYVY, FCC('UYVY'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_YVYU, FCC('YVYU'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_NV12, FCC('NV12'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_NV21, FCC('NV21'));
SDL_DEFINE_MEDIATYPE_GUID(MFVideoFormat_MJPG, FCC('MJPG'));
#undef SDL_DEFINE_MEDIATYPE_GUID

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static const struct
{
    const GUID *guid;
    SDL_PixelFormat format;
    SDL_Colorspace colorspace;
} fmtmappings[] = {
    // This is not every possible format, just popular ones that SDL can reasonably handle.
    // (and we should probably trim this list more.)
    { &SDL_MFVideoFormat_RGB555, SDL_PIXELFORMAT_XRGB1555, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_RGB565, SDL_PIXELFORMAT_RGB565, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_RGB24, SDL_PIXELFORMAT_RGB24, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_RGB32, SDL_PIXELFORMAT_XRGB8888, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_ARGB32, SDL_PIXELFORMAT_ARGB8888, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_A2R10G10B10, SDL_PIXELFORMAT_ARGB2101010, SDL_COLORSPACE_SRGB },
    { &SDL_MFVideoFormat_YV12, SDL_PIXELFORMAT_YV12, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_IYUV, SDL_PIXELFORMAT_IYUV, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_YUY2,  SDL_PIXELFORMAT_YUY2, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_UYVY, SDL_PIXELFORMAT_UYVY, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_YVYU, SDL_PIXELFORMAT_YVYU, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_NV12, SDL_PIXELFORMAT_NV12, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_NV21, SDL_PIXELFORMAT_NV21, SDL_COLORSPACE_BT709_LIMITED },
    { &SDL_MFVideoFormat_MJPG, SDL_PIXELFORMAT_MJPG, SDL_COLORSPACE_SRGB }
};

static SDL_Colorspace GetMediaTypeColorspace(IMFMediaType *mediatype, SDL_Colorspace default_colorspace)
{
    SDL_Colorspace colorspace = default_colorspace;

    if (SDL_COLORSPACETYPE(colorspace) == SDL_COLOR_TYPE_YCBCR) {
        HRESULT ret;
        UINT32 range = 0, primaries = 0, transfer = 0, matrix = 0, chroma = 0;

        ret = IMFMediaType_GetUINT32(mediatype, &SDL_MF_MT_VIDEO_NOMINAL_RANGE, &range);
        if (SUCCEEDED(ret)) {
            switch (range) {
            case MFNominalRange_0_255:
                range = SDL_COLOR_RANGE_FULL;
                break;
            case MFNominalRange_16_235:
                range = SDL_COLOR_RANGE_LIMITED;
                break;
            default:
                range = (UINT32)SDL_COLORSPACERANGE(default_colorspace);
                break;
            }
        } else {
            range = (UINT32)SDL_COLORSPACERANGE(default_colorspace);
        }

        ret = IMFMediaType_GetUINT32(mediatype, &SDL_MF_MT_VIDEO_PRIMARIES, &primaries);
        if (SUCCEEDED(ret)) {
            switch (primaries) {
            case MFVideoPrimaries_BT709:
                primaries = SDL_COLOR_PRIMARIES_BT709;
                break;
            case MFVideoPrimaries_BT470_2_SysM:
                primaries = SDL_COLOR_PRIMARIES_BT470M;
                break;
            case MFVideoPrimaries_BT470_2_SysBG:
                primaries = SDL_COLOR_PRIMARIES_BT470BG;
                break;
            case MFVideoPrimaries_SMPTE170M:
                primaries = SDL_COLOR_PRIMARIES_BT601;
                break;
            case MFVideoPrimaries_SMPTE240M:
                primaries = SDL_COLOR_PRIMARIES_SMPTE240;
                break;
            case MFVideoPrimaries_EBU3213:
                primaries = SDL_COLOR_PRIMARIES_EBU3213;
                break;
            case MFVideoPrimaries_BT2020:
                primaries = SDL_COLOR_PRIMARIES_BT2020;
                break;
            case MFVideoPrimaries_XYZ:
                primaries = SDL_COLOR_PRIMARIES_XYZ;
                break;
            case MFVideoPrimaries_DCI_P3:
                primaries = SDL_COLOR_PRIMARIES_SMPTE432;
                break;
            default:
                primaries = (UINT32)SDL_COLORSPACEPRIMARIES(default_colorspace);
                break;
            }
        } else {
            primaries = (UINT32)SDL_COLORSPACEPRIMARIES(default_colorspace);
        }

        ret = IMFMediaType_GetUINT32(mediatype, &SDL_MF_MT_TRANSFER_FUNCTION, &transfer);
        if (SUCCEEDED(ret)) {
            switch (transfer) {
            case MFVideoTransFunc_10:
                transfer = SDL_TRANSFER_CHARACTERISTICS_LINEAR;
                break;
            case MFVideoTransFunc_22:
                transfer = SDL_TRANSFER_CHARACTERISTICS_GAMMA22;
                break;
            case MFVideoTransFunc_709:
                transfer = SDL_TRANSFER_CHARACTERISTICS_BT709;
                break;
            case MFVideoTransFunc_240M:
                transfer = SDL_TRANSFER_CHARACTERISTICS_SMPTE240;
                break;
            case MFVideoTransFunc_sRGB:
                transfer = SDL_TRANSFER_CHARACTERISTICS_SRGB;
                break;
            case MFVideoTransFunc_28:
                transfer = SDL_TRANSFER_CHARACTERISTICS_GAMMA28;
                break;
            case MFVideoTransFunc_Log_100:
                transfer = SDL_TRANSFER_CHARACTERISTICS_LOG100;
                break;
            case MFVideoTransFunc_2084:
                transfer = SDL_TRANSFER_CHARACTERISTICS_PQ;
                break;
            case MFVideoTransFunc_HLG:
                transfer = SDL_TRANSFER_CHARACTERISTICS_HLG;
                break;
            case 18 /* MFVideoTransFunc_BT1361_ECG */:
                transfer = SDL_TRANSFER_CHARACTERISTICS_BT1361;
                break;
            case 19 /* MFVideoTransFunc_SMPTE428 */:
                transfer = SDL_TRANSFER_CHARACTERISTICS_SMPTE428;
                break;
            default:
                transfer = (UINT32)SDL_COLORSPACETRANSFER(default_colorspace);
                break;
            }
        } else {
            transfer = (UINT32)SDL_COLORSPACETRANSFER(default_colorspace);
        }

        ret = IMFMediaType_GetUINT32(mediatype, &SDL_MF_MT_YUV_MATRIX, &matrix);
        if (SUCCEEDED(ret)) {
            switch (matrix) {
            case MFVideoTransferMatrix_BT709:
                matrix = SDL_MATRIX_COEFFICIENTS_BT709;
                break;
            case MFVideoTransferMatrix_BT601:
                matrix = SDL_MATRIX_COEFFICIENTS_BT601;
                break;
            case MFVideoTransferMatrix_SMPTE240M:
                matrix = SDL_MATRIX_COEFFICIENTS_SMPTE240;
                break;
            case MFVideoTransferMatrix_BT2020_10:
                matrix = SDL_MATRIX_COEFFICIENTS_BT2020_NCL;
                break;
            case 6 /* MFVideoTransferMatrix_Identity */:
                matrix = SDL_MATRIX_COEFFICIENTS_IDENTITY;
                break;
            case 7 /* MFVideoTransferMatrix_FCC47 */:
                matrix = SDL_MATRIX_COEFFICIENTS_FCC;
                break;
            case 8 /* MFVideoTransferMatrix_YCgCo */:
                matrix = SDL_MATRIX_COEFFICIENTS_YCGCO;
                break;
            case 9 /* MFVideoTransferMatrix_SMPTE2085 */:
                matrix = SDL_MATRIX_COEFFICIENTS_SMPTE2085;
                break;
            case 10 /* MFVideoTransferMatrix_Chroma */:
                matrix = SDL_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL;
                break;
            case 11 /* MFVideoTransferMatrix_Chroma_const */:
                matrix = SDL_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL;
                break;
            case 12 /* MFVideoTransferMatrix_ICtCp */:
                matrix = SDL_MATRIX_COEFFICIENTS_ICTCP;
                break;
            default:
                matrix = (UINT32)SDL_COLORSPACEMATRIX(default_colorspace);
                break;
            }
        } else {
            matrix = (UINT32)SDL_COLORSPACEMATRIX(default_colorspace);
        }

        ret = IMFMediaType_GetUINT32(mediatype, &SDL_MF_MT_VIDEO_CHROMA_SITING, &chroma);
        if (SUCCEEDED(ret)) {
            switch (chroma) {
            case MFVideoChromaSubsampling_MPEG2:
                chroma = SDL_CHROMA_LOCATION_LEFT;
                break;
            case MFVideoChromaSubsampling_MPEG1:
                chroma = SDL_CHROMA_LOCATION_CENTER;
                break;
            case MFVideoChromaSubsampling_DV_PAL:
                chroma = SDL_CHROMA_LOCATION_TOPLEFT;
                break;
            default:
                chroma = (UINT32)SDL_COLORSPACECHROMA(default_colorspace);
                break;
            }
        } else {
            chroma = (UINT32)SDL_COLORSPACECHROMA(default_colorspace);
        }

        colorspace = SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR, range, primaries, transfer, matrix, chroma);
    }
    return colorspace;
}

static void MediaTypeToSDLFmt(IMFMediaType *mediatype, SDL_PixelFormat *format, SDL_Colorspace *colorspace)
{
    HRESULT ret;
    GUID type;

    ret = IMFMediaType_GetGUID(mediatype, &SDL_MF_MT_SUBTYPE, &type);
    if (SUCCEEDED(ret)) {
        for (size_t i = 0; i < SDL_arraysize(fmtmappings); i++) {
            if (WIN_IsEqualGUID(&type, fmtmappings[i].guid)) {
                *format = fmtmappings[i].format;
                *colorspace = GetMediaTypeColorspace(mediatype, fmtmappings[i].colorspace);
                return;
            }
        }
    }
#if DEBUG_CAMERA
    SDL_Log("Unknown media type: 0x%x (%c%c%c%c)", type.Data1,
            (char)(Uint8)(type.Data1 >>  0),
            (char)(Uint8)(type.Data1 >>  8),
            (char)(Uint8)(type.Data1 >> 16),
            (char)(Uint8)(type.Data1 >> 24));
#endif
    *format = SDL_PIXELFORMAT_UNKNOWN;
    *colorspace = SDL_COLORSPACE_UNKNOWN;
}

static const GUID *SDLFmtToMFVidFmtGuid(SDL_PixelFormat format)
{
    for (size_t i = 0; i < SDL_arraysize(fmtmappings); i++) {
        if (fmtmappings[i].format == format) {
            return fmtmappings[i].guid;
        }
    }
    return NULL;
}


// handle to Media Foundation libs--Vista and later!--for access to the Media Foundation API.

// mf.dll ...
static HMODULE libmf = NULL;
typedef HRESULT(WINAPI *pfnMFEnumDeviceSources)(IMFAttributes *,IMFActivate ***,UINT32 *);
typedef HRESULT(WINAPI *pfnMFCreateDeviceSource)(IMFAttributes  *, IMFMediaSource **);
static pfnMFEnumDeviceSources pMFEnumDeviceSources = NULL;
static pfnMFCreateDeviceSource pMFCreateDeviceSource = NULL;

// mfplat.dll ...
static HMODULE libmfplat = NULL;
typedef HRESULT(WINAPI *pfnMFStartup)(ULONG, DWORD);
typedef HRESULT(WINAPI *pfnMFShutdown)(void);
typedef HRESULT(WINAPI *pfnMFCreateAttributes)(IMFAttributes **, UINT32);
typedef HRESULT(WINAPI *pfnMFCreateMediaType)(IMFMediaType **);
typedef HRESULT(WINAPI *pfnMFGetStrideForBitmapInfoHeader)(DWORD, DWORD, LONG *);

static pfnMFStartup pMFStartup = NULL;
static pfnMFShutdown pMFShutdown = NULL;
static pfnMFCreateAttributes pMFCreateAttributes = NULL;
static pfnMFCreateMediaType pMFCreateMediaType = NULL;
static pfnMFGetStrideForBitmapInfoHeader pMFGetStrideForBitmapInfoHeader = NULL;

// mfreadwrite.dll ...
static HMODULE libmfreadwrite = NULL;
typedef HRESULT(WINAPI *pfnMFCreateSourceReaderFromMediaSource)(IMFMediaSource *, IMFAttributes *, IMFSourceReader **);
static pfnMFCreateSourceReaderFromMediaSource pMFCreateSourceReaderFromMediaSource = NULL;


typedef struct SDL_PrivateCameraData
{
    IMFSourceReader *srcreader;
    IMFSample *current_sample;
    int pitch;
} SDL_PrivateCameraData;

static bool MEDIAFOUNDATION_WaitDevice(SDL_Camera *device)
{
    SDL_assert(device->hidden->current_sample == NULL);

    IMFSourceReader *srcreader = device->hidden->srcreader;
    IMFSample *sample = NULL;

    while (!SDL_GetAtomicInt(&device->shutdown)) {
        DWORD stream_flags = 0;
        const HRESULT ret = IMFSourceReader_ReadSample(srcreader, (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, &stream_flags, NULL, &sample);
        if (FAILED(ret)) {
            return false;   // ruh roh.
        }

        // we currently ignore stream_flags format changes, but my _hope_ is that IMFSourceReader is handling this and
        // will continue to give us the explicitly-specified format we requested when opening the device, though, and
        // we don't have to manually deal with it.

        if (sample != NULL) {
            break;
        } else if (stream_flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM)) {
            return false;  // apparently this camera has gone down.  :/
        }

        // otherwise, there was some minor burp, probably; just try again.
    }

    device->hidden->current_sample = sample;

    return true;
}


#ifdef KEEP_ACQUIRED_BUFFERS_LOCKED

#define PROP_SURFACE_IMFOBJS_POINTER "SDL.camera.mediafoundation.imfobjs"

typedef struct SDL_IMFObjects
{
    IMF2DBuffer2 *buffer2d2;
    IMF2DBuffer *buffer2d;
    IMFMediaBuffer *buffer;
    IMFSample *sample;
} SDL_IMFObjects;

static void SDLCALL CleanupIMF2DBuffer2(void *userdata, void *value)
{
    SDL_IMFObjects *objs = (SDL_IMFObjects *)value;
    IMF2DBuffer2_Unlock2D(objs->buffer2d2);
    IMF2DBuffer2_Release(objs->buffer2d2);
    IMFMediaBuffer_Release(objs->buffer);
    IMFSample_Release(objs->sample);
    SDL_free(objs);
}

static void SDLCALL CleanupIMF2DBuffer(void *userdata, void *value)
{
    SDL_IMFObjects *objs = (SDL_IMFObjects *)value;
    IMF2DBuffer_Unlock2D(objs->buffer2d);
    IMF2DBuffer_Release(objs->buffer2d);
    IMFMediaBuffer_Release(objs->buffer);
    IMFSample_Release(objs->sample);
    SDL_free(objs);
}

static void SDLCALL CleanupIMFMediaBuffer(void *userdata, void *value)
{
    SDL_IMFObjects *objs = (SDL_IMFObjects *)value;
    IMFMediaBuffer_Unlock(objs->buffer);
    IMFMediaBuffer_Release(objs->buffer);
    IMFSample_Release(objs->sample);
    SDL_free(objs);
}

static SDL_CameraFrameResult MEDIAFOUNDATION_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    SDL_assert(device->hidden->current_sample != NULL);

    SDL_CameraFrameResult result = SDL_CAMERA_FRAME_READY;
    HRESULT ret;
    LONGLONG timestamp100NS = 0;
    SDL_IMFObjects *objs = (SDL_IMFObjects *) SDL_calloc(1, sizeof (SDL_IMFObjects));

    if (objs == NULL) {
        return SDL_CAMERA_FRAME_ERROR;
    }

    objs->sample = device->hidden->current_sample;
    device->hidden->current_sample = NULL;

    const SDL_PropertiesID surfprops = SDL_GetSurfaceProperties(frame);
    if (!surfprops) {
        result = SDL_CAMERA_FRAME_ERROR;
    } else {
        ret = IMFSample_GetSampleTime(objs->sample, &timestamp100NS);
        if (FAILED(ret)) {
            result = SDL_CAMERA_FRAME_ERROR;
        }

        *timestampNS = timestamp100NS * 100;  // the timestamps are in 100-nanosecond increments; move to full nanoseconds.
    }

    ret = (result == SDL_CAMERA_FRAME_ERROR) ? E_FAIL : IMFSample_ConvertToContiguousBuffer(objs->sample, &objs->buffer); // IMFSample_GetBufferByIndex(objs->sample, 0, &objs->buffer);

    if (FAILED(ret)) {
        SDL_free(objs);
        result = SDL_CAMERA_FRAME_ERROR;
    } else {
        BYTE *pixels = NULL;
        LONG pitch = 0;
        DWORD buflen = 0;

        if (SUCCEEDED(IMFMediaBuffer_QueryInterface(objs->buffer, &SDL_IID_IMF2DBuffer2, (void **)&objs->buffer2d2))) {
            BYTE *bufstart = NULL;
            ret = IMF2DBuffer2_Lock2DSize(objs->buffer2d2, MF2DBuffer_LockFlags_Read, &pixels, &pitch, &bufstart, &buflen);
            if (FAILED(ret)) {
                result = SDL_CAMERA_FRAME_ERROR;
                CleanupIMF2DBuffer2(NULL, objs);
            } else {
                if (frame->format == SDL_PIXELFORMAT_MJPG) {
                    pitch = (LONG)buflen;
                }
                if (pitch < 0) { // image rows are reversed.
                    pixels += -pitch * (frame->h - 1);
                }
                frame->pixels = pixels;
                frame->pitch = (int)pitch;
                if (!SDL_SetPointerPropertyWithCleanup(surfprops, PROP_SURFACE_IMFOBJS_POINTER, objs, CleanupIMF2DBuffer2, NULL)) {
                    result = SDL_CAMERA_FRAME_ERROR;
                }
            }
        } else if (frame->format != SDL_PIXELFORMAT_MJPG &&
                   SUCCEEDED(IMFMediaBuffer_QueryInterface(objs->buffer, &SDL_IID_IMF2DBuffer, (void **)&objs->buffer2d))) {
            ret = IMF2DBuffer_Lock2D(objs->buffer2d, &pixels, &pitch);
            if (FAILED(ret)) {
                CleanupIMF2DBuffer(NULL, objs);
                result = SDL_CAMERA_FRAME_ERROR;
            } else {
                if (pitch < 0) { // image rows are reversed.
                    pixels += -pitch * (frame->h - 1);
                }
                frame->pixels = pixels;
                frame->pitch = (int)pitch;
                if (!SDL_SetPointerPropertyWithCleanup(surfprops, PROP_SURFACE_IMFOBJS_POINTER, objs, CleanupIMF2DBuffer, NULL)) {
                    result = SDL_CAMERA_FRAME_ERROR;
                }
            }
        } else {
            DWORD maxlen = 0;
            ret = IMFMediaBuffer_Lock(objs->buffer, &pixels, &maxlen, &buflen);
            if (FAILED(ret)) {
                CleanupIMFMediaBuffer(NULL, objs);
                result = SDL_CAMERA_FRAME_ERROR;
            } else {
                if (frame->format == SDL_PIXELFORMAT_MJPG) {
                    pitch = (LONG)buflen;
                } else {
                    pitch = (LONG)device->hidden->pitch;
                }
                if (pitch < 0) { // image rows are reversed.
                    pixels += -pitch * (frame->h - 1);
                }
                frame->pixels = pixels;
                frame->pitch = (int)pitch;
                if (!SDL_SetPointerPropertyWithCleanup(surfprops, PROP_SURFACE_IMFOBJS_POINTER, objs, CleanupIMFMediaBuffer, NULL)) {
                    result = SDL_CAMERA_FRAME_ERROR;
                }
            }
        }
    }

    if (result != SDL_CAMERA_FRAME_READY) {
        *timestampNS = 0;
    }

    return result;
}

static void MEDIAFOUNDATION_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    const SDL_PropertiesID surfprops = SDL_GetSurfaceProperties(frame);
    if (surfprops) {
        // this will release the IMFBuffer and IMFSample objects for this frame.
        SDL_ClearProperty(surfprops, PROP_SURFACE_IMFOBJS_POINTER);
    }
}

#else

static SDL_CameraFrameResult MEDIAFOUNDATION_CopyFrame(SDL_Surface *frame, const BYTE *pixels, LONG pitch, DWORD buflen)
{
    frame->pixels = SDL_aligned_alloc(SDL_GetSIMDAlignment(), buflen);
    if (!frame->pixels) {
        return SDL_CAMERA_FRAME_ERROR;
    }

    const BYTE *start = pixels;
    if (pitch < 0) { // image rows are reversed.
        start += -pitch * (frame->h - 1);
    }
    SDL_memcpy(frame->pixels, start, buflen);
    frame->pitch = (int)pitch;

    return SDL_CAMERA_FRAME_READY;
}

static SDL_CameraFrameResult MEDIAFOUNDATION_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    SDL_assert(device->hidden->current_sample != NULL);

    SDL_CameraFrameResult result = SDL_CAMERA_FRAME_READY;
    HRESULT ret;
    LONGLONG timestamp100NS = 0;

    IMFSample *sample = device->hidden->current_sample;
    device->hidden->current_sample = NULL;

    const SDL_PropertiesID surfprops = SDL_GetSurfaceProperties(frame);
    if (!surfprops) {
        result = SDL_CAMERA_FRAME_ERROR;
    } else {
        ret = IMFSample_GetSampleTime(sample, &timestamp100NS);
        if (FAILED(ret)) {
            result = SDL_CAMERA_FRAME_ERROR;
        }

        *timestampNS = timestamp100NS * 100; // the timestamps are in 100-nanosecond increments; move to full nanoseconds.
    }

    IMFMediaBuffer *buffer = NULL;
    ret = (result < 0) ? E_FAIL : IMFSample_ConvertToContiguousBuffer(sample, &buffer); // IMFSample_GetBufferByIndex(sample, 0, &buffer);

    if (FAILED(ret)) {
        result = SDL_CAMERA_FRAME_ERROR;
    } else {
        IMF2DBuffer *buffer2d = NULL;
        IMF2DBuffer2 *buffer2d2 = NULL;
        BYTE *pixels = NULL;
        LONG pitch = 0;
        DWORD buflen = 0;

        if (SUCCEEDED(IMFMediaBuffer_QueryInterface(buffer, &SDL_IID_IMF2DBuffer2, (void **)&buffer2d2))) {
            BYTE *bufstart = NULL;
            ret = IMF2DBuffer2_Lock2DSize(buffer2d2, MF2DBuffer_LockFlags_Read, &pixels, &pitch, &bufstart, &buflen);
            if (FAILED(ret)) {
                result = SDL_CAMERA_FRAME_ERROR;
            } else {
                if (frame->format == SDL_PIXELFORMAT_MJPG) {
                    pitch = (LONG)buflen;
                }
                result = MEDIAFOUNDATION_CopyFrame(frame, pixels, pitch, buflen);
                IMF2DBuffer2_Unlock2D(buffer2d2);
            }
            IMF2DBuffer2_Release(buffer2d2);
        } else if (frame->format != SDL_PIXELFORMAT_MJPG &&
                   SUCCEEDED(IMFMediaBuffer_QueryInterface(buffer, &SDL_IID_IMF2DBuffer, (void **)&buffer2d))) {
            ret = IMF2DBuffer_Lock2D(buffer2d, &pixels, &pitch);
            if (FAILED(ret)) {
                result = SDL_CAMERA_FRAME_ERROR;
            } else {
                buflen = SDL_abs((int)pitch) * frame->h;
                result = MEDIAFOUNDATION_CopyFrame(frame, pixels, pitch, buflen);
                IMF2DBuffer_Unlock2D(buffer2d);
            }
            IMF2DBuffer_Release(buffer2d);
        } else {
            DWORD maxlen = 0;
            ret = IMFMediaBuffer_Lock(buffer, &pixels, &maxlen, &buflen);
            if (FAILED(ret)) {
                result = SDL_CAMERA_FRAME_ERROR;
            } else {
                if (frame->format == SDL_PIXELFORMAT_MJPG) {
                    pitch = (LONG)buflen;
                } else {
                    pitch = (LONG)device->hidden->pitch;
                }
                result = MEDIAFOUNDATION_CopyFrame(frame, pixels, pitch, buflen);
                IMFMediaBuffer_Unlock(buffer);
            }
        }
        IMFMediaBuffer_Release(buffer);
    }

    IMFSample_Release(sample);

    if (result != SDL_CAMERA_FRAME_READY) {
        *timestampNS = 0;
    }

    return result;
}

static void MEDIAFOUNDATION_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    SDL_aligned_free(frame->pixels);
}

#endif

static void MEDIAFOUNDATION_CloseDevice(SDL_Camera *device)
{
    if (device && device->hidden) {
        if (device->hidden->srcreader) {
            IMFSourceReader_Release(device->hidden->srcreader);
        }
        if (device->hidden->current_sample) {
            IMFSample_Release(device->hidden->current_sample);
        }
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

// this function is from https://learn.microsoft.com/en-us/windows/win32/medfound/uncompressed-video-buffers
static HRESULT GetDefaultStride(IMFMediaType *pType, LONG *plStride)
{
    LONG lStride = 0;

    // Try to get the default stride from the media type.
    HRESULT ret = IMFMediaType_GetUINT32(pType, &SDL_MF_MT_DEFAULT_STRIDE, (UINT32*)&lStride);
    if (FAILED(ret)) {
        // Attribute not set. Try to calculate the default stride.

        GUID subtype = GUID_NULL;
        UINT32 width = 0;
        // UINT32 height = 0;
        UINT64 val = 0;

        // Get the subtype and the image size.
        ret = IMFMediaType_GetGUID(pType, &SDL_MF_MT_SUBTYPE, &subtype);
        if (FAILED(ret)) {
            goto done;
        }

        ret = IMFMediaType_GetUINT64(pType, &SDL_MF_MT_FRAME_SIZE, &val);
        if (FAILED(ret)) {
            goto done;
        }

        width = (UINT32) (val >> 32);
        // height = (UINT32) val;

        ret = pMFGetStrideForBitmapInfoHeader(subtype.Data1, width, &lStride);
        if (FAILED(ret)) {
            goto done;
        }

        // Set the attribute for later reference.
        IMFMediaType_SetUINT32(pType, &SDL_MF_MT_DEFAULT_STRIDE, (UINT32) lStride);
    }

    if (SUCCEEDED(ret)) {
        *plStride = lStride;
    }

done:
    return ret;
}


static bool MEDIAFOUNDATION_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
    const char *utf8symlink = (const char *) device->handle;
    IMFAttributes *attrs = NULL;
    LPWSTR wstrsymlink = NULL;
    IMFMediaSource *source = NULL;
    IMFMediaType *mediatype = NULL;
    IMFSourceReader *srcreader = NULL;
#if 0
    DWORD num_streams = 0;
#endif
    LONG lstride = 0;
    //PROPVARIANT var;
    HRESULT ret;

    #if 0
    IMFStreamDescriptor *streamdesc = NULL;
    IMFPresentationDescriptor *presentdesc = NULL;
    IMFMediaTypeHandler *handler = NULL;
    #endif

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: opening device with symlink of '%s'", utf8symlink);
    #endif

    wstrsymlink = WIN_UTF8ToString(utf8symlink);
    if (!wstrsymlink) {
        goto failed;
    }

    #define CHECK_HRESULT(what, r) if (FAILED(r)) { WIN_SetErrorFromHRESULT(what " failed", r); goto failed; }

    ret = pMFCreateAttributes(&attrs, 1);
    CHECK_HRESULT("MFCreateAttributes", ret);

    ret = IMFAttributes_SetGUID(attrs, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    CHECK_HRESULT("IMFAttributes_SetGUID(srctype)", ret);

    ret = IMFAttributes_SetString(attrs, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, wstrsymlink);
    CHECK_HRESULT("IMFAttributes_SetString(symlink)", ret);

    ret = pMFCreateDeviceSource(attrs, &source);
    CHECK_HRESULT("MFCreateDeviceSource", ret);

    IMFAttributes_Release(attrs);
    SDL_free(wstrsymlink);
    attrs = NULL;
    wstrsymlink = NULL;

    // !!! FIXME: I think it'd be nice to do this without an IMFSourceReader,
    // since it's just utility code that has to handle more complex media streams
    // than we're dealing with, but this will do for now. The docs are slightly
    // insistent that you should use one, though...Maybe it's extremely hard
    // to handle directly at the IMFMediaSource layer...?
    ret = pMFCreateSourceReaderFromMediaSource(source, NULL, &srcreader);
    CHECK_HRESULT("MFCreateSourceReaderFromMediaSource", ret);

    // !!! FIXME: do we actually have to find the media type object in the source reader or can we just roll our own like this?
    ret = pMFCreateMediaType(&mediatype);
    CHECK_HRESULT("MFCreateMediaType", ret);

    ret = IMFMediaType_SetGUID(mediatype, &SDL_MF_MT_MAJOR_TYPE, &SDL_MFMediaType_Video);
    CHECK_HRESULT("IMFMediaType_SetGUID(major_type)", ret);

    ret = IMFMediaType_SetGUID(mediatype, &SDL_MF_MT_SUBTYPE, SDLFmtToMFVidFmtGuid(spec->format));
    CHECK_HRESULT("IMFMediaType_SetGUID(subtype)", ret);

    ret = IMFMediaType_SetUINT64(mediatype, &SDL_MF_MT_FRAME_SIZE, (((UINT64)spec->width) << 32) | ((UINT64)spec->height));
    CHECK_HRESULT("MFSetAttributeSize(frame_size)", ret);

    ret = IMFMediaType_SetUINT64(mediatype, &SDL_MF_MT_FRAME_RATE, (((UINT64)spec->framerate_numerator) << 32) | ((UINT64)spec->framerate_denominator));
    CHECK_HRESULT("MFSetAttributeRatio(frame_rate)", ret);

    ret = IMFSourceReader_SetCurrentMediaType(srcreader, (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, mediatype);
    CHECK_HRESULT("IMFSourceReader_SetCurrentMediaType", ret);

    #if 0  // this (untested thing) is what we would do to get started with a IMFMediaSource that _doesn't_ use IMFSourceReader...
    ret = IMFMediaSource_CreatePresentationDescriptor(source, &presentdesc);
    CHECK_HRESULT("IMFMediaSource_CreatePresentationDescriptor", ret);

    ret = IMFPresentationDescriptor_GetStreamDescriptorCount(presentdesc, &num_streams);
    CHECK_HRESULT("IMFPresentationDescriptor_GetStreamDescriptorCount", ret);

    for (DWORD i = 0; i < num_streams; i++) {
        BOOL selected = FALSE;
        ret = IMFPresentationDescriptor_GetStreamDescriptorByIndex(presentdesc, i, &selected, &streamdesc);
        CHECK_HRESULT("IMFPresentationDescriptor_GetStreamDescriptorByIndex", ret);

        if (selected) {
            ret = IMFStreamDescriptor_GetMediaTypeHandler(streamdesc, &handler);
            CHECK_HRESULT("IMFStreamDescriptor_GetMediaTypeHandler", ret);
            IMFMediaTypeHandler_SetCurrentMediaType(handler, mediatype);
            IMFMediaTypeHandler_Release(handler);
            handler = NULL;
        }

        IMFStreamDescriptor_Release(streamdesc);
        streamdesc = NULL;
    }

    PropVariantInit(&var);
    var.vt = VT_EMPTY;
    ret = IMFMediaSource_Start(source, presentdesc, NULL, &var);
    PropVariantClear(&var);
    CHECK_HRESULT("IMFMediaSource_Start", ret);

    IMFPresentationDescriptor_Release(presentdesc);
    presentdesc = NULL;
    #endif

    ret = GetDefaultStride(mediatype, &lstride);
    CHECK_HRESULT("GetDefaultStride", ret);

    IMFMediaType_Release(mediatype);
    mediatype = NULL;

    device->hidden = (SDL_PrivateCameraData *) SDL_calloc(1, sizeof (SDL_PrivateCameraData));
    if (!device->hidden) {
        goto failed;
    }

    device->hidden->pitch = (int) lstride;
    device->hidden->srcreader = srcreader;
    IMFMediaSource_Release(source);  // srcreader is holding a reference to this.

    // There is no user permission prompt for camera access (I think?)
    SDL_CameraPermissionOutcome(device, true);

    #undef CHECK_HRESULT

    return true;

failed:

    if (srcreader) {
        IMFSourceReader_Release(srcreader);
    }

    #if 0
    if (handler) {
        IMFMediaTypeHandler_Release(handler);
    }

    if (streamdesc) {
        IMFStreamDescriptor_Release(streamdesc);
    }

    if (presentdesc) {
        IMFPresentationDescriptor_Release(presentdesc);
    }
    #endif

    if (source) {
        IMFMediaSource_Shutdown(source);
        IMFMediaSource_Release(source);
    }

    if (mediatype) {
        IMFMediaType_Release(mediatype);
    }

    if (attrs) {
        IMFAttributes_Release(attrs);
    }
    SDL_free(wstrsymlink);

    return false;
}

static void MEDIAFOUNDATION_FreeDeviceHandle(SDL_Camera *device)
{
    if (device) {
        SDL_free(device->handle);  // the device's symlink string.
    }
}

static char *QueryActivationObjectString(IMFActivate *activation, const GUID *pguid)
{
    LPWSTR wstr = NULL;
    UINT32 wlen = 0;
    HRESULT ret = IMFActivate_GetAllocatedString(activation, pguid, &wstr, &wlen);
    if (FAILED(ret)) {
        return NULL;
    }

    char *utf8str = WIN_StringToUTF8(wstr);
    CoTaskMemFree(wstr);
    return utf8str;
}

static void GatherCameraSpecs(IMFMediaSource *source, CameraFormatAddData *add_data)
{
    HRESULT ret;

    // this has like a thousand steps.  :/

    SDL_zerop(add_data);

    IMFPresentationDescriptor *presentdesc = NULL;
    ret = IMFMediaSource_CreatePresentationDescriptor(source, &presentdesc);
    if (FAILED(ret) || !presentdesc) {
        return;
    }

    DWORD num_streams = 0;
    ret = IMFPresentationDescriptor_GetStreamDescriptorCount(presentdesc, &num_streams);
    if (FAILED(ret)) {
        num_streams = 0;
    }

    for (DWORD i = 0; i < num_streams; i++) {
        IMFStreamDescriptor *streamdesc = NULL;
        BOOL selected = FALSE;
        ret = IMFPresentationDescriptor_GetStreamDescriptorByIndex(presentdesc, i, &selected, &streamdesc);
        if (FAILED(ret) || !streamdesc) {
            continue;
        }

        if (selected) {
            IMFMediaTypeHandler *handler = NULL;
            ret = IMFStreamDescriptor_GetMediaTypeHandler(streamdesc, &handler);
            if (SUCCEEDED(ret) && handler) {
                DWORD num_mediatype = 0;
                ret = IMFMediaTypeHandler_GetMediaTypeCount(handler, &num_mediatype);
                if (FAILED(ret)) {
                    num_mediatype = 0;
                }

                for (DWORD j = 0; j < num_mediatype; j++) {
                    IMFMediaType *mediatype = NULL;
                    ret = IMFMediaTypeHandler_GetMediaTypeByIndex(handler, j, &mediatype);
                    if (SUCCEEDED(ret) && mediatype) {
                        GUID type;
                        ret = IMFMediaType_GetGUID(mediatype, &SDL_MF_MT_MAJOR_TYPE, &type);
                        if (SUCCEEDED(ret) && WIN_IsEqualGUID(&type, &SDL_MFMediaType_Video)) {
                            SDL_PixelFormat sdlfmt = SDL_PIXELFORMAT_UNKNOWN;
                            SDL_Colorspace colorspace = SDL_COLORSPACE_UNKNOWN;
                            MediaTypeToSDLFmt(mediatype, &sdlfmt, &colorspace);
                            if (sdlfmt != SDL_PIXELFORMAT_UNKNOWN) {
                                UINT64 val = 0;
                                UINT32 w = 0, h = 0;
                                ret = IMFMediaType_GetUINT64(mediatype, &SDL_MF_MT_FRAME_SIZE, &val);
                                w = (UINT32)(val >> 32);
                                h = (UINT32)val;
                                if (SUCCEEDED(ret) && w && h) {
                                    UINT32 framerate_numerator = 0, framerate_denominator = 0;
                                    ret = IMFMediaType_GetUINT64(mediatype, &SDL_MF_MT_FRAME_RATE, &val);
                                    framerate_numerator = (UINT32)(val >> 32);
                                    framerate_denominator = (UINT32)val;
                                    if (SUCCEEDED(ret) && framerate_numerator && framerate_denominator) {
                                        SDL_AddCameraFormat(add_data, sdlfmt, colorspace, (int) w, (int) h, (int)framerate_numerator, (int)framerate_denominator);
                                    }
                                }
                            }
                        }
                        IMFMediaType_Release(mediatype);
                    }
                }
                IMFMediaTypeHandler_Release(handler);
            }
        }
        IMFStreamDescriptor_Release(streamdesc);
    }

    IMFPresentationDescriptor_Release(presentdesc);
}

static bool FindMediaFoundationCameraBySymlink(SDL_Camera *device, void *userdata)
{
    return (SDL_strcmp((const char *) device->handle, (const char *) userdata) == 0);
}

static void MaybeAddDevice(IMFActivate *activation)
{
    char *symlink = QueryActivationObjectString(activation, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);

    if (SDL_FindPhysicalCameraByCallback(FindMediaFoundationCameraBySymlink, symlink)) {
        SDL_free(symlink);
        return;  // already have this one.
    }

    char *name = QueryActivationObjectString(activation, &SDL_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
    if (name && symlink) {
        IMFMediaSource *source = NULL;
        // "activating" here only creates an object, it doesn't open the actual camera hardware or start recording.
        HRESULT ret = IMFActivate_ActivateObject(activation, &SDL_IID_IMFMediaSource, (void**)&source);
        if (SUCCEEDED(ret) && source) {
            CameraFormatAddData add_data;
            GatherCameraSpecs(source, &add_data);
            if (add_data.num_specs > 0) {
                SDL_AddCamera(name, SDL_CAMERA_POSITION_UNKNOWN, add_data.num_specs, add_data.specs, symlink);
            }
            SDL_free(add_data.specs);
            IMFActivate_ShutdownObject(activation);
            IMFMediaSource_Release(source);
        }
    }

    SDL_free(name);
}

static void MEDIAFOUNDATION_DetectDevices(void)
{
    // !!! FIXME: use CM_Register_Notification (Win8+) to get device notifications.
    // !!! FIXME: Earlier versions can use RegisterDeviceNotification, but I'm not bothering: no hotplug for you!
    HRESULT ret;

    IMFAttributes *attrs = NULL;
    ret = pMFCreateAttributes(&attrs, 1);
    if (FAILED(ret)) {
        return;  // oh well, no cameras for you.
    }

    ret = IMFAttributes_SetGUID(attrs, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, &SDL_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(ret)) {
        IMFAttributes_Release(attrs);
        return;  // oh well, no cameras for you.
    }

    IMFActivate **activations = NULL;
    UINT32 total = 0;
    ret = pMFEnumDeviceSources(attrs, &activations, &total);
    IMFAttributes_Release(attrs);
    if (FAILED(ret)) {
        return;  // oh well, no cameras for you.
    }

    for (UINT32 i = 0; i < total; i++) {
        MaybeAddDevice(activations[i]);
        IMFActivate_Release(activations[i]);
    }

    CoTaskMemFree(activations);
}

static void MEDIAFOUNDATION_Deinitialize(void)
{
    pMFShutdown();

    FreeLibrary(libmfreadwrite);
    libmfreadwrite = NULL;
    FreeLibrary(libmfplat);
    libmfplat = NULL;
    FreeLibrary(libmf);
    libmf = NULL;

    pMFEnumDeviceSources = NULL;
    pMFCreateDeviceSource = NULL;
    pMFStartup = NULL;
    pMFShutdown = NULL;
    pMFCreateAttributes = NULL;
    pMFCreateMediaType = NULL;
    pMFCreateSourceReaderFromMediaSource = NULL;
    pMFGetStrideForBitmapInfoHeader = NULL;
}

static bool MEDIAFOUNDATION_Init(SDL_CameraDriverImpl *impl)
{
    // !!! FIXME: slide this off into a subroutine
    HMODULE mf = LoadLibrary(TEXT("Mf.dll")); // this library is available in Vista and later, but also can be on XP with service packs and Windows
    if (!mf) {
        return false;
    }

    HMODULE mfplat = LoadLibrary(TEXT("Mfplat.dll")); // this library is available in Vista and later. No WinXP, so have to LoadLibrary to use it for now!
    if (!mfplat) {
        FreeLibrary(mf);
        return false;
    }

    HMODULE mfreadwrite = LoadLibrary(TEXT("Mfreadwrite.dll")); // this library is available in Vista and later. No WinXP, so have to LoadLibrary to use it for now!
    if (!mfreadwrite) {
        FreeLibrary(mfplat);
        FreeLibrary(mf);
        return false;
    }

    bool okay = true;
    #define LOADSYM(lib, fn) if (okay) { p##fn = (pfn##fn) GetProcAddress(lib, #fn); if (!p##fn) { okay = false; } }
    LOADSYM(mf, MFEnumDeviceSources);
    LOADSYM(mf, MFCreateDeviceSource);
    LOADSYM(mfplat, MFStartup);
    LOADSYM(mfplat, MFShutdown);
    LOADSYM(mfplat, MFCreateAttributes);
    LOADSYM(mfplat, MFCreateMediaType);
    LOADSYM(mfplat, MFGetStrideForBitmapInfoHeader);
    LOADSYM(mfreadwrite, MFCreateSourceReaderFromMediaSource);
    #undef LOADSYM

    if (okay) {
        const HRESULT ret = pMFStartup(MF_VERSION, MFSTARTUP_LITE);
        if (FAILED(ret)) {
            okay = false;
        }
    }

    if (!okay) {
        FreeLibrary(mfreadwrite);
        FreeLibrary(mfplat);
        FreeLibrary(mf);
        return false;
    }

    libmf = mf;
    libmfplat = mfplat;
    libmfreadwrite = mfreadwrite;

    impl->DetectDevices = MEDIAFOUNDATION_DetectDevices;
    impl->OpenDevice = MEDIAFOUNDATION_OpenDevice;
    impl->CloseDevice = MEDIAFOUNDATION_CloseDevice;
    impl->WaitDevice = MEDIAFOUNDATION_WaitDevice;
    impl->AcquireFrame = MEDIAFOUNDATION_AcquireFrame;
    impl->ReleaseFrame = MEDIAFOUNDATION_ReleaseFrame;
    impl->FreeDeviceHandle = MEDIAFOUNDATION_FreeDeviceHandle;
    impl->Deinitialize = MEDIAFOUNDATION_Deinitialize;

    return true;
}

CameraBootStrap MEDIAFOUNDATION_bootstrap = {
    "mediafoundation", "SDL Windows Media Foundation camera driver", MEDIAFOUNDATION_Init, false
};

#endif // SDL_CAMERA_DRIVER_MEDIAFOUNDATION

