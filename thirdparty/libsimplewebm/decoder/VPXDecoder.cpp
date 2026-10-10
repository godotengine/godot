/*
 *    MIT License
 *
 *    Copyright (c) 2016 Błażej Szczygieł
 * 	  Copyright (c) 2026-present The Simplicity Group & contributors
 * 	  See AUTHORS.md for more information
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include "include/decoder/VPXDecoder.h"

#include <vpx/vpx_decoder.h>
#include <vpx/vp8dx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

VPXDecoder::VPXDecoder(const WebMDemuxer &demuxer, unsigned threads) :
	m_ctx(NULL),
	m_alphaCtx(NULL),
	m_codecIface(NULL),
	m_iter(NULL),
	m_alphaIter(NULL),
	m_threads(1),
	m_delay(0),
	m_last_space(VPX_CS_UNKNOWN)
{
	if (threads > 8)
		threads = 8;
	else if (threads < 1)
		threads = 1;
	
	m_threads = threads;

	const vpx_codec_dec_cfg_t codecCfg = {
		threads,
		0,
		0
	};

	vpx_codec_iface_t *codecIface = NULL;

	switch (demuxer.getVideoCodec())
	{
		case WebMDemuxer::VIDEO_VP8:
			codecIface = vpx_codec_vp8_dx();
			break;

		case WebMDemuxer::VIDEO_VP9:
			codecIface = vpx_codec_vp9_dx();
			m_delay = threads - 1;
			break;

		default:
			return;
	}

	m_codecIface = codecIface;

	m_ctx = new vpx_codec_ctx_t;

	if (vpx_codec_dec_init(m_ctx, codecIface, &codecCfg, m_delay > 0 ? VPX_CODEC_USE_FRAME_THREADING : 0))
	{
		delete m_ctx;
		m_ctx = NULL;
	}
}

VPXDecoder::~VPXDecoder()
{
	if (m_alphaCtx)
	{
		vpx_codec_destroy(m_alphaCtx);
		delete m_alphaCtx;
	}

	if (m_ctx)
	{
		vpx_codec_destroy(m_ctx);
		delete m_ctx;
	}
}

bool VPXDecoder::decode(const WebMFrame &frame)
{
	if (!m_ctx || !frame.isValid())
		return false;

	m_iter = NULL;
	m_alphaIter = NULL;

	if (vpx_codec_decode(m_ctx, frame.buffer, static_cast<unsigned int>(frame.bufferSize), NULL, 0))
	{
		return false;
	}

	// WebM alpha is stored as a second VP8/VP9 bitstream in
	// BlockAdditional with BlockAddID 1.
	// libvpx doesn't decode that data into VPX_PLANE_ALPHA.
	// It needs its own decoder instance.

	if (frame.hasAlpha())
	{
		if (!m_alphaCtx)
		{
			const vpx_codec_dec_cfg_t alphaCodecCfg = {
				m_threads,
				0,
				0
			};

			m_alphaCtx = new vpx_codec_ctx_t;

			if (vpx_codec_dec_init(m_alphaCtx, m_codecIface, &alphaCodecCfg, m_delay > 0 ? VPX_CODEC_USE_FRAME_THREADING : 0))
			{
				delete m_alphaCtx;
				m_alphaCtx = NULL;
				return false;
			}
		}

		if (vpx_codec_decode(m_alphaCtx, frame.alphaBuffer, static_cast<unsigned int>(frame.alphaBufferSize), NULL, 0))
		{
			return false;
		}
	}

	return true;
}

VPXDecoder::IMAGE_ERROR VPXDecoder::getImage(Image &image)
{
	IMAGE_ERROR err = NO_FRAME;

	vpx_image_t *img = NULL;
	vpx_image_t *alphaImg = NULL;

	if (m_ctx)
		img = vpx_codec_get_frame(m_ctx, &m_iter);

	if (m_alphaCtx)
		alphaImg = vpx_codec_get_frame(m_alphaCtx, &m_alphaIter);

	if (!img)
		return NO_FRAME;


	// The alpha decoder returns an ordinary I420 image.
	// Its Y plane is the actual alpha plane.

	if (alphaImg)
	{
		if (img->d_w != alphaImg->d_w || img->d_h != alphaImg->d_h)
		{
			return UNSUPPORTED_FRAME;
		}
	}

	if (img->cs != VPX_CS_UNKNOWN)
		m_last_space = img->cs;

	if ((img->fmt & VPX_IMG_FMT_PLANAR) && !(img->fmt & VPX_IMG_FMT_HIGHBITDEPTH))
	{
		if (img->stride[0] && img->stride[1] && img->stride[2])
		{
			const int uPlane = !!(img->fmt & VPX_IMG_FMT_UV_FLIP) + 1;

			const int vPlane = !(img->fmt & VPX_IMG_FMT_UV_FLIP) + 1;

			image.w = img->d_w;
			image.h = img->d_h;
			image.cs = m_last_space;
			image.chromaShiftW = img->x_chroma_shift;
			image.chromaShiftH = img->y_chroma_shift;

			image.planes[0] = img->planes[0];
			image.planes[1] = img->planes[uPlane];
			image.planes[2] = img->planes[vPlane];

			image.linesize[0] = img->stride[0];
			image.linesize[1] = img->stride[uPlane];
			image.linesize[2] = img->stride[vPlane];

			if (alphaImg && alphaImg->planes[VPX_PLANE_Y] && alphaImg->stride[VPX_PLANE_Y])
			{
				image.alpha = alphaImg->planes[VPX_PLANE_Y];
				image.alphaLinesize = alphaImg->stride[VPX_PLANE_Y];
			}
			else
			{
				image.alpha = NULL;
				image.alphaLinesize = 0;
			}

			err = NO_ERROR;
		}
	}
	else
	{
		err = UNSUPPORTED_FRAME;
	}

	return err;
}

static inline int ceilRshift(int val, int shift)
{
	return (val + (1 << shift) - 1) >> shift;
}

// 0 = Y, 1 = U, 2 = V, 3 = alpha.
int VPXDecoder::Image::getWidth(int plane) const
{
	if (plane == 0 || plane == 3)
		return w;

	return ceilRshift(w, chromaShiftW);
}

int VPXDecoder::Image::getHeight(int plane) const
{
	if (plane == 0 || plane == 3)
		return h;

	return ceilRshift(h, chromaShiftH);
}
