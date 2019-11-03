//============================================================================

#include "PvrTcEncoder.h"
#include "AlphaBitmap.h"
#include "PvrTcPacket.h"
#include "RgbBitmap.h"
#include "RgbaBitmap.h"
#include "MortonTable.h"
#include "BitUtility.h"
#include "Interval.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>

//============================================================================

using namespace Javelin;
using Data::MORTON_TABLE;

//============================================================================

static const unsigned char MODULATION_LUT[16] =
{
	0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3
};

//============================================================================

inline unsigned PvrTcEncoder::GetMortonNumber(int x, int y)
{
	return MORTON_TABLE[x >> 8] << 17 | MORTON_TABLE[y >> 8] << 16 | MORTON_TABLE[x & 0xFF] << 1 | MORTON_TABLE[y & 0xFF];
}

//============================================================================

void PvrTcEncoder::EncodeAlpha2Bpp(void* result, const AlphaBitmap& bitmap)
{
	int size = bitmap.GetBitmapWidth();
	assert(size == bitmap.GetBitmapHeight());
	assert(BitUtility::IsPowerOf2(size));
	
	// Blocks in each dimension.
	int xBlocks = size/8;
	int yBlocks = size/4;
	
	const unsigned char* bitmapData = bitmap.GetRawData();
	
	PvrTcPacket* packets = static_cast<PvrTcPacket*>(result);
	for(int y = 0; y < yBlocks; ++y)
	{
		for(int x = 0; x < xBlocks; ++x)
		{
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->usePunchthroughAlpha = 0;
			packet->colorAIsOpaque = 0;
			packet->colorA = 0x7ff;		// White, with 0 alpha
			packet->colorBIsOpaque = 1;
			packet->colorB = 0x7fff;	// White with full alpha
			
			const unsigned char* blockBitmapData = &bitmapData[y*4*size + x*8];
			
			uint32_t modulationData = 0;
			for(int py = 0; py < 4; ++py)
			{
				const unsigned char* rowBitmapData = blockBitmapData;
				for(int px = 0; px < 8; ++px)
				{
					unsigned char pixel = *rowBitmapData++;
					modulationData = BitUtility::RotateRight(modulationData | (pixel >> 7), 1);
				}
				blockBitmapData += size;
			}
			packet->modulationData = modulationData;
		}
	}
}

void PvrTcEncoder::EncodeAlpha4Bpp(void* result, const AlphaBitmap& bitmap)
{
	int size = bitmap.GetBitmapWidth();
	assert(size == bitmap.GetBitmapHeight());
	assert(BitUtility::IsPowerOf2(size));
	
	// Blocks in each dimension.
	int blocks = size/4;
	
	const unsigned char* bitmapData = bitmap.GetRawData();
	
	PvrTcPacket* packets = static_cast<PvrTcPacket*>(result);
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->usePunchthroughAlpha = 0;
			packet->colorAIsOpaque = 0;
			packet->colorA = 0x7ff;		// White, with 0 alpha
			packet->colorBIsOpaque = 1;
			packet->colorB = 0x7fff;	// White with full alpha

			const unsigned char* blockBitmapData = &bitmapData[(y*size + x)*4];
			
			uint32_t modulationData = 0;
			for(int py = 0; py < 4; ++py)
			{
				const unsigned char* rowBitmapData = blockBitmapData;
				for(int px = 0; px < 4; ++px)
				{
					unsigned char pixel = *rowBitmapData++;
					modulationData = BitUtility::RotateRight(modulationData | MODULATION_LUT[pixel>>4], 2);
				}
				blockBitmapData += size;
			}
			packet->modulationData = modulationData;
		}
	}
}

//============================================================================

typedef Interval<ColorRgb<unsigned char> > ColorRgbBoundingBox;

static void CalculateBoundingBox(ColorRgbBoundingBox& cbb, const RgbBitmap& bitmap, int blockX, int blockY)
{
	int size = bitmap.GetBitmapWidth();
	const ColorRgb<unsigned char>* data = bitmap.GetData() + blockY * 4 * size + blockX * 4;
	
	cbb.min = data[0];
	cbb.max = data[0];
	cbb |= data[1];
	cbb |= data[2];
	cbb |= data[3];
	
	cbb |= data[size];
	cbb |= data[size+1];
	cbb |= data[size+2];
	cbb |= data[size+3];

	cbb |= data[2*size];
	cbb |= data[2*size+1];
	cbb |= data[2*size+2];
	cbb |= data[2*size+3];

	cbb |= data[3*size];
	cbb |= data[3*size+1];
	cbb |= data[3*size+2];
	cbb |= data[3*size+3];
}

void PvrTcEncoder::EncodeRgb4Bpp(void* result, const RgbBitmap& bitmap)
{
	assert(bitmap.GetBitmapWidth() == bitmap.GetBitmapHeight());
	assert(BitUtility::IsPowerOf2(bitmap.GetBitmapWidth()));
	const int size = bitmap.GetBitmapWidth();
	const int blocks = size / 4;
	const int blockMask = blocks-1;
	
	PvrTcPacket* packets = static_cast<PvrTcPacket*>(result);

	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			ColorRgbBoundingBox cbb;
			CalculateBoundingBox(cbb, bitmap, x, y);
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->usePunchthroughAlpha = 0;
			packet->SetColorA(cbb.min);
			packet->SetColorB(cbb.max);
		}
	}
	
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			const unsigned char (*factor)[4] = PvrTcPacket::BILINEAR_FACTORS;
			const ColorRgb<unsigned char>* data = bitmap.GetData() + y * 4 * size + x * 4;

			uint32_t modulationData = 0;
			
			for(int py = 0; py < 4; ++py)
			{
				const int yOffset = (py < 2) ? -1 : 0;
				const int y0 = (y + yOffset) & blockMask;
				const int y1 = (y0+1) & blockMask;

				for(int px = 0; px < 4; ++px)
				{
					const int xOffset = (px < 2) ? -1 : 0;
					const int x0 = (x + xOffset) & blockMask;
					const int x1 = (x0+1) & blockMask;
					
					const PvrTcPacket* p0 = packets + GetMortonNumber(x0, y0);
					const PvrTcPacket* p1 = packets + GetMortonNumber(x1, y0);
					const PvrTcPacket* p2 = packets + GetMortonNumber(x0, y1);
					const PvrTcPacket* p3 = packets + GetMortonNumber(x1, y1);
					
					ColorRgb<int> ca = p0->GetColorRgbA() * (*factor)[0] +
									   p1->GetColorRgbA() * (*factor)[1] +
									   p2->GetColorRgbA() * (*factor)[2] +
									   p3->GetColorRgbA() * (*factor)[3];
					
					ColorRgb<int> cb = p0->GetColorRgbB() * (*factor)[0] +
									   p1->GetColorRgbB() * (*factor)[1] +
									   p2->GetColorRgbB() * (*factor)[2] +
									   p3->GetColorRgbB() * (*factor)[3];
					
					const ColorRgb<unsigned char>& pixel = data[py*size + px];
					ColorRgb<int> d = cb - ca;
					ColorRgb<int> p;
					p.r=pixel.r*16;
					p.g=pixel.g*16;
					p.b=pixel.b*16;
					ColorRgb<int> v = p - ca;
					
					// PVRTC uses weightings of 0, 3/8, 5/8 and 1
					// The boundaries for these are 3/16, 1/2 (=8/16), 13/16
					int projection = (v % d) * 16;
					int lengthSquared = d % d;
					if(projection > 3*lengthSquared) modulationData++;
					if(projection > 8*lengthSquared) modulationData++;
					if(projection > 13*lengthSquared) modulationData++;
					
					modulationData = BitUtility::RotateRight(modulationData, 2);
					
					factor++;
				}
			}

			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->modulationData = modulationData;
		}
	}
}

//============================================================================

static void CalculateBoundingBox(ColorRgbBoundingBox& cbb, const RgbaBitmap& bitmap, int blockX, int blockY)
{
	int size = bitmap.GetBitmapWidth();
	const ColorRgba<unsigned char>* data = bitmap.GetData() + blockY * 4 * size + blockX * 4;
	
	cbb.min = data[0];
	cbb.max = data[0];
	
	cbb |= data[1];
	cbb |= data[2];
	cbb |= data[3];
	
	cbb |= data[size];
	cbb |= data[size+1];
	cbb |= data[size+2];
	cbb |= data[size+3];
	
	cbb |= data[2*size];
	cbb |= data[2*size+1];
	cbb |= data[2*size+2];
	cbb |= data[2*size+3];
	
	cbb |= data[3*size];
	cbb |= data[3*size+1];
	cbb |= data[3*size+2];
	cbb |= data[3*size+3];
}

void PvrTcEncoder::EncodeRgb4Bpp(void* result, const RgbaBitmap& bitmap)
{
	assert(bitmap.GetBitmapWidth() == bitmap.GetBitmapHeight());
	assert(BitUtility::IsPowerOf2(bitmap.GetBitmapWidth()));
	const int size = bitmap.GetBitmapWidth();
	const int blocks = size / 4;
	const int blockMask = blocks-1;
	
	PvrTcPacket* packets = static_cast<PvrTcPacket*>(result);
	
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			ColorRgbBoundingBox cbb;
			CalculateBoundingBox(cbb, bitmap, x, y);
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->usePunchthroughAlpha = 0;
			packet->SetColorA(cbb.min);
			packet->SetColorB(cbb.max);
		}
	}
	
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			const unsigned char (*factor)[4] = PvrTcPacket::BILINEAR_FACTORS;
			const ColorRgba<unsigned char>* data = bitmap.GetData() + y * 4 * size + x * 4;
			
			uint32_t modulationData = 0;
			
			for(int py = 0; py < 4; ++py)
			{
				const int yOffset = (py < 2) ? -1 : 0;
				const int y0 = (y + yOffset) & blockMask;
				const int y1 = (y0+1) & blockMask;

				for(int px = 0; px < 4; ++px)
				{
					const int xOffset = (px < 2) ? -1 : 0;
					const int x0 = (x + xOffset) & blockMask;
					const int x1 = (x0+1) & blockMask;
					
					const PvrTcPacket* p0 = packets + GetMortonNumber(x0, y0);
					const PvrTcPacket* p1 = packets + GetMortonNumber(x1, y0);
					const PvrTcPacket* p2 = packets + GetMortonNumber(x0, y1);
					const PvrTcPacket* p3 = packets + GetMortonNumber(x1, y1);
					
					ColorRgb<int> ca = p0->GetColorRgbA() * (*factor)[0] +
									   p1->GetColorRgbA() * (*factor)[1] +
									   p2->GetColorRgbA() * (*factor)[2] +
									   p3->GetColorRgbA() * (*factor)[3];
					
					ColorRgb<int> cb = p0->GetColorRgbB() * (*factor)[0] +
									   p1->GetColorRgbB() * (*factor)[1] +
									   p2->GetColorRgbB() * (*factor)[2] +
									   p3->GetColorRgbB() * (*factor)[3];
					
					const ColorRgb<unsigned char>& pixel = data[py*size + px];
					ColorRgb<int> d = cb - ca;
					ColorRgb<int> p;
					p.r=pixel.r*16;
					p.g=pixel.g*16;
					p.b=pixel.b*16;
					ColorRgb<int> v = p - ca;
					
					// PVRTC uses weightings of 0, 3/8, 5/8 and 1
					// The boundaries for these are 3/16, 1/2 (=8/16), 13/16
					int projection = (v % d) * 16;
					int lengthSquared = d % d;
					if(projection > 3*lengthSquared) modulationData++;
					if(projection > 8*lengthSquared) modulationData++;
					if(projection > 13*lengthSquared) modulationData++;
					
					modulationData = BitUtility::RotateRight(modulationData, 2);
					
					factor++;
				}
			}

			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->modulationData = modulationData;
		}
	}
}

//============================================================================

typedef Interval<ColorRgba<unsigned char> > ColorRgbaBoundingBox;

static void CalculateBoundingBox(ColorRgbaBoundingBox& cbb, const RgbaBitmap& bitmap, int blockX, int blockY)
{
	int size = bitmap.GetBitmapWidth();
	const ColorRgba<unsigned char>* data = bitmap.GetData() + blockY * 4 * size + blockX * 4;
	
	cbb.min = data[0];
	cbb.max = data[0];
	
	cbb |= data[1];
	cbb |= data[2];
	cbb |= data[3];
	
	cbb |= data[size];
	cbb |= data[size+1];
	cbb |= data[size+2];
	cbb |= data[size+3];
	
	cbb |= data[2*size];
	cbb |= data[2*size+1];
	cbb |= data[2*size+2];
	cbb |= data[2*size+3];
	
	cbb |= data[3*size];
	cbb |= data[3*size+1];
	cbb |= data[3*size+2];
	cbb |= data[3*size+3];
}

void PvrTcEncoder::EncodeRgba4Bpp(void* result, const RgbaBitmap& bitmap)
{
	assert(bitmap.GetBitmapWidth() == bitmap.GetBitmapHeight());
	assert(BitUtility::IsPowerOf2(bitmap.GetBitmapWidth()));
	const int size = bitmap.GetBitmapWidth();
	const int blocks = size / 4;
	const int blockMask = blocks-1;
	
	PvrTcPacket* packets = static_cast<PvrTcPacket*>(result);
	
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			ColorRgbaBoundingBox cbb;
			CalculateBoundingBox(cbb, bitmap, x, y);
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->usePunchthroughAlpha = 0;
			packet->SetColorA(cbb.min);
			packet->SetColorB(cbb.max);
		}
	}
	
	for(int y = 0; y < blocks; ++y)
	{
		for(int x = 0; x < blocks; ++x)
		{
			const unsigned char (*factor)[4] = PvrTcPacket::BILINEAR_FACTORS;
			const ColorRgba<unsigned char>* data = bitmap.GetData() + y * 4 * size + x * 4;
			
			uint32_t modulationData = 0;
			
			for(int py = 0; py < 4; ++py)
			{
				const int yOffset = (py < 2) ? -1 : 0;
				const int y0 = (y + yOffset) & blockMask;
				const int y1 = (y0+1) & blockMask;
				
				for(int px = 0; px < 4; ++px)
				{
					const int xOffset = (px < 2) ? -1 : 0;
					const int x0 = (x + xOffset) & blockMask;
					const int x1 = (x0+1) & blockMask;
					
					const PvrTcPacket* p0 = packets + GetMortonNumber(x0, y0);
					const PvrTcPacket* p1 = packets + GetMortonNumber(x1, y0);
					const PvrTcPacket* p2 = packets + GetMortonNumber(x0, y1);
					const PvrTcPacket* p3 = packets + GetMortonNumber(x1, y1);
					
					ColorRgba<int> ca = p0->GetColorRgbaA() * (*factor)[0] +
										p1->GetColorRgbaA() * (*factor)[1] +
										p2->GetColorRgbaA() * (*factor)[2] +
										p3->GetColorRgbaA() * (*factor)[3];
					
					ColorRgba<int> cb = p0->GetColorRgbaB() * (*factor)[0] +
										p1->GetColorRgbaB() * (*factor)[1] +
										p2->GetColorRgbaB() * (*factor)[2] +
										p3->GetColorRgbaB() * (*factor)[3];
					
					const ColorRgba<unsigned char>& pixel = data[py*size + px];
					ColorRgba<int> d = cb - ca;
					ColorRgba<int> p;
					p.r=pixel.r*16;
					p.g=pixel.g*16;
					p.b=pixel.b*16;
					p.a=pixel.a*16;
					ColorRgba<int> v = p - ca;
					
					// PVRTC uses weightings of 0, 3/8, 5/8 and 1
					// The boundaries for these are 3/16, 1/2 (=8/16), 13/16
					int projection = (v % d) * 16;
					int lengthSquared = d % d;
					if(projection > 3*lengthSquared) modulationData++;
					if(projection > 8*lengthSquared) modulationData++;
					if(projection > 13*lengthSquared) modulationData++;
					
					modulationData = BitUtility::RotateRight(modulationData, 2);
					
					factor++;
				}
			}
			
			PvrTcPacket* packet = packets + GetMortonNumber(x, y);
			packet->modulationData = modulationData;
		}
	}
}

//============================================================================
