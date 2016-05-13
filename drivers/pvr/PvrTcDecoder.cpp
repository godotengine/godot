//============================================================================

#include "PvrTcDecoder.h"
#include "PvrTcPacket.h"

#include "MortonTable.h"
#include <assert.h>

//============================================================================

using namespace Javelin;
using Data::MORTON_TABLE;

//============================================================================

inline unsigned PvrTcDecoder::GetMortonNumber(int x, int y)
{
    return MORTON_TABLE[x >> 8] << 17 | MORTON_TABLE[y >> 8] << 16 | MORTON_TABLE[x & 0xFF] << 1 | MORTON_TABLE[y & 0xFF];
}

//============================================================================

void PvrTcDecoder::DecodeRgb4Bpp(ColorRgb<unsigned char>* result, const Point2<int>& size, const void* data)
{
    assert(size.x == size.y);
	
	const int blocks = size.x / 4;
	const int blockMask = blocks-1;
    const PvrTcPacket* packets = static_cast<const PvrTcPacket*>(data);
    
    for(int y = 0; y < blocks; ++y)
    {
        for(int x = 0; x < blocks; ++x)
        {
            const PvrTcPacket* packet = packets + GetMortonNumber(x, y);
            
            unsigned mod = packet->modulationData;
			const unsigned char (*weights)[4] = PvrTcPacket::WEIGHTS + 4*packet->usePunchthroughAlpha;
            const unsigned char (*factor)[4] = PvrTcPacket::BILINEAR_FACTORS;
			
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
					
					const unsigned char* w = weights[mod&3];
					ColorRgb<unsigned char> c;
					c.r = (ca.r * w[0] + cb.r * w[1]) >> 7;
					c.g = (ca.g * w[0] + cb.g * w[1]) >> 7;
					c.b = (ca.b * w[0] + cb.b * w[1]) >> 7;
					
					result[(py+y*4)*size.x + (px+x*4)] = c;
					mod >>= 2;
					factor++;
				}
			}
        }
    }
}

void PvrTcDecoder::DecodeRgba4Bpp(ColorRgba<unsigned char>* result, const Point2<int>& size, const void* data)
{
    assert(size.x == size.y);
    
	const int blocks = size.x / 4;
	const int blockMask = blocks-1;
    const PvrTcPacket* packets = static_cast<const PvrTcPacket*>(data);
    
    for(int y = 0; y < blocks; ++y)
    {
        for(int x = 0; x < blocks; ++x)
        {
            const PvrTcPacket* packet = packets + GetMortonNumber(x, y);
            
            unsigned mod = packet->modulationData;
            const unsigned char (*weights)[4] = PvrTcPacket::WEIGHTS + 4*packet->usePunchthroughAlpha;
            const unsigned char (*factor)[4] = PvrTcPacket::BILINEAR_FACTORS;
			
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
					
					const unsigned char* w = weights[mod&3];
					ColorRgba<unsigned char> c;
					c.r = (ca.r * w[0] + cb.r * w[1]) >> 7;
					c.g = (ca.g * w[0] + cb.g * w[1]) >> 7;
					c.b = (ca.b * w[0] + cb.b * w[1]) >> 7;
					c.a = (ca.a * w[2] + cb.a * w[3]) >> 7;
					
					result[(py+y*4)*size.x + (px+x*4)] = c;
					mod >>= 2;
					factor++;
				}
			}
        }
    }
}

//============================================================================
