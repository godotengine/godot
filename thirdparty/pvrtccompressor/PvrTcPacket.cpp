//============================================================================

#include "PvrTcPacket.h"
#include "BitScale.h"

//============================================================================

using namespace Javelin;

//============================================================================

const unsigned char PvrTcPacket::BILINEAR_FACTORS[16][4] =
{
	{ 4, 4, 4, 4 },
	{ 2, 6, 2, 6 },
	{ 8, 0, 8, 0 },
	{ 6, 2, 6, 2 },
	
	{ 2, 2, 6, 6 },
	{ 1, 3, 3, 9 },
	{ 4, 0, 12, 0 },
	{ 3, 1, 9, 3 },
	
	{ 8, 8, 0, 0 },
	{ 4, 12, 0, 0 },
	{ 16, 0, 0, 0 },
	{ 12, 4, 0, 0 },
	
	{ 6, 6, 2, 2 },
	{ 3, 9, 1, 3 },
	{ 12, 0, 4, 0 },
	{ 9, 3, 3, 1 },
};

// Weights are { colorA, colorB, alphaA, alphaB }
const unsigned char PvrTcPacket::WEIGHTS[8][4] =
{
	// Weights for Mode=0
	{ 8, 0, 8, 0 },
	{ 5, 3, 5, 3 },
	{ 3, 5, 3, 5 },
	{ 0, 8, 0, 8 },
	
	// Weights for Mode=1
	{ 8, 0, 8, 0 },
	{ 4, 4, 4, 4 },
	{ 4, 4, 0, 0 },
	{ 0, 8, 0, 8 },
};

//============================================================================

ColorRgb<int> PvrTcPacket::GetColorRgbA() const
{
	if(colorAIsOpaque)
	{
		unsigned char r = colorA >> 9;
		unsigned char g = colorA >> 4 & 0x1f;
		unsigned char b = colorA & 0xf;
		return ColorRgb<int>(Data::BITSCALE_5_TO_8[r],
							 Data::BITSCALE_5_TO_8[g],
							 Data::BITSCALE_4_TO_8[b]);
	}
	else
	{
		unsigned char r = (colorA >> 7) & 0xf;
		unsigned char g = (colorA >> 3) & 0xf;
		unsigned char b = colorA & 7;
		return ColorRgb<int>(Data::BITSCALE_4_TO_8[r],
							 Data::BITSCALE_4_TO_8[g],
							 Data::BITSCALE_3_TO_8[b]);
	}
}

ColorRgb<int> PvrTcPacket::GetColorRgbB() const
{
	if(colorBIsOpaque)
	{
		unsigned char r = colorB >> 10;
		unsigned char g = colorB >> 5 & 0x1f;
		unsigned char b = colorB & 0x1f;
		return ColorRgb<int>(Data::BITSCALE_5_TO_8[r],
							 Data::BITSCALE_5_TO_8[g],
							 Data::BITSCALE_5_TO_8[b]);
	}
	else
	{
		unsigned char r = colorB >> 8 & 0xf;
		unsigned char g = colorB >> 4 & 0xf;
		unsigned char b = colorB & 0xf;
		return ColorRgb<int>(Data::BITSCALE_4_TO_8[r],
							 Data::BITSCALE_4_TO_8[g],
							 Data::BITSCALE_4_TO_8[b]);
	}
}

ColorRgba<int> PvrTcPacket::GetColorRgbaA() const
{
	if(colorAIsOpaque)
	{
		unsigned char r = colorA >> 9;
		unsigned char g = colorA >> 4 & 0x1f;
		unsigned char b = colorA & 0xf;
		return ColorRgba<int>(Data::BITSCALE_5_TO_8[r],
							  Data::BITSCALE_5_TO_8[g],
							  Data::BITSCALE_4_TO_8[b],
							  255);
	}
	else
	{
		unsigned char a = colorA >> 11 & 7;
		unsigned char r = colorA >> 7 & 0xf;
		unsigned char g = colorA >> 3 & 0xf;
		unsigned char b = colorA & 7;
		return ColorRgba<int>(Data::BITSCALE_4_TO_8[r],
							  Data::BITSCALE_4_TO_8[g],
							  Data::BITSCALE_3_TO_8[b],
							  Data::BITSCALE_3_TO_8[a]);
	}
}

ColorRgba<int> PvrTcPacket::GetColorRgbaB() const
{
	if(colorBIsOpaque)
	{
		unsigned char r = colorB >> 10;
		unsigned char g = colorB >> 5 & 0x1f;
		unsigned char b = colorB & 0x1f;
		return ColorRgba<int>(Data::BITSCALE_5_TO_8[r],
							  Data::BITSCALE_5_TO_8[g],
							  Data::BITSCALE_5_TO_8[b],
							  255);
	}
	else
	{
		unsigned char a = colorB >> 12 & 7;
		unsigned char r = colorB >> 8 & 0xf;
		unsigned char g = colorB >> 4 & 0xf;
		unsigned char b = colorB & 0xf;
		return ColorRgba<int>(Data::BITSCALE_4_TO_8[r],
							  Data::BITSCALE_4_TO_8[g],
							  Data::BITSCALE_4_TO_8[b],
							  Data::BITSCALE_3_TO_8[a]);
	}
}

//============================================================================

void PvrTcPacket::SetColorA(const ColorRgb<unsigned char>& c)
{
	int r = Data::BITSCALE_8_TO_5_FLOOR[c.r];
	int g = Data::BITSCALE_8_TO_5_FLOOR[c.g];
	int b = Data::BITSCALE_8_TO_4_FLOOR[c.b];
	colorA = r<<9 | g<<4 | b;
	colorAIsOpaque = true;
}

void PvrTcPacket::SetColorB(const ColorRgb<unsigned char>& c)
{
	int r = Data::BITSCALE_8_TO_5_CEIL[c.r];
	int g = Data::BITSCALE_8_TO_5_CEIL[c.g];
	int b = Data::BITSCALE_8_TO_5_CEIL[c.b];
	colorB = r<<10 | g<<5 | b;
	colorBIsOpaque = true;
}

void PvrTcPacket::SetColorA(const ColorRgba<unsigned char>& c)
{
	int a = Data::BITSCALE_8_TO_3_FLOOR[c.a];
	if(a == 7)
	{
		int r = Data::BITSCALE_8_TO_5_FLOOR[c.r];
		int g = Data::BITSCALE_8_TO_5_FLOOR[c.g];
		int b = Data::BITSCALE_8_TO_4_FLOOR[c.b];
		colorA = r<<9 | g<<4 | b;
		colorAIsOpaque = true;
	}
	else
	{
		int r = Data::BITSCALE_8_TO_4_FLOOR[c.r];
		int g = Data::BITSCALE_8_TO_4_FLOOR[c.g];
		int b = Data::BITSCALE_8_TO_3_FLOOR[c.b];
		colorA = a<<11 | r<<7 | g<<3 | b;
		colorAIsOpaque = false;
	}
}

void PvrTcPacket::SetColorB(const ColorRgba<unsigned char>& c)
{
	int a = Data::BITSCALE_8_TO_3_CEIL[c.a];
	if(a == 7)
	{
		int r = Data::BITSCALE_8_TO_5_CEIL[c.r];
		int g = Data::BITSCALE_8_TO_5_CEIL[c.g];
		int b = Data::BITSCALE_8_TO_5_CEIL[c.b];
		colorB = r<<10 | g<<5 | b;
		colorBIsOpaque = true;
	}
	else
	{
		int r = Data::BITSCALE_8_TO_4_CEIL[c.r];
		int g = Data::BITSCALE_8_TO_4_CEIL[c.g];
		int b = Data::BITSCALE_8_TO_4_CEIL[c.b];
		colorB = a<<12 | r<<8 | g<<4 | b;
		colorBIsOpaque = false;
	}
}

//============================================================================
