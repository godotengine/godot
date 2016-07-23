// File: rg_etc1.cpp - Fast, high quality ETC1 block packer/unpacker - Rich Geldreich <richgel99@gmail.com>
// Please see ZLIB license at the end of rg_etc1.h.
//
// For more information Ericsson Texture Compression (ETC/ETC1), see:
// http://www.khronos.org/registry/gles/extensions/OES/OES_compressed_ETC1_RGB8_texture.txt
//
// v1.04 - 5/15/14 - Fix signed vs. unsigned subtraction problem (noticed when compiled with gcc) in pack_etc1_block_init(). 
//         This issue would cause an assert when this func. was called in debug. (Note this module was developed/testing with MSVC, 
//         I still need to test it throughly when compiled with gcc.)
//
// v1.03 - 5/12/13 - Initial public release
#include "rg_etc1.h"

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
//#include <stdio.h>
#include <math.h>

#pragma warning (disable: 4201) //  nonstandard extension used : nameless struct/union

#if defined(_DEBUG) || defined(DEBUG)
#define RG_ETC1_BUILD_DEBUG
#endif

#define RG_ETC1_ASSERT assert

namespace rg_etc1
{
   typedef unsigned char uint8;
   typedef unsigned short uint16;
   typedef unsigned int uint;
   typedef unsigned int uint32;
   typedef long long int64;
   typedef unsigned long long uint64;

   const uint32 cUINT32_MAX = 0xFFFFFFFFU;
   const uint64 cUINT64_MAX = 0xFFFFFFFFFFFFFFFFULL; //0xFFFFFFFFFFFFFFFFui64;
   
   template<typename T> inline T minimum(T a, T b) { return (a < b) ? a : b; }
   template<typename T> inline T minimum(T a, T b, T c) { return minimum(minimum(a, b), c); }
   template<typename T> inline T maximum(T a, T b) { return (a > b) ? a : b; }
   template<typename T> inline T maximum(T a, T b, T c) { return maximum(maximum(a, b), c); }
   template<typename T> inline T clamp(T value, T low, T high) { return (value < low) ? low : ((value > high) ? high : value); }
   template<typename T> inline T square(T value) { return value * value; }
   template<typename T> inline void zero_object(T& obj) { memset((void*)&obj, 0, sizeof(obj)); }
   template<typename T> inline void zero_this(T* pObj) { memset((void*)pObj, 0, sizeof(*pObj)); }

   template<class T, size_t N> T decay_array_to_subtype(T (&a)[N]);   

#define RG_ETC1_ARRAY_SIZE(X) (sizeof(X) / sizeof(decay_array_to_subtype(X)))

   enum eNoClamp { cNoClamp };

   struct color_quad_u8
   {
      static inline int clamp(int v) { if (v & 0xFFFFFF00U) v = (~(static_cast<int>(v) >> 31)) & 0xFF; return v; }

      struct component_traits { enum { cSigned = false, cFloat = false, cMin = 0U, cMax = 255U }; };

   public:
      typedef unsigned char component_t;
      typedef int parameter_t;

      enum { cNumComps = 4 };

      union
      {
         struct
         {
            component_t r;
            component_t g;
            component_t b;
            component_t a;
         };

         component_t c[cNumComps];

         uint32 m_u32;
      };

      inline color_quad_u8()
      {
      }

      inline color_quad_u8(const color_quad_u8& other) : m_u32(other.m_u32)
      {
      }

      explicit inline color_quad_u8(parameter_t y, parameter_t alpha = component_traits::cMax)
      {
         set(y, alpha);
      }

      inline color_quad_u8(parameter_t red, parameter_t green, parameter_t blue, parameter_t alpha = component_traits::cMax)
      {
         set(red, green, blue, alpha);
      }

      explicit inline color_quad_u8(eNoClamp, parameter_t y, parameter_t alpha = component_traits::cMax)
      {
         set_noclamp_y_alpha(y, alpha);
      }

      inline color_quad_u8(eNoClamp, parameter_t red, parameter_t green, parameter_t blue, parameter_t alpha = component_traits::cMax)
      {
         set_noclamp_rgba(red, green, blue, alpha);
      }

      inline void clear()
      {
         m_u32 = 0;
      }

      inline color_quad_u8& operator= (const color_quad_u8& other)
      {
         m_u32 = other.m_u32;
         return *this;
      }

      inline color_quad_u8& set_rgb(const color_quad_u8& other)
      {
         r = other.r;
         g = other.g;
         b = other.b;
         return *this;
      }

      inline color_quad_u8& operator= (parameter_t y)
      {
         set(y, component_traits::cMax);
         return *this;
      }

      inline color_quad_u8& set(parameter_t y, parameter_t alpha = component_traits::cMax)
      {
         y = clamp(y);
         alpha = clamp(alpha);
         r = static_cast<component_t>(y);
         g = static_cast<component_t>(y);
         b = static_cast<component_t>(y);
         a = static_cast<component_t>(alpha);
         return *this;
      }

      inline color_quad_u8& set_noclamp_y_alpha(parameter_t y, parameter_t alpha = component_traits::cMax)
      {
         RG_ETC1_ASSERT( (y >= component_traits::cMin) && (y <= component_traits::cMax) );
         RG_ETC1_ASSERT( (alpha >= component_traits::cMin) && (alpha <= component_traits::cMax) );

         r = static_cast<component_t>(y);
         g = static_cast<component_t>(y);
         b = static_cast<component_t>(y);
         a = static_cast<component_t>(alpha);
         return *this;
      }

      inline color_quad_u8& set(parameter_t red, parameter_t green, parameter_t blue, parameter_t alpha = component_traits::cMax)
      {
         r = static_cast<component_t>(clamp(red));
         g = static_cast<component_t>(clamp(green));
         b = static_cast<component_t>(clamp(blue));
         a = static_cast<component_t>(clamp(alpha));
         return *this;
      }

      inline color_quad_u8& set_noclamp_rgba(parameter_t red, parameter_t green, parameter_t blue, parameter_t alpha)
      {
         RG_ETC1_ASSERT( (red >= component_traits::cMin) && (red <= component_traits::cMax) );
         RG_ETC1_ASSERT( (green >= component_traits::cMin) && (green <= component_traits::cMax) );
         RG_ETC1_ASSERT( (blue >= component_traits::cMin) && (blue <= component_traits::cMax) );
         RG_ETC1_ASSERT( (alpha >= component_traits::cMin) && (alpha <= component_traits::cMax) );

         r = static_cast<component_t>(red);
         g = static_cast<component_t>(green);
         b = static_cast<component_t>(blue);
         a = static_cast<component_t>(alpha);
         return *this;
      }

      inline color_quad_u8& set_noclamp_rgb(parameter_t red, parameter_t green, parameter_t blue)
      {
         RG_ETC1_ASSERT( (red >= component_traits::cMin) && (red <= component_traits::cMax) );
         RG_ETC1_ASSERT( (green >= component_traits::cMin) && (green <= component_traits::cMax) );
         RG_ETC1_ASSERT( (blue >= component_traits::cMin) && (blue <= component_traits::cMax) );

         r = static_cast<component_t>(red);
         g = static_cast<component_t>(green);
         b = static_cast<component_t>(blue);
         return *this;
      }

      static inline parameter_t get_min_comp() { return component_traits::cMin; }
      static inline parameter_t get_max_comp() { return component_traits::cMax; }
      static inline bool get_comps_are_signed() { return component_traits::cSigned; }

      inline component_t operator[] (uint i) const { RG_ETC1_ASSERT(i < cNumComps); return c[i]; }
      inline component_t& operator[] (uint i) { RG_ETC1_ASSERT(i < cNumComps); return c[i]; }

      inline color_quad_u8& set_component(uint i, parameter_t f)
      {
         RG_ETC1_ASSERT(i < cNumComps);

         c[i] = static_cast<component_t>(clamp(f));

         return *this;
      }

      inline color_quad_u8& set_grayscale(parameter_t l)
      {
         component_t x = static_cast<component_t>(clamp(l));
         c[0] = x;
         c[1] = x;
         c[2] = x;
         return *this;
      }

      inline color_quad_u8& clamp(const color_quad_u8& l, const color_quad_u8& h)
      {
         for (uint i = 0; i < cNumComps; i++)
            c[i] = static_cast<component_t>(rg_etc1::clamp<parameter_t>(c[i], l[i], h[i]));
         return *this;
      }

      inline color_quad_u8& clamp(parameter_t l, parameter_t h)
      {
         for (uint i = 0; i < cNumComps; i++)
            c[i] = static_cast<component_t>(rg_etc1::clamp<parameter_t>(c[i], l, h));
         return *this;
      }

      // Returns CCIR 601 luma (consistent with color_utils::RGB_To_Y).
      inline parameter_t get_luma() const
      {
         return static_cast<parameter_t>((19595U * r + 38470U * g + 7471U * b + 32768U) >> 16U);
      }

      // Returns REC 709 luma.
      inline parameter_t get_luma_rec709() const
      {
         return static_cast<parameter_t>((13938U * r + 46869U * g + 4729U * b + 32768U) >> 16U);
      }

      inline uint squared_distance_rgb(const color_quad_u8& c) const
      {
         return rg_etc1::square(r - c.r) + rg_etc1::square(g - c.g) + rg_etc1::square(b - c.b);
      }

      inline uint squared_distance_rgba(const color_quad_u8& c) const
      {
         return rg_etc1::square(r - c.r) + rg_etc1::square(g - c.g) + rg_etc1::square(b - c.b) + rg_etc1::square(a - c.a);
      }

      inline bool rgb_equals(const color_quad_u8& rhs) const
      {
         return (r == rhs.r) && (g == rhs.g) && (b == rhs.b);
      }

      inline bool operator== (const color_quad_u8& rhs) const
      {
         return m_u32 == rhs.m_u32;
      }

      color_quad_u8& operator+= (const color_quad_u8& other)
      {
         for (uint i = 0; i < 4; i++)
            c[i] = static_cast<component_t>(clamp(c[i] + other.c[i]));
         return *this;
      }

      color_quad_u8& operator-= (const color_quad_u8& other)
      {
         for (uint i = 0; i < 4; i++)
            c[i] = static_cast<component_t>(clamp(c[i] - other.c[i]));
         return *this;
      }

      friend color_quad_u8 operator+ (const color_quad_u8& lhs, const color_quad_u8& rhs)
      {
         color_quad_u8 result(lhs);
         result += rhs;
         return result;
      }

      friend color_quad_u8 operator- (const color_quad_u8& lhs, const color_quad_u8& rhs)
      {
         color_quad_u8 result(lhs);
         result -= rhs;
         return result;
      }
   }; // class color_quad_u8

   struct vec3F
   {
      float m_s[3];
      
      inline vec3F() { }
      inline vec3F(float s) { m_s[0] = s; m_s[1] = s; m_s[2] = s; }
      inline vec3F(float x, float y, float z) { m_s[0] = x; m_s[1] = y; m_s[2] = z; }
      
      inline float operator[] (uint i) const { RG_ETC1_ASSERT(i < 3); return m_s[i]; }

      inline vec3F& operator += (const vec3F& other) { for (uint i = 0; i < 3; i++) m_s[i] += other.m_s[i]; return *this; }

      inline vec3F& operator *= (float s) { for (uint i = 0; i < 3; i++) m_s[i] *= s; return *this; }
   };
     
   enum etc_constants
   {
      cETC1BytesPerBlock = 8U,

      cETC1SelectorBits = 2U,
      cETC1SelectorValues = 1U << cETC1SelectorBits,
      cETC1SelectorMask = cETC1SelectorValues - 1U,

      cETC1BlockShift = 2U,
      cETC1BlockSize = 1U << cETC1BlockShift,

      cETC1LSBSelectorIndicesBitOffset = 0,
      cETC1MSBSelectorIndicesBitOffset = 16,

      cETC1FlipBitOffset = 32,
      cETC1DiffBitOffset = 33,

      cETC1IntenModifierNumBits = 3,
      cETC1IntenModifierValues = 1 << cETC1IntenModifierNumBits,
      cETC1RightIntenModifierTableBitOffset = 34,
      cETC1LeftIntenModifierTableBitOffset = 37,

      // Base+Delta encoding (5 bit bases, 3 bit delta)
      cETC1BaseColorCompNumBits = 5,
      cETC1BaseColorCompMax = 1 << cETC1BaseColorCompNumBits,

      cETC1DeltaColorCompNumBits = 3,
      cETC1DeltaColorComp = 1 << cETC1DeltaColorCompNumBits,
      cETC1DeltaColorCompMax = 1 << cETC1DeltaColorCompNumBits,

      cETC1BaseColor5RBitOffset = 59,
      cETC1BaseColor5GBitOffset = 51,
      cETC1BaseColor5BBitOffset = 43,

      cETC1DeltaColor3RBitOffset = 56,
      cETC1DeltaColor3GBitOffset = 48,
      cETC1DeltaColor3BBitOffset = 40,

      // Absolute (non-delta) encoding (two 4-bit per component bases)
      cETC1AbsColorCompNumBits = 4,
      cETC1AbsColorCompMax = 1 << cETC1AbsColorCompNumBits,

      cETC1AbsColor4R1BitOffset = 60,
      cETC1AbsColor4G1BitOffset = 52,
      cETC1AbsColor4B1BitOffset = 44,

      cETC1AbsColor4R2BitOffset = 56,
      cETC1AbsColor4G2BitOffset = 48,
      cETC1AbsColor4B2BitOffset = 40,

      cETC1ColorDeltaMin = -4,
      cETC1ColorDeltaMax = 3,

      // Delta3:
      // 0   1   2   3   4   5   6   7
      // 000 001 010 011 100 101 110 111
      // 0   1   2   3   -4  -3  -2  -1
   };
   
   static uint8 g_quant5_tab[256+16];

   static const int g_etc1_inten_tables[cETC1IntenModifierValues][cETC1SelectorValues] = 
   { 
      { -8,  -2,   2,   8 }, { -17,  -5,  5,  17 }, { -29,  -9,   9,  29 }, {  -42, -13, 13,  42 }, 
      { -60, -18, 18,  60 }, { -80, -24, 24,  80 }, { -106, -33, 33, 106 }, { -183, -47, 47, 183 } 
   };

   static const uint8 g_etc1_to_selector_index[cETC1SelectorValues] = { 2, 3, 1, 0 };
   static const uint8 g_selector_index_to_etc1[cETC1SelectorValues] = { 3, 2, 0, 1 };
      
   // Given an ETC1 diff/inten_table/selector, and an 8-bit desired color, this table encodes the best packed_color in the low byte, and the abs error in the high byte.
   static uint16 g_etc1_inverse_lookup[2*8*4][256];      // [diff/inten_table/selector][desired_color]

   // g_color8_to_etc_block_config[color][table_index] = Supplies for each 8-bit color value a list of packed ETC1 diff/intensity table/selectors/packed_colors that map to that color.
   // To pack: diff | (inten << 1) | (selector << 4) | (packed_c << 8)
   static const uint16 g_color8_to_etc_block_config_0_255[2][33] =
   {
      { 0x0000,  0x0010,  0x0002,  0x0012,  0x0004,  0x0014,  0x0006,  0x0016,  0x0008,  0x0018,  0x000A,  0x001A,  0x000C,  0x001C,  0x000E,  0x001E,
        0x0001,  0x0011,  0x0003,  0x0013,  0x0005,  0x0015,  0x0007,  0x0017,  0x0009,  0x0019,  0x000B,  0x001B,  0x000D,  0x001D,  0x000F,  0x001F, 0xFFFF },
      { 0x0F20,  0x0F30,  0x0E32,  0x0F22,  0x0E34,  0x0F24,  0x0D36,  0x0F26,  0x0C38,  0x0E28,  0x0B3A,  0x0E2A,  0x093C,  0x0E2C,  0x053E,  0x0D2E,
        0x1E31,  0x1F21,  0x1D33,  0x1F23,  0x1C35,  0x1E25,  0x1A37,  0x1E27,  0x1839,  0x1D29,  0x163B,  0x1C2B,  0x133D,  0x1B2D,  0x093F,  0x1A2F, 0xFFFF },
   };

   // Really only [254][11].
   static const uint16 g_color8_to_etc_block_config_1_to_254[254][12] = 
   {
      { 0x021C, 0x0D0D, 0xFFFF }, { 0x0020, 0x0021, 0x0A0B, 0x061F, 0xFFFF }, { 0x0113, 0x0217, 0xFFFF }, { 0x0116, 0x031E,
      0x0B0E, 0x0405, 0xFFFF }, { 0x0022, 0x0204, 0x050A, 0x0023, 0xFFFF }, { 0x0111, 0x0319, 0x0809, 0x170F, 0xFFFF }, {
      0x0303, 0x0215, 0x0607, 0xFFFF }, { 0x0030, 0x0114, 0x0408, 0x0031, 0x0201, 0x051D, 0xFFFF }, { 0x0100, 0x0024, 0x0306,
      0x0025, 0x041B, 0x0E0D, 0xFFFF }, { 0x021A, 0x0121, 0x0B0B, 0x071F, 0xFFFF }, { 0x0213, 0x0317, 0xFFFF }, { 0x0112,
      0x0505, 0xFFFF }, { 0x0026, 0x070C, 0x0123, 0x0027, 0xFFFF }, { 0x0211, 0x0909, 0xFFFF }, { 0x0110, 0x0315, 0x0707,
      0x0419, 0x180F, 0xFFFF }, { 0x0218, 0x0131, 0x0301, 0x0403, 0x061D, 0xFFFF }, { 0x0032, 0x0202, 0x0033, 0x0125, 0x051B,
      0x0F0D, 0xFFFF }, { 0x0028, 0x031C, 0x0221, 0x0029, 0xFFFF }, { 0x0120, 0x0313, 0x0C0B, 0x081F, 0xFFFF }, { 0x0605,
      0x0417, 0xFFFF }, { 0x0216, 0x041E, 0x0C0E, 0x0223, 0x0127, 0xFFFF }, { 0x0122, 0x0304, 0x060A, 0x0311, 0x0A09, 0xFFFF
      }, { 0x0519, 0x190F, 0xFFFF }, { 0x002A, 0x0231, 0x0503, 0x0415, 0x0807, 0x002B, 0x071D, 0xFFFF }, { 0x0130, 0x0214,
      0x0508, 0x0401, 0x0133, 0x0225, 0x061B, 0xFFFF }, { 0x0200, 0x0124, 0x0406, 0x0321, 0x0129, 0x100D, 0xFFFF }, { 0x031A,
      0x0D0B, 0x091F, 0xFFFF }, { 0x0413, 0x0705, 0x0517, 0xFFFF }, { 0x0212, 0x0034, 0x0323, 0x0035, 0x0227, 0xFFFF }, {
      0x0126, 0x080C, 0x0B09, 0xFFFF }, { 0x0411, 0x0619, 0x1A0F, 0xFFFF }, { 0x0210, 0x0331, 0x0603, 0x0515, 0x0907, 0x012B,
      0xFFFF }, { 0x0318, 0x002C, 0x0501, 0x0233, 0x0325, 0x071B, 0x002D, 0x081D, 0xFFFF }, { 0x0132, 0x0302, 0x0229, 0x110D,
      0xFFFF }, { 0x0128, 0x041C, 0x0421, 0x0E0B, 0x0A1F, 0xFFFF }, { 0x0220, 0x0513, 0x0617, 0xFFFF }, { 0x0135, 0x0805,
      0x0327, 0xFFFF }, { 0x0316, 0x051E, 0x0D0E, 0x0423, 0xFFFF }, { 0x0222, 0x0404, 0x070A, 0x0511, 0x0719, 0x0C09, 0x1B0F,
      0xFFFF }, { 0x0703, 0x0615, 0x0A07, 0x022B, 0xFFFF }, { 0x012A, 0x0431, 0x0601, 0x0333, 0x012D, 0x091D, 0xFFFF }, {
      0x0230, 0x0314, 0x0036, 0x0608, 0x0425, 0x0037, 0x0329, 0x081B, 0x120D, 0xFFFF }, { 0x0300, 0x0224, 0x0506, 0x0521,
      0x0F0B, 0x0B1F, 0xFFFF }, { 0x041A, 0x0613, 0x0717, 0xFFFF }, { 0x0235, 0x0905, 0xFFFF }, { 0x0312, 0x0134, 0x0523,
      0x0427, 0xFFFF }, { 0x0226, 0x090C, 0x002E, 0x0611, 0x0D09, 0x002F, 0xFFFF }, { 0x0715, 0x0B07, 0x0819, 0x032B, 0x1C0F,
      0xFFFF }, { 0x0310, 0x0531, 0x0701, 0x0803, 0x022D, 0x0A1D, 0xFFFF }, { 0x0418, 0x012C, 0x0433, 0x0525, 0x0137, 0x091B,
      0x130D, 0xFFFF }, { 0x0232, 0x0402, 0x0621, 0x0429, 0xFFFF }, { 0x0228, 0x051C, 0x0713, 0x100B, 0x0C1F, 0xFFFF }, {
      0x0320, 0x0335, 0x0A05, 0x0817, 0xFFFF }, { 0x0623, 0x0527, 0xFFFF }, { 0x0416, 0x061E, 0x0E0E, 0x0711, 0x0E09, 0x012F,
      0xFFFF }, { 0x0322, 0x0504, 0x080A, 0x0919, 0x1D0F, 0xFFFF }, { 0x0631, 0x0903, 0x0815, 0x0C07, 0x042B, 0x032D, 0x0B1D,
      0xFFFF }, { 0x022A, 0x0801, 0x0533, 0x0625, 0x0237, 0x0A1B, 0xFFFF }, { 0x0330, 0x0414, 0x0136, 0x0708, 0x0721, 0x0529,
      0x140D, 0xFFFF }, { 0x0400, 0x0324, 0x0606, 0x0038, 0x0039, 0x110B, 0x0D1F, 0xFFFF }, { 0x051A, 0x0813, 0x0B05, 0x0917,
      0xFFFF }, { 0x0723, 0x0435, 0x0627, 0xFFFF }, { 0x0412, 0x0234, 0x0F09, 0x022F, 0xFFFF }, { 0x0326, 0x0A0C, 0x012E,
      0x0811, 0x0A19, 0x1E0F, 0xFFFF }, { 0x0731, 0x0A03, 0x0915, 0x0D07, 0x052B, 0xFFFF }, { 0x0410, 0x0901, 0x0633, 0x0725,
      0x0337, 0x0B1B, 0x042D, 0x0C1D, 0xFFFF }, { 0x0518, 0x022C, 0x0629, 0x150D, 0xFFFF }, { 0x0332, 0x0502, 0x0821, 0x0139,
      0x120B, 0x0E1F, 0xFFFF }, { 0x0328, 0x061C, 0x0913, 0x0A17, 0xFFFF }, { 0x0420, 0x0535, 0x0C05, 0x0727, 0xFFFF }, {
      0x0823, 0x032F, 0xFFFF }, { 0x0516, 0x071E, 0x0F0E, 0x0911, 0x0B19, 0x1009, 0x1F0F, 0xFFFF }, { 0x0422, 0x0604, 0x090A,
      0x0B03, 0x0A15, 0x0E07, 0x062B, 0xFFFF }, { 0x0831, 0x0A01, 0x0733, 0x052D, 0x0D1D, 0xFFFF }, { 0x032A, 0x0825, 0x0437,
      0x0729, 0x0C1B, 0x160D, 0xFFFF }, { 0x0430, 0x0514, 0x0236, 0x0808, 0x0921, 0x0239, 0x130B, 0x0F1F, 0xFFFF }, { 0x0500,
      0x0424, 0x0706, 0x0138, 0x0A13, 0x0B17, 0xFFFF }, { 0x061A, 0x0635, 0x0D05, 0xFFFF }, { 0x0923, 0x0827, 0xFFFF }, {
      0x0512, 0x0334, 0x003A, 0x0A11, 0x1109, 0x003B, 0x042F, 0xFFFF }, { 0x0426, 0x0B0C, 0x022E, 0x0B15, 0x0F07, 0x0C19,
      0x072B, 0xFFFF }, { 0x0931, 0x0B01, 0x0C03, 0x062D, 0x0E1D, 0xFFFF }, { 0x0510, 0x0833, 0x0925, 0x0537, 0x0D1B, 0x170D,
      0xFFFF }, { 0x0618, 0x032C, 0x0A21, 0x0339, 0x0829, 0xFFFF }, { 0x0432, 0x0602, 0x0B13, 0x140B, 0x101F, 0xFFFF }, {
      0x0428, 0x071C, 0x0735, 0x0E05, 0x0C17, 0xFFFF }, { 0x0520, 0x0A23, 0x0927, 0xFFFF }, { 0x0B11, 0x1209, 0x013B, 0x052F,
      0xFFFF }, { 0x0616, 0x081E, 0x0D19, 0xFFFF }, { 0x0522, 0x0704, 0x0A0A, 0x0A31, 0x0D03, 0x0C15, 0x1007, 0x082B, 0x072D,
      0x0F1D, 0xFFFF }, { 0x0C01, 0x0933, 0x0A25, 0x0637, 0x0E1B, 0xFFFF }, { 0x042A, 0x0B21, 0x0929, 0x180D, 0xFFFF }, {
      0x0530, 0x0614, 0x0336, 0x0908, 0x0439, 0x150B, 0x111F, 0xFFFF }, { 0x0600, 0x0524, 0x0806, 0x0238, 0x0C13, 0x0F05,
      0x0D17, 0xFFFF }, { 0x071A, 0x0B23, 0x0835, 0x0A27, 0xFFFF }, { 0x1309, 0x023B, 0x062F, 0xFFFF }, { 0x0612, 0x0434,
      0x013A, 0x0C11, 0x0E19, 0xFFFF }, { 0x0526, 0x0C0C, 0x032E, 0x0B31, 0x0E03, 0x0D15, 0x1107, 0x092B, 0xFFFF }, { 0x0D01,
      0x0A33, 0x0B25, 0x0737, 0x0F1B, 0x082D, 0x101D, 0xFFFF }, { 0x0610, 0x0A29, 0x190D, 0xFFFF }, { 0x0718, 0x042C, 0x0C21,
      0x0539, 0x160B, 0x121F, 0xFFFF }, { 0x0532, 0x0702, 0x0D13, 0x0E17, 0xFFFF }, { 0x0528, 0x081C, 0x0935, 0x1005, 0x0B27,
      0xFFFF }, { 0x0620, 0x0C23, 0x033B, 0x072F, 0xFFFF }, { 0x0D11, 0x0F19, 0x1409, 0xFFFF }, { 0x0716, 0x003C, 0x091E,
      0x0F03, 0x0E15, 0x1207, 0x0A2B, 0x003D, 0xFFFF }, { 0x0622, 0x0804, 0x0B0A, 0x0C31, 0x0E01, 0x0B33, 0x092D, 0x111D,
      0xFFFF }, { 0x0C25, 0x0837, 0x0B29, 0x101B, 0x1A0D, 0xFFFF }, { 0x052A, 0x0D21, 0x0639, 0x170B, 0x131F, 0xFFFF }, {
      0x0630, 0x0714, 0x0436, 0x0A08, 0x0E13, 0x0F17, 0xFFFF }, { 0x0700, 0x0624, 0x0906, 0x0338, 0x0A35, 0x1105, 0xFFFF }, {
      0x081A, 0x0D23, 0x0C27, 0xFFFF }, { 0x0E11, 0x1509, 0x043B, 0x082F, 0xFFFF }, { 0x0712, 0x0534, 0x023A, 0x0F15, 0x1307,
      0x1019, 0x0B2B, 0x013D, 0xFFFF }, { 0x0626, 0x0D0C, 0x042E, 0x0D31, 0x0F01, 0x1003, 0x0A2D, 0x121D, 0xFFFF }, { 0x0C33,
      0x0D25, 0x0937, 0x111B, 0x1B0D, 0xFFFF }, { 0x0710, 0x0E21, 0x0739, 0x0C29, 0xFFFF }, { 0x0818, 0x052C, 0x0F13, 0x180B,
      0x141F, 0xFFFF }, { 0x0632, 0x0802, 0x0B35, 0x1205, 0x1017, 0xFFFF }, { 0x0628, 0x091C, 0x0E23, 0x0D27, 0xFFFF }, {
      0x0720, 0x0F11, 0x1609, 0x053B, 0x092F, 0xFFFF }, { 0x1119, 0x023D, 0xFFFF }, { 0x0816, 0x013C, 0x0A1E, 0x0E31, 0x1103,
      0x1015, 0x1407, 0x0C2B, 0x0B2D, 0x131D, 0xFFFF }, { 0x0722, 0x0904, 0x0C0A, 0x1001, 0x0D33, 0x0E25, 0x0A37, 0x121B,
      0xFFFF }, { 0x0F21, 0x0D29, 0x1C0D, 0xFFFF }, { 0x062A, 0x0839, 0x190B, 0x151F, 0xFFFF }, { 0x0730, 0x0814, 0x0536,
      0x0B08, 0x1013, 0x1305, 0x1117, 0xFFFF }, { 0x0800, 0x0724, 0x0A06, 0x0438, 0x0F23, 0x0C35, 0x0E27, 0xFFFF }, { 0x091A,
      0x1709, 0x063B, 0x0A2F, 0xFFFF }, { 0x1011, 0x1219, 0x033D, 0xFFFF }, { 0x0812, 0x0634, 0x033A, 0x0F31, 0x1203, 0x1115,
      0x1507, 0x0D2B, 0xFFFF }, { 0x0726, 0x0E0C, 0x052E, 0x1101, 0x0E33, 0x0F25, 0x0B37, 0x131B, 0x0C2D, 0x141D, 0xFFFF }, {
      0x0E29, 0x1D0D, 0xFFFF }, { 0x0810, 0x1021, 0x0939, 0x1A0B, 0x161F, 0xFFFF }, { 0x0918, 0x062C, 0x1113, 0x1217, 0xFFFF
      }, { 0x0732, 0x0902, 0x0D35, 0x1405, 0x0F27, 0xFFFF }, { 0x0728, 0x0A1C, 0x1023, 0x073B, 0x0B2F, 0xFFFF }, { 0x0820,
      0x1111, 0x1319, 0x1809, 0xFFFF }, { 0x1303, 0x1215, 0x1607, 0x0E2B, 0x043D, 0xFFFF }, { 0x0916, 0x023C, 0x0B1E, 0x1031,
      0x1201, 0x0F33, 0x0D2D, 0x151D, 0xFFFF }, { 0x0822, 0x0A04, 0x0D0A, 0x1025, 0x0C37, 0x0F29, 0x141B, 0x1E0D, 0xFFFF }, {
      0x1121, 0x0A39, 0x1B0B, 0x171F, 0xFFFF }, { 0x072A, 0x1213, 0x1317, 0xFFFF }, { 0x0830, 0x0914, 0x0636, 0x0C08, 0x0E35,
      0x1505, 0xFFFF }, { 0x0900, 0x0824, 0x0B06, 0x0538, 0x1123, 0x1027, 0xFFFF }, { 0x0A1A, 0x1211, 0x1909, 0x083B, 0x0C2F,
      0xFFFF }, { 0x1315, 0x1707, 0x1419, 0x0F2B, 0x053D, 0xFFFF }, { 0x0912, 0x0734, 0x043A, 0x1131, 0x1301, 0x1403, 0x0E2D,
      0x161D, 0xFFFF }, { 0x0826, 0x0F0C, 0x062E, 0x1033, 0x1125, 0x0D37, 0x151B, 0x1F0D, 0xFFFF }, { 0x1221, 0x0B39, 0x1029,
      0xFFFF }, { 0x0910, 0x1313, 0x1C0B, 0x181F, 0xFFFF }, { 0x0A18, 0x072C, 0x0F35, 0x1605, 0x1417, 0xFFFF }, { 0x0832,
      0x0A02, 0x1223, 0x1127, 0xFFFF }, { 0x0828, 0x0B1C, 0x1311, 0x1A09, 0x093B, 0x0D2F, 0xFFFF }, { 0x0920, 0x1519, 0x063D,
      0xFFFF }, { 0x1231, 0x1503, 0x1415, 0x1807, 0x102B, 0x0F2D, 0x171D, 0xFFFF }, { 0x0A16, 0x033C, 0x0C1E, 0x1401, 0x1133,
      0x1225, 0x0E37, 0x161B, 0xFFFF }, { 0x0922, 0x0B04, 0x0E0A, 0x1321, 0x1129, 0xFFFF }, { 0x0C39, 0x1D0B, 0x191F, 0xFFFF
      }, { 0x082A, 0x1413, 0x1705, 0x1517, 0xFFFF }, { 0x0930, 0x0A14, 0x0736, 0x0D08, 0x1323, 0x1035, 0x1227, 0xFFFF }, {
      0x0A00, 0x0924, 0x0C06, 0x0638, 0x1B09, 0x0A3B, 0x0E2F, 0xFFFF }, { 0x0B1A, 0x1411, 0x1619, 0x073D, 0xFFFF }, { 0x1331,
      0x1603, 0x1515, 0x1907, 0x112B, 0xFFFF }, { 0x0A12, 0x0834, 0x053A, 0x1501, 0x1233, 0x1325, 0x0F37, 0x171B, 0x102D,
      0x181D, 0xFFFF }, { 0x0926, 0x072E, 0x1229, 0xFFFF }, { 0x1421, 0x0D39, 0x1E0B, 0x1A1F, 0xFFFF }, { 0x0A10, 0x1513,
      0x1617, 0xFFFF }, { 0x0B18, 0x082C, 0x1135, 0x1805, 0x1327, 0xFFFF }, { 0x0932, 0x0B02, 0x1423, 0x0B3B, 0x0F2F, 0xFFFF
      }, { 0x0928, 0x0C1C, 0x1511, 0x1719, 0x1C09, 0xFFFF }, { 0x0A20, 0x1703, 0x1615, 0x1A07, 0x122B, 0x083D, 0xFFFF }, {
      0x1431, 0x1601, 0x1333, 0x112D, 0x191D, 0xFFFF }, { 0x0B16, 0x043C, 0x0D1E, 0x1425, 0x1037, 0x1329, 0x181B, 0xFFFF }, {
      0x0A22, 0x0C04, 0x0F0A, 0x1521, 0x0E39, 0x1F0B, 0x1B1F, 0xFFFF }, { 0x1613, 0x1717, 0xFFFF }, { 0x092A, 0x1235, 0x1905,
      0xFFFF }, { 0x0A30, 0x0B14, 0x0836, 0x0E08, 0x1523, 0x1427, 0xFFFF }, { 0x0B00, 0x0A24, 0x0D06, 0x0738, 0x1611, 0x1D09,
      0x0C3B, 0x102F, 0xFFFF }, { 0x0C1A, 0x1715, 0x1B07, 0x1819, 0x132B, 0x093D, 0xFFFF }, { 0x1531, 0x1701, 0x1803, 0x122D,
      0x1A1D, 0xFFFF }, { 0x0B12, 0x0934, 0x063A, 0x1433, 0x1525, 0x1137, 0x191B, 0xFFFF }, { 0x0A26, 0x003E, 0x082E, 0x1621,
      0x0F39, 0x1429, 0x003F, 0xFFFF }, { 0x1713, 0x1C1F, 0xFFFF }, { 0x0B10, 0x1335, 0x1A05, 0x1817, 0xFFFF }, { 0x0C18,
      0x092C, 0x1623, 0x1527, 0xFFFF }, { 0x0A32, 0x0C02, 0x1711, 0x1E09, 0x0D3B, 0x112F, 0xFFFF }, { 0x0A28, 0x0D1C, 0x1919,
      0x0A3D, 0xFFFF }, { 0x0B20, 0x1631, 0x1903, 0x1815, 0x1C07, 0x142B, 0x132D, 0x1B1D, 0xFFFF }, { 0x1801, 0x1533, 0x1625,
      0x1237, 0x1A1B, 0xFFFF }, { 0x0C16, 0x053C, 0x0E1E, 0x1721, 0x1529, 0x013F, 0xFFFF }, { 0x0B22, 0x0D04, 0x1039, 0x1D1F,
      0xFFFF }, { 0x1813, 0x1B05, 0x1917, 0xFFFF }, { 0x0A2A, 0x1723, 0x1435, 0x1627, 0xFFFF }, { 0x0B30, 0x0C14, 0x0936,
      0x0F08, 0x1F09, 0x0E3B, 0x122F, 0xFFFF }, { 0x0C00, 0x0B24, 0x0E06, 0x0838, 0x1811, 0x1A19, 0x0B3D, 0xFFFF }, { 0x0D1A,
      0x1731, 0x1A03, 0x1915, 0x1D07, 0x152B, 0xFFFF }, { 0x1901, 0x1633, 0x1725, 0x1337, 0x1B1B, 0x142D, 0x1C1D, 0xFFFF }, {
      0x0C12, 0x0A34, 0x073A, 0x1629, 0x023F, 0xFFFF }, { 0x0B26, 0x013E, 0x092E, 0x1821, 0x1139, 0x1E1F, 0xFFFF }, { 0x1913,
      0x1A17, 0xFFFF }, { 0x0C10, 0x1535, 0x1C05, 0x1727, 0xFFFF }, { 0x0D18, 0x0A2C, 0x1823, 0x0F3B, 0x132F, 0xFFFF }, {
      0x0B32, 0x0D02, 0x1911, 0x1B19, 0xFFFF }, { 0x0B28, 0x0E1C, 0x1B03, 0x1A15, 0x1E07, 0x162B, 0x0C3D, 0xFFFF }, { 0x0C20,
      0x1831, 0x1A01, 0x1733, 0x152D, 0x1D1D, 0xFFFF }, { 0x1825, 0x1437, 0x1729, 0x1C1B, 0x033F, 0xFFFF }, { 0x0D16, 0x063C,
      0x0F1E, 0x1921, 0x1239, 0x1F1F, 0xFFFF }, { 0x0C22, 0x0E04, 0x1A13, 0x1B17, 0xFFFF }, { 0x1635, 0x1D05, 0xFFFF }, {
      0x0B2A, 0x1923, 0x1827, 0xFFFF }, { 0x0C30, 0x0D14, 0x0A36, 0x1A11, 0x103B, 0x142F, 0xFFFF }, { 0x0D00, 0x0C24, 0x0F06,
      0x0938, 0x1B15, 0x1F07, 0x1C19, 0x172B, 0x0D3D, 0xFFFF }, { 0x0E1A, 0x1931, 0x1B01, 0x1C03, 0x162D, 0x1E1D, 0xFFFF }, {
      0x1833, 0x1925, 0x1537, 0x1D1B, 0xFFFF }, { 0x0D12, 0x0B34, 0x083A, 0x1A21, 0x1339, 0x1829, 0x043F, 0xFFFF }, { 0x0C26,
      0x023E, 0x0A2E, 0x1B13, 0xFFFF }, { 0x1735, 0x1E05, 0x1C17, 0xFFFF }, { 0x0D10, 0x1A23, 0x1927, 0xFFFF }, { 0x0E18,
      0x0B2C, 0x1B11, 0x113B, 0x152F, 0xFFFF }, { 0x0C32, 0x0E02, 0x1D19, 0x0E3D, 0xFFFF }, { 0x0C28, 0x0F1C, 0x1A31, 0x1D03,
      0x1C15, 0x182B, 0x172D, 0x1F1D, 0xFFFF }, { 0x0D20, 0x1C01, 0x1933, 0x1A25, 0x1637, 0x1E1B, 0xFFFF }, { 0x1B21, 0x1929,
      0x053F, 0xFFFF }, { 0x0E16, 0x073C, 0x1439, 0xFFFF }, { 0x0D22, 0x0F04, 0x1C13, 0x1F05, 0x1D17, 0xFFFF }, { 0x1B23,
      0x1835, 0x1A27, 0xFFFF }, { 0x0C2A, 0x123B, 0x162F, 0xFFFF }, { 0x0D30, 0x0E14, 0x0B36, 0x1C11, 0x1E19, 0x0F3D, 0xFFFF
      }, { 0x0E00, 0x0D24, 0x0A38, 0x1B31, 0x1E03, 0x1D15, 0x192B, 0xFFFF }, { 0x0F1A, 0x1D01, 0x1A33, 0x1B25, 0x1737, 0x1F1B,
      0x182D, 0xFFFF }, { 0x1A29, 0x063F, 0xFFFF }, { 0x0E12, 0x0C34, 0x093A, 0x1C21, 0x1539, 0xFFFF }, { 0x0D26, 0x033E,
      0x0B2E, 0x1D13, 0x1E17, 0xFFFF }, { 0x1935, 0x1B27, 0xFFFF }, { 0x0E10, 0x1C23, 0x133B, 0x172F, 0xFFFF }, { 0x0F18,
      0x0C2C, 0x1D11, 0x1F19, 0xFFFF }, { 0x0D32, 0x0F02, 0x1F03, 0x1E15, 0x1A2B, 0x103D, 0xFFFF }, { 0x0D28, 0x1C31, 0x1E01,
      0x1B33, 0x192D, 0xFFFF }, { 0x0E20, 0x1C25, 0x1837, 0x1B29, 0x073F, 0xFFFF }, { 0x1D21, 0x1639, 0xFFFF }, { 0x0F16,
      0x083C, 0x1E13, 0x1F17, 0xFFFF }, { 0x0E22, 0x1A35, 0xFFFF }, { 0x1D23, 0x1C27, 0xFFFF }, { 0x0D2A, 0x1E11, 0x143B,
      0x182F, 0xFFFF }, { 0x0E30, 0x0F14, 0x0C36, 0x1F15, 0x1B2B, 0x113D, 0xFFFF }, { 0x0F00, 0x0E24, 0x0B38, 0x1D31, 0x1F01,
      0x1A2D, 0xFFFF }, { 0x1C33, 0x1D25, 0x1937, 0xFFFF }, { 0x1E21, 0x1739, 0x1C29, 0x083F, 0xFFFF }, { 0x0F12, 0x0D34,
      0x0A3A, 0x1F13, 0xFFFF }, { 0x0E26, 0x043E, 0x0C2E, 0x1B35, 0xFFFF }, { 0x1E23, 0x1D27, 0xFFFF }, { 0x0F10, 0x1F11,
      0x153B, 0x192F, 0xFFFF }, { 0x0D2C, 0x123D, 0xFFFF },
   };

   struct etc1_block
   {
      // big endian uint64:
      // bit ofs:  56  48  40  32  24  16   8   0
      // byte ofs: b0, b1, b2, b3, b4, b5, b6, b7 
      union 
      {
         uint64 m_uint64;
         uint8 m_bytes[8];
      };

      uint8 m_low_color[2];
      uint8 m_high_color[2];

      enum { cNumSelectorBytes = 4 };
      uint8 m_selectors[cNumSelectorBytes];

      inline void clear()
      {
         zero_this(this);
      }

      inline uint get_byte_bits(uint ofs, uint num) const
      {
         RG_ETC1_ASSERT((ofs + num) <= 64U);
         RG_ETC1_ASSERT(num && (num <= 8U));
         RG_ETC1_ASSERT((ofs >> 3) == ((ofs + num - 1) >> 3));
         const uint byte_ofs = 7 - (ofs >> 3);
         const uint byte_bit_ofs = ofs & 7;
         return (m_bytes[byte_ofs] >> byte_bit_ofs) & ((1 << num) - 1);
      }

      inline void set_byte_bits(uint ofs, uint num, uint bits)
      {
         RG_ETC1_ASSERT((ofs + num) <= 64U);
         RG_ETC1_ASSERT(num && (num < 32U));
         RG_ETC1_ASSERT((ofs >> 3) == ((ofs + num - 1) >> 3));
         RG_ETC1_ASSERT(bits < (1U << num));
         const uint byte_ofs = 7 - (ofs >> 3);
         const uint byte_bit_ofs = ofs & 7;
         const uint mask = (1 << num) - 1;
         m_bytes[byte_ofs] &= ~(mask << byte_bit_ofs);
         m_bytes[byte_ofs] |= (bits << byte_bit_ofs);
      }

      // false = left/right subblocks
      // true = upper/lower subblocks
      inline bool get_flip_bit() const 
      {
         return (m_bytes[3] & 1) != 0;
      }   

      inline void set_flip_bit(bool flip)
      {
         m_bytes[3] &= ~1;
         m_bytes[3] |= static_cast<uint8>(flip);
      }

      inline bool get_diff_bit() const
      {
         return (m_bytes[3] & 2) != 0;
      }

      inline void set_diff_bit(bool diff)
      {
         m_bytes[3] &= ~2;
         m_bytes[3] |= (static_cast<uint>(diff) << 1);
      }

      // Returns intensity modifier table (0-7) used by subblock subblock_id.
      // subblock_id=0 left/top (CW 1), 1=right/bottom (CW 2)
      inline uint get_inten_table(uint subblock_id) const
      {
         RG_ETC1_ASSERT(subblock_id < 2);
         const uint ofs = subblock_id ? 2 : 5;
         return (m_bytes[3] >> ofs) & 7;
      }

      // Sets intensity modifier table (0-7) used by subblock subblock_id (0 or 1)
      inline void set_inten_table(uint subblock_id, uint t)
      {
         RG_ETC1_ASSERT(subblock_id < 2);
         RG_ETC1_ASSERT(t < 8);
         const uint ofs = subblock_id ? 2 : 5;
         m_bytes[3] &= ~(7 << ofs);
         m_bytes[3] |= (t << ofs);
      }

      // Returned selector value ranges from 0-3 and is a direct index into g_etc1_inten_tables.
      inline uint get_selector(uint x, uint y) const
      {
         RG_ETC1_ASSERT((x | y) < 4);

         const uint bit_index = x * 4 + y;
         const uint byte_bit_ofs = bit_index & 7;
         const uint8 *p = &m_bytes[7 - (bit_index >> 3)];
         const uint lsb = (p[0] >> byte_bit_ofs) & 1;
         const uint msb = (p[-2] >> byte_bit_ofs) & 1;
         const uint val = lsb | (msb << 1);

         return g_etc1_to_selector_index[val];
      }

      // Selector "val" ranges from 0-3 and is a direct index into g_etc1_inten_tables.
      inline void set_selector(uint x, uint y, uint val)
      {
         RG_ETC1_ASSERT((x | y | val) < 4);
         const uint bit_index = x * 4 + y;

         uint8 *p = &m_bytes[7 - (bit_index >> 3)];

         const uint byte_bit_ofs = bit_index & 7;
         const uint mask = 1 << byte_bit_ofs;

         const uint etc1_val = g_selector_index_to_etc1[val];

         const uint lsb = etc1_val & 1;
         const uint msb = etc1_val >> 1;

         p[0] &= ~mask;
         p[0] |= (lsb << byte_bit_ofs);

         p[-2] &= ~mask;
         p[-2] |= (msb << byte_bit_ofs);
      }

      inline void set_base4_color(uint idx, uint16 c)
      {
         if (idx)
         {
            set_byte_bits(cETC1AbsColor4R2BitOffset, 4, (c >> 8) & 15);
            set_byte_bits(cETC1AbsColor4G2BitOffset, 4, (c >> 4) & 15);
            set_byte_bits(cETC1AbsColor4B2BitOffset, 4, c & 15);
         }
         else
         {
            set_byte_bits(cETC1AbsColor4R1BitOffset, 4, (c >> 8) & 15);
            set_byte_bits(cETC1AbsColor4G1BitOffset, 4, (c >> 4) & 15);
            set_byte_bits(cETC1AbsColor4B1BitOffset, 4, c & 15);
         }
      }

      inline uint16 get_base4_color(uint idx) const
      {
         uint r, g, b;
         if (idx)
         {
            r = get_byte_bits(cETC1AbsColor4R2BitOffset, 4);
            g = get_byte_bits(cETC1AbsColor4G2BitOffset, 4);
            b = get_byte_bits(cETC1AbsColor4B2BitOffset, 4);
         }
         else
         {
            r = get_byte_bits(cETC1AbsColor4R1BitOffset, 4);
            g = get_byte_bits(cETC1AbsColor4G1BitOffset, 4);
            b = get_byte_bits(cETC1AbsColor4B1BitOffset, 4);
         }
         return static_cast<uint16>(b | (g << 4U) | (r << 8U));
      }

      inline void set_base5_color(uint16 c)
      {
         set_byte_bits(cETC1BaseColor5RBitOffset, 5, (c >> 10) & 31);
         set_byte_bits(cETC1BaseColor5GBitOffset, 5, (c >> 5) & 31);
         set_byte_bits(cETC1BaseColor5BBitOffset, 5, c & 31);
      }

      inline uint16 get_base5_color() const
      {
         const uint r = get_byte_bits(cETC1BaseColor5RBitOffset, 5);
         const uint g = get_byte_bits(cETC1BaseColor5GBitOffset, 5);
         const uint b = get_byte_bits(cETC1BaseColor5BBitOffset, 5);
         return static_cast<uint16>(b | (g << 5U) | (r << 10U));
      }

      void set_delta3_color(uint16 c)
      {
         set_byte_bits(cETC1DeltaColor3RBitOffset, 3, (c >> 6) & 7);
         set_byte_bits(cETC1DeltaColor3GBitOffset, 3, (c >> 3) & 7);
         set_byte_bits(cETC1DeltaColor3BBitOffset, 3, c & 7);
      }

      inline uint16 get_delta3_color() const
      {
         const uint r = get_byte_bits(cETC1DeltaColor3RBitOffset, 3);
         const uint g = get_byte_bits(cETC1DeltaColor3GBitOffset, 3);
         const uint b = get_byte_bits(cETC1DeltaColor3BBitOffset, 3);
         return static_cast<uint16>(b | (g << 3U) | (r << 6U));
      }

      // Base color 5
      static uint16 pack_color5(const color_quad_u8& color, bool scaled, uint bias = 127U);
      static uint16 pack_color5(uint r, uint g, uint b, bool scaled, uint bias = 127U);

      static color_quad_u8 unpack_color5(uint16 packed_color5, bool scaled, uint alpha = 255U);
      static void unpack_color5(uint& r, uint& g, uint& b, uint16 packed_color, bool scaled);

      static bool unpack_color5(color_quad_u8& result, uint16 packed_color5, uint16 packed_delta3, bool scaled, uint alpha = 255U);
      static bool unpack_color5(uint& r, uint& g, uint& b, uint16 packed_color5, uint16 packed_delta3, bool scaled, uint alpha = 255U);

      // Delta color 3
      // Inputs range from -4 to 3 (cETC1ColorDeltaMin to cETC1ColorDeltaMax)
      static uint16 pack_delta3(int r, int g, int b);

      // Results range from -4 to 3 (cETC1ColorDeltaMin to cETC1ColorDeltaMax)
      static void unpack_delta3(int& r, int& g, int& b, uint16 packed_delta3);

      // Abs color 4
      static uint16 pack_color4(const color_quad_u8& color, bool scaled, uint bias = 127U);
      static uint16 pack_color4(uint r, uint g, uint b, bool scaled, uint bias = 127U);

      static color_quad_u8 unpack_color4(uint16 packed_color4, bool scaled, uint alpha = 255U);
      static void unpack_color4(uint& r, uint& g, uint& b, uint16 packed_color4, bool scaled);

      // subblock colors
      static void get_diff_subblock_colors(color_quad_u8* pDst, uint16 packed_color5, uint table_idx);
      static bool get_diff_subblock_colors(color_quad_u8* pDst, uint16 packed_color5, uint16 packed_delta3, uint table_idx);
      static void get_abs_subblock_colors(color_quad_u8* pDst, uint16 packed_color4, uint table_idx);

      static inline void unscaled_to_scaled_color(color_quad_u8& dst, const color_quad_u8& src, bool color4)
      {
         if (color4)
         {
            dst.r = src.r | (src.r << 4);
            dst.g = src.g | (src.g << 4);
            dst.b = src.b | (src.b << 4);
         }
         else
         {
            dst.r = (src.r >> 2) | (src.r << 3);
            dst.g = (src.g >> 2) | (src.g << 3);
            dst.b = (src.b >> 2) | (src.b << 3);
         }
         dst.a = src.a;
      }
   };

   // Returns pointer to sorted array.
   template<typename T, typename Q>
   T* indirect_radix_sort(uint num_indices, T* pIndices0, T* pIndices1, const Q* pKeys, uint key_ofs, uint key_size, bool init_indices)
   {  
      RG_ETC1_ASSERT((key_ofs >= 0) && (key_ofs < sizeof(T)));
      RG_ETC1_ASSERT((key_size >= 1) && (key_size <= 4));

      if (init_indices)
      {
         T* p = pIndices0;
         T* q = pIndices0 + (num_indices >> 1) * 2;
         uint i;
         for (i = 0; p != q; p += 2, i += 2)
         {
            p[0] = static_cast<T>(i);
            p[1] = static_cast<T>(i + 1); 
         }

         if (num_indices & 1)
            *p = static_cast<T>(i);
      }

      uint hist[256 * 4];

      memset(hist, 0, sizeof(hist[0]) * 256 * key_size);

#define RG_ETC1_GET_KEY(p) (*(const uint*)((const uint8*)(pKeys + *(p)) + key_ofs))
#define RG_ETC1_GET_KEY_FROM_INDEX(i) (*(const uint*)((const uint8*)(pKeys + (i)) + key_ofs))

      if (key_size == 4)
      {
         T* p = pIndices0;
         T* q = pIndices0 + num_indices;
         for ( ; p != q; p++)
         {
            const uint key = RG_ETC1_GET_KEY(p);

            hist[        key        & 0xFF]++;
            hist[256 + ((key >>  8) & 0xFF)]++;
            hist[512 + ((key >> 16) & 0xFF)]++;
            hist[768 + ((key >> 24) & 0xFF)]++;
         }
      }
      else if (key_size == 3)
      {
         T* p = pIndices0;
         T* q = pIndices0 + num_indices;
         for ( ; p != q; p++)
         {
            const uint key = RG_ETC1_GET_KEY(p);

            hist[        key        & 0xFF]++;
            hist[256 + ((key >>  8) & 0xFF)]++;
            hist[512 + ((key >> 16) & 0xFF)]++;
         }
      }   
      else if (key_size == 2)
      {
         T* p = pIndices0;
         T* q = pIndices0 + (num_indices >> 1) * 2;

         for ( ; p != q; p += 2)
         {
            const uint key0 = RG_ETC1_GET_KEY(p);
            const uint key1 = RG_ETC1_GET_KEY(p+1);

            hist[        key0         & 0xFF]++;
            hist[256 + ((key0 >>  8) & 0xFF)]++;

            hist[        key1        & 0xFF]++;
            hist[256 + ((key1 >>  8) & 0xFF)]++;
         }

         if (num_indices & 1)
         {
            const uint key = RG_ETC1_GET_KEY(p);

            hist[        key        & 0xFF]++;
            hist[256 + ((key >>  8) & 0xFF)]++;
         }
      }      
      else
      {
         RG_ETC1_ASSERT(key_size == 1);
         if (key_size != 1)
            return NULL;

         T* p = pIndices0;
         T* q = pIndices0 + (num_indices >> 1) * 2;

         for ( ; p != q; p += 2)
         {
            const uint key0 = RG_ETC1_GET_KEY(p);
            const uint key1 = RG_ETC1_GET_KEY(p+1);

            hist[key0 & 0xFF]++;
            hist[key1 & 0xFF]++;
         }

         if (num_indices & 1)
         {
            const uint key = RG_ETC1_GET_KEY(p);

            hist[key & 0xFF]++;
         }
      }      

      T* pCur = pIndices0;
      T* pNew = pIndices1;

      for (uint pass = 0; pass < key_size; pass++)
      {
         const uint* pHist = &hist[pass << 8];

         uint offsets[256];

         uint cur_ofs = 0;
         for (uint i = 0; i < 256; i += 2)
         {
            offsets[i] = cur_ofs;
            cur_ofs += pHist[i];

            offsets[i+1] = cur_ofs;
            cur_ofs += pHist[i+1];
         }

         const uint pass_shift = pass << 3;

         T* p = pCur;
         T* q = pCur + (num_indices >> 1) * 2;

         for ( ; p != q; p += 2)
         {
            uint index0 = p[0];
            uint index1 = p[1];

            uint c0 = (RG_ETC1_GET_KEY_FROM_INDEX(index0) >> pass_shift) & 0xFF;
            uint c1 = (RG_ETC1_GET_KEY_FROM_INDEX(index1) >> pass_shift) & 0xFF;

            if (c0 == c1)
            {
               uint dst_offset0 = offsets[c0];

               offsets[c0] = dst_offset0 + 2;

               pNew[dst_offset0] = static_cast<T>(index0);
               pNew[dst_offset0 + 1] = static_cast<T>(index1);
            }
            else
            {
               uint dst_offset0 = offsets[c0]++;
               uint dst_offset1 = offsets[c1]++;

               pNew[dst_offset0] = static_cast<T>(index0);
               pNew[dst_offset1] = static_cast<T>(index1);
            }
         }

         if (num_indices & 1)
         {
            uint index = *p;
            uint c = (RG_ETC1_GET_KEY_FROM_INDEX(index) >> pass_shift) & 0xFF;

            uint dst_offset = offsets[c];
            offsets[c] = dst_offset + 1;

            pNew[dst_offset] = static_cast<T>(index);
         }

         T* t = pCur;
         pCur = pNew;
         pNew = t;
      }            

      return pCur;
   }

#undef RG_ETC1_GET_KEY
#undef RG_ETC1_GET_KEY_FROM_INDEX

   uint16 etc1_block::pack_color5(const color_quad_u8& color, bool scaled, uint bias)
   {
      return pack_color5(color.r, color.g, color.b, scaled, bias);
   }
   
   uint16 etc1_block::pack_color5(uint r, uint g, uint b, bool scaled, uint bias)
   {
      if (scaled)
      {
         r = (r * 31U + bias) / 255U;
         g = (g * 31U + bias) / 255U;
         b = (b * 31U + bias) / 255U;
      }

      r = rg_etc1::minimum(r, 31U);
      g = rg_etc1::minimum(g, 31U);
      b = rg_etc1::minimum(b, 31U);

      return static_cast<uint16>(b | (g << 5U) | (r << 10U));
   }

   color_quad_u8 etc1_block::unpack_color5(uint16 packed_color5, bool scaled, uint alpha)
   {
      uint b = packed_color5 & 31U;
      uint g = (packed_color5 >> 5U) & 31U;
      uint r = (packed_color5 >> 10U) & 31U;

      if (scaled)
      {
         b = (b << 3U) | (b >> 2U);
         g = (g << 3U) | (g >> 2U);
         r = (r << 3U) | (r >> 2U);
      }

      return color_quad_u8(cNoClamp, r, g, b, rg_etc1::minimum(alpha, 255U));
   }

   void etc1_block::unpack_color5(uint& r, uint& g, uint& b, uint16 packed_color5, bool scaled)
   {
      color_quad_u8 c(unpack_color5(packed_color5, scaled, 0));
      r = c.r;
      g = c.g;
      b = c.b;
   }

   bool etc1_block::unpack_color5(color_quad_u8& result, uint16 packed_color5, uint16 packed_delta3, bool scaled, uint alpha)
   {
      int dc_r, dc_g, dc_b;
      unpack_delta3(dc_r, dc_g, dc_b, packed_delta3);
      
      int b = (packed_color5 & 31U) + dc_b;
      int g = ((packed_color5 >> 5U) & 31U) + dc_g;
      int r = ((packed_color5 >> 10U) & 31U) + dc_r;

      bool success = true;
      if (static_cast<uint>(r | g | b) > 31U)
      {
         success = false;
         r = rg_etc1::clamp<int>(r, 0, 31);
         g = rg_etc1::clamp<int>(g, 0, 31);
         b = rg_etc1::clamp<int>(b, 0, 31);
      }

      if (scaled)
      {
         b = (b << 3U) | (b >> 2U);
         g = (g << 3U) | (g >> 2U);
         r = (r << 3U) | (r >> 2U);
      }

      result.set_noclamp_rgba(r, g, b, rg_etc1::minimum(alpha, 255U));
      return success;
   }

   bool etc1_block::unpack_color5(uint& r, uint& g, uint& b, uint16 packed_color5, uint16 packed_delta3, bool scaled, uint alpha)
   {
      color_quad_u8 result;
      const bool success = unpack_color5(result, packed_color5, packed_delta3, scaled, alpha);
      r = result.r;
      g = result.g;
      b = result.b;
      return success;
   }
     
   uint16 etc1_block::pack_delta3(int r, int g, int b)
   {
      RG_ETC1_ASSERT((r >= cETC1ColorDeltaMin) && (r <= cETC1ColorDeltaMax));
      RG_ETC1_ASSERT((g >= cETC1ColorDeltaMin) && (g <= cETC1ColorDeltaMax));
      RG_ETC1_ASSERT((b >= cETC1ColorDeltaMin) && (b <= cETC1ColorDeltaMax));
      if (r < 0) r += 8;
      if (g < 0) g += 8;
      if (b < 0) b += 8;
      return static_cast<uint16>(b | (g << 3) | (r << 6));
   }
   
   void etc1_block::unpack_delta3(int& r, int& g, int& b, uint16 packed_delta3)
   {
      r = (packed_delta3 >> 6) & 7;
      g = (packed_delta3 >> 3) & 7;
      b = packed_delta3 & 7;
      if (r >= 4) r -= 8;
      if (g >= 4) g -= 8;
      if (b >= 4) b -= 8;
   }

   uint16 etc1_block::pack_color4(const color_quad_u8& color, bool scaled, uint bias)
   {
      return pack_color4(color.r, color.g, color.b, scaled, bias);
   }
   
   uint16 etc1_block::pack_color4(uint r, uint g, uint b, bool scaled, uint bias)
   {
      if (scaled)
      {
         r = (r * 15U + bias) / 255U;
         g = (g * 15U + bias) / 255U;
         b = (b * 15U + bias) / 255U;
      }

      r = rg_etc1::minimum(r, 15U);
      g = rg_etc1::minimum(g, 15U);
      b = rg_etc1::minimum(b, 15U);

      return static_cast<uint16>(b | (g << 4U) | (r << 8U));
   }

   color_quad_u8 etc1_block::unpack_color4(uint16 packed_color4, bool scaled, uint alpha)
   {
      uint b = packed_color4 & 15U;
      uint g = (packed_color4 >> 4U) & 15U;
      uint r = (packed_color4 >> 8U) & 15U;

      if (scaled)
      {
         b = (b << 4U) | b;
         g = (g << 4U) | g;
         r = (r << 4U) | r;
      }

      return color_quad_u8(cNoClamp, r, g, b, rg_etc1::minimum(alpha, 255U));
   }
   
   void etc1_block::unpack_color4(uint& r, uint& g, uint& b, uint16 packed_color4, bool scaled)
   {
      color_quad_u8 c(unpack_color4(packed_color4, scaled, 0));
      r = c.r;
      g = c.g;
      b = c.b;
   }

   void etc1_block::get_diff_subblock_colors(color_quad_u8* pDst, uint16 packed_color5, uint table_idx)
   {
      RG_ETC1_ASSERT(table_idx < cETC1IntenModifierValues);
      const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

      uint r, g, b;
      unpack_color5(r, g, b, packed_color5, true);

      const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

      const int y0 = pInten_modifer_table[0];
      pDst[0].set(ir + y0, ig + y0, ib + y0);

      const int y1 = pInten_modifer_table[1];
      pDst[1].set(ir + y1, ig + y1, ib + y1);

      const int y2 = pInten_modifer_table[2];
      pDst[2].set(ir + y2, ig + y2, ib + y2);

      const int y3 = pInten_modifer_table[3];
      pDst[3].set(ir + y3, ig + y3, ib + y3);
   }
   
   bool etc1_block::get_diff_subblock_colors(color_quad_u8* pDst, uint16 packed_color5, uint16 packed_delta3, uint table_idx)
   {
      RG_ETC1_ASSERT(table_idx < cETC1IntenModifierValues);
      const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

      uint r, g, b;
      bool success = unpack_color5(r, g, b, packed_color5, packed_delta3, true);

      const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

      const int y0 = pInten_modifer_table[0];
      pDst[0].set(ir + y0, ig + y0, ib + y0);

      const int y1 = pInten_modifer_table[1];
      pDst[1].set(ir + y1, ig + y1, ib + y1);

      const int y2 = pInten_modifer_table[2];
      pDst[2].set(ir + y2, ig + y2, ib + y2);

      const int y3 = pInten_modifer_table[3];
      pDst[3].set(ir + y3, ig + y3, ib + y3);

      return success;
   }
   
   void etc1_block::get_abs_subblock_colors(color_quad_u8* pDst, uint16 packed_color4, uint table_idx)
   {
      RG_ETC1_ASSERT(table_idx < cETC1IntenModifierValues);
      const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

      uint r, g, b;
      unpack_color4(r, g, b, packed_color4, true);
      
      const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

      const int y0 = pInten_modifer_table[0];
      pDst[0].set(ir + y0, ig + y0, ib + y0);
      
      const int y1 = pInten_modifer_table[1];
      pDst[1].set(ir + y1, ig + y1, ib + y1);

      const int y2 = pInten_modifer_table[2];
      pDst[2].set(ir + y2, ig + y2, ib + y2);

      const int y3 = pInten_modifer_table[3];
      pDst[3].set(ir + y3, ig + y3, ib + y3);
   }
      
   bool unpack_etc1_block(const void* pETC1_block, unsigned int* pDst_pixels_rgba, bool preserve_alpha)
   {
      color_quad_u8* pDst = reinterpret_cast<color_quad_u8*>(pDst_pixels_rgba);
      const etc1_block& block = *static_cast<const etc1_block*>(pETC1_block);

      const bool diff_flag = block.get_diff_bit();
      const bool flip_flag = block.get_flip_bit();
      const uint table_index0 = block.get_inten_table(0);
      const uint table_index1 = block.get_inten_table(1);

      color_quad_u8 subblock_colors0[4];
      color_quad_u8 subblock_colors1[4];
      bool success = true;

      if (diff_flag)
      {
         const uint16 base_color5 = block.get_base5_color();
         const uint16 delta_color3 = block.get_delta3_color();
         etc1_block::get_diff_subblock_colors(subblock_colors0, base_color5, table_index0);
            
         if (!etc1_block::get_diff_subblock_colors(subblock_colors1, base_color5, delta_color3, table_index1))
            success = false;
      }
      else
      {
         const uint16 base_color4_0 = block.get_base4_color(0);
         etc1_block::get_abs_subblock_colors(subblock_colors0, base_color4_0, table_index0);

         const uint16 base_color4_1 = block.get_base4_color(1);
         etc1_block::get_abs_subblock_colors(subblock_colors1, base_color4_1, table_index1);
      }

      if (preserve_alpha)
      {
         if (flip_flag)
         {
            for (uint y = 0; y < 2; y++)
            {
               pDst[0].set_rgb(subblock_colors0[block.get_selector(0, y)]);
               pDst[1].set_rgb(subblock_colors0[block.get_selector(1, y)]);
               pDst[2].set_rgb(subblock_colors0[block.get_selector(2, y)]);
               pDst[3].set_rgb(subblock_colors0[block.get_selector(3, y)]);
               pDst += 4;
            }

            for (uint y = 2; y < 4; y++)
            {
               pDst[0].set_rgb(subblock_colors1[block.get_selector(0, y)]);
               pDst[1].set_rgb(subblock_colors1[block.get_selector(1, y)]);
               pDst[2].set_rgb(subblock_colors1[block.get_selector(2, y)]);
               pDst[3].set_rgb(subblock_colors1[block.get_selector(3, y)]);
               pDst += 4;
            }
         }
         else
         {
            for (uint y = 0; y < 4; y++)
            {
               pDst[0].set_rgb(subblock_colors0[block.get_selector(0, y)]);
               pDst[1].set_rgb(subblock_colors0[block.get_selector(1, y)]);
               pDst[2].set_rgb(subblock_colors1[block.get_selector(2, y)]);
               pDst[3].set_rgb(subblock_colors1[block.get_selector(3, y)]);
               pDst += 4;
            }
         }
      }
      else 
      {
         if (flip_flag)
         {
            // 0000
            // 0000
            // 1111
            // 1111
            for (uint y = 0; y < 2; y++)
            {
               pDst[0] = subblock_colors0[block.get_selector(0, y)];
               pDst[1] = subblock_colors0[block.get_selector(1, y)];
               pDst[2] = subblock_colors0[block.get_selector(2, y)];
               pDst[3] = subblock_colors0[block.get_selector(3, y)];
               pDst += 4;
            }

            for (uint y = 2; y < 4; y++)
            {
               pDst[0] = subblock_colors1[block.get_selector(0, y)];
               pDst[1] = subblock_colors1[block.get_selector(1, y)];
               pDst[2] = subblock_colors1[block.get_selector(2, y)];
               pDst[3] = subblock_colors1[block.get_selector(3, y)];
               pDst += 4;
            }
         }
         else
         {
            // 0011
            // 0011
            // 0011
            // 0011
            for (uint y = 0; y < 4; y++)
            {
               pDst[0] = subblock_colors0[block.get_selector(0, y)];
               pDst[1] = subblock_colors0[block.get_selector(1, y)];
               pDst[2] = subblock_colors1[block.get_selector(2, y)];
               pDst[3] = subblock_colors1[block.get_selector(3, y)];
               pDst += 4;
            }
         }
      }
      
      return success;
   }

   struct etc1_solution_coordinates
   {
      inline etc1_solution_coordinates() :
      m_unscaled_color(0, 0, 0, 0),
         m_inten_table(0),
         m_color4(false)
      {
      }

      inline etc1_solution_coordinates(uint r, uint g, uint b, uint inten_table, bool color4) : 
      m_unscaled_color(r, g, b, 255),
         m_inten_table(inten_table),
         m_color4(color4)
      {
      }

      inline etc1_solution_coordinates(const color_quad_u8& c, uint inten_table, bool color4) : 
      m_unscaled_color(c),
         m_inten_table(inten_table),
         m_color4(color4)
      {
      }

      inline etc1_solution_coordinates(const etc1_solution_coordinates& other)
      {
         *this = other;
      }

      inline etc1_solution_coordinates& operator= (const etc1_solution_coordinates& rhs)
      {
         m_unscaled_color = rhs.m_unscaled_color;
         m_inten_table = rhs.m_inten_table;
         m_color4 = rhs.m_color4;
         return *this;
      }

      inline void clear()
      {
         m_unscaled_color.clear();
         m_inten_table = 0;
         m_color4 = false;
      }

      inline color_quad_u8 get_scaled_color() const
      {
         int br, bg, bb;
         if (m_color4)
         {
            br = m_unscaled_color.r | (m_unscaled_color.r << 4);
            bg = m_unscaled_color.g | (m_unscaled_color.g << 4);
            bb = m_unscaled_color.b | (m_unscaled_color.b << 4);
         }
         else
         {
            br = (m_unscaled_color.r >> 2) | (m_unscaled_color.r << 3);
            bg = (m_unscaled_color.g >> 2) | (m_unscaled_color.g << 3);
            bb = (m_unscaled_color.b >> 2) | (m_unscaled_color.b << 3);
         }
         return color_quad_u8(br, bg, bb);
      }

      inline void get_block_colors(color_quad_u8* pBlock_colors)
      {
         int br, bg, bb;
         if (m_color4)
         {
            br = m_unscaled_color.r | (m_unscaled_color.r << 4);
            bg = m_unscaled_color.g | (m_unscaled_color.g << 4);
            bb = m_unscaled_color.b | (m_unscaled_color.b << 4);
         }
         else
         {
            br = (m_unscaled_color.r >> 2) | (m_unscaled_color.r << 3);
            bg = (m_unscaled_color.g >> 2) | (m_unscaled_color.g << 3);
            bb = (m_unscaled_color.b >> 2) | (m_unscaled_color.b << 3);
         }
         const int* pInten_table = g_etc1_inten_tables[m_inten_table];
         pBlock_colors[0].set(br + pInten_table[0], bg + pInten_table[0], bb + pInten_table[0]);
         pBlock_colors[1].set(br + pInten_table[1], bg + pInten_table[1], bb + pInten_table[1]);
         pBlock_colors[2].set(br + pInten_table[2], bg + pInten_table[2], bb + pInten_table[2]);
         pBlock_colors[3].set(br + pInten_table[3], bg + pInten_table[3], bb + pInten_table[3]);
      }

      color_quad_u8 m_unscaled_color;
      uint m_inten_table;
      bool m_color4;
   };

   class etc1_optimizer
   {
      etc1_optimizer(const etc1_optimizer&);
      etc1_optimizer& operator= (const etc1_optimizer&);

   public:
      etc1_optimizer()
      {
         clear();
      }

      void clear()
      {
         m_pParams = NULL;
         m_pResult = NULL;
         m_pSorted_luma = NULL;
         m_pSorted_luma_indices = NULL;
      }

      struct params : etc1_pack_params
      {
         params()
         {
            clear();
         }

         params(const etc1_pack_params& base_params) : 
         etc1_pack_params(base_params)
         {
            clear_optimizer_params();
         }

         void clear()
         {
            etc1_pack_params::clear();
            clear_optimizer_params();
         }

         void clear_optimizer_params()
         {
            m_num_src_pixels = 0;
            m_pSrc_pixels = 0;

            m_use_color4 = false;
            static const int s_default_scan_delta[] = { 0 };
            m_pScan_deltas = s_default_scan_delta;
            m_scan_delta_size = 1;

            m_base_color5.clear();
            m_constrain_against_base_color5 = false;
         }

         uint m_num_src_pixels;
         const color_quad_u8* m_pSrc_pixels;

         bool m_use_color4;
         const int* m_pScan_deltas;
         uint m_scan_delta_size;

         color_quad_u8 m_base_color5;
         bool m_constrain_against_base_color5;
      };

      struct results
      {
         uint64 m_error;
         color_quad_u8 m_block_color_unscaled;
         uint m_block_inten_table;
         uint m_n;
         uint8* m_pSelectors;
         bool m_block_color4;

         inline results& operator= (const results& rhs)
         {
            m_block_color_unscaled = rhs.m_block_color_unscaled;
            m_block_color4 = rhs.m_block_color4;
            m_block_inten_table = rhs.m_block_inten_table;
            m_error = rhs.m_error;
            RG_ETC1_ASSERT(m_n == rhs.m_n);
            memcpy(m_pSelectors, rhs.m_pSelectors, rhs.m_n);
            return *this;
         }
      };

      void init(const params& params, results& result);
      bool compute();

   private:      
      struct potential_solution
      {
         potential_solution() : m_coords(), m_error(cUINT64_MAX), m_valid(false)
         {
         }

         etc1_solution_coordinates  m_coords;
         uint8                      m_selectors[8];
         uint64                     m_error;
         bool                       m_valid;

         void clear()
         {
            m_coords.clear();
            m_error = cUINT64_MAX;
            m_valid = false;
         }
      };

      const params* m_pParams;
      results* m_pResult;

      int m_limit;

      vec3F m_avg_color;
      int m_br, m_bg, m_bb;
      uint16 m_luma[8];
      uint32 m_sorted_luma[2][8];
      const uint32* m_pSorted_luma_indices;
      uint32* m_pSorted_luma;

      uint8 m_selectors[8];
      uint8 m_best_selectors[8];

      potential_solution m_best_solution;
      potential_solution m_trial_solution;
      uint8 m_temp_selectors[8];

      bool evaluate_solution(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution);
      bool evaluate_solution_fast(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution);
   };
      
   bool etc1_optimizer::compute()
   {
      const uint n = m_pParams->m_num_src_pixels;
      const int scan_delta_size = m_pParams->m_scan_delta_size;
      
      // Scan through a subset of the 3D lattice centered around the avg block color trying each 3D (555 or 444) lattice point as a potential block color.
      // Each time a better solution is found try to refine the current solution's block color based of the current selectors and intensity table index.
      for (int zdi = 0; zdi < scan_delta_size; zdi++)
      {
         const int zd = m_pParams->m_pScan_deltas[zdi];
         const int mbb = m_bb + zd;
         if (mbb < 0) continue; else if (mbb > m_limit) break;
         
         for (int ydi = 0; ydi < scan_delta_size; ydi++)
         {
            const int yd = m_pParams->m_pScan_deltas[ydi];
            const int mbg = m_bg + yd;
            if (mbg < 0) continue; else if (mbg > m_limit) break;

            for (int xdi = 0; xdi < scan_delta_size; xdi++)
            {
               const int xd = m_pParams->m_pScan_deltas[xdi];
               const int mbr = m_br + xd;
               if (mbr < 0) continue; else if (mbr > m_limit) break;
      
               etc1_solution_coordinates coords(mbr, mbg, mbb, 0, m_pParams->m_use_color4);
               if (m_pParams->m_quality == cHighQuality)
               {
                  if (!evaluate_solution(coords, m_trial_solution, &m_best_solution))
                     continue;
               }
               else
               {
                  if (!evaluate_solution_fast(coords, m_trial_solution, &m_best_solution))
                     continue;
               }
               
               // Now we have the input block, the avg. color of the input pixels, a set of trial selector indices, and the block color+intensity index.
               // Now, for each component, attempt to refine the current solution by solving a simple linear equation. For example, for 4 colors:
               // The goal is:
               // pixel0 - (block_color+inten_table[selector0]) + pixel1 - (block_color+inten_table[selector1]) + pixel2 - (block_color+inten_table[selector2]) + pixel3 - (block_color+inten_table[selector3]) = 0
               // Rearranging this:
               // (pixel0 + pixel1 + pixel2 + pixel3) - (block_color+inten_table[selector0]) - (block_color+inten_table[selector1]) - (block_color+inten_table[selector2]) - (block_color+inten_table[selector3]) = 0
               // (pixel0 + pixel1 + pixel2 + pixel3) - block_color - inten_table[selector0] - block_color-inten_table[selector1] - block_color-inten_table[selector2] - block_color-inten_table[selector3] = 0
               // (pixel0 + pixel1 + pixel2 + pixel3) - 4*block_color - inten_table[selector0] - inten_table[selector1] - inten_table[selector2] - inten_table[selector3] = 0
               // (pixel0 + pixel1 + pixel2 + pixel3) - 4*block_color - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3]) = 0
               // (pixel0 + pixel1 + pixel2 + pixel3)/4 - block_color - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3])/4 = 0
               // block_color = (pixel0 + pixel1 + pixel2 + pixel3)/4 - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3])/4
               // So what this means:
               // optimal_block_color = avg_input - avg_inten_delta
               // So the optimal block color can be computed by taking the average block color and subtracting the current average of the intensity delta.
               // Unfortunately, optimal_block_color must then be quantized to 555 or 444 so it's not always possible to improve matters using this formula.
               // Also, the above formula is for unclamped intensity deltas. The actual implementation takes into account clamping.

               const uint max_refinement_trials = (m_pParams->m_quality == cLowQuality) ? 2 : (((xd | yd | zd) == 0) ? 4 : 2);
               for (uint refinement_trial = 0; refinement_trial < max_refinement_trials; refinement_trial++)
               {
                  const uint8* pSelectors = m_best_solution.m_selectors;
                  const int* pInten_table = g_etc1_inten_tables[m_best_solution.m_coords.m_inten_table];

                  int delta_sum_r = 0, delta_sum_g = 0, delta_sum_b = 0;
                  const color_quad_u8 base_color(m_best_solution.m_coords.get_scaled_color());
                  for (uint r = 0; r < n; r++)
                  {
                     const uint s = *pSelectors++;
                     const int yd = pInten_table[s];
                     // Compute actual delta being applied to each pixel, taking into account clamping.
                     delta_sum_r += rg_etc1::clamp<int>(base_color.r + yd, 0, 255) - base_color.r;
                     delta_sum_g += rg_etc1::clamp<int>(base_color.g + yd, 0, 255) - base_color.g;
                     delta_sum_b += rg_etc1::clamp<int>(base_color.b + yd, 0, 255) - base_color.b;
                  }
                  if ((!delta_sum_r) && (!delta_sum_g) && (!delta_sum_b))
                     break;
                  const float avg_delta_r_f = static_cast<float>(delta_sum_r) / n;
                  const float avg_delta_g_f = static_cast<float>(delta_sum_g) / n;
                  const float avg_delta_b_f = static_cast<float>(delta_sum_b) / n;
                  const int br1 = rg_etc1::clamp<int>(static_cast<uint>((m_avg_color[0] - avg_delta_r_f) * m_limit / 255.0f + .5f), 0, m_limit);
                  const int bg1 = rg_etc1::clamp<int>(static_cast<uint>((m_avg_color[1] - avg_delta_g_f) * m_limit / 255.0f + .5f), 0, m_limit);
                  const int bb1 = rg_etc1::clamp<int>(static_cast<uint>((m_avg_color[2] - avg_delta_b_f) * m_limit / 255.0f + .5f), 0, m_limit);
                  
                  bool skip = false;
                  
                  if ((mbr == br1) && (mbg == bg1) && (mbb == bb1))
                     skip = true;
                  else if ((br1 == m_best_solution.m_coords.m_unscaled_color.r) && (bg1 == m_best_solution.m_coords.m_unscaled_color.g) && (bb1 == m_best_solution.m_coords.m_unscaled_color.b))
                     skip = true;
                  else if ((m_br == br1) && (m_bg == bg1) && (m_bb == bb1))
                     skip = true;

                  if (skip)
                     break;

                  etc1_solution_coordinates coords1(br1, bg1, bb1, 0, m_pParams->m_use_color4);
                  if (m_pParams->m_quality == cHighQuality)
                  {
                     if (!evaluate_solution(coords1, m_trial_solution, &m_best_solution)) 
                        break;
                  }
                  else
                  {
                     if (!evaluate_solution_fast(coords1, m_trial_solution, &m_best_solution))
                        break;
                  }

               }  // refinement_trial

            } // xdi
         } // ydi
      } // zdi

      if (!m_best_solution.m_valid)
      {
         m_pResult->m_error = cUINT32_MAX;
         return false;
      }
      
      const uint8* pSelectors = m_best_solution.m_selectors;

#ifdef RG_ETC1_BUILD_DEBUG
      {
         color_quad_u8 block_colors[4];
         m_best_solution.m_coords.get_block_colors(block_colors);

         const color_quad_u8* pSrc_pixels = m_pParams->m_pSrc_pixels;
         uint64 actual_error = 0;
         for (uint i = 0; i < n; i++)
            actual_error += pSrc_pixels[i].squared_distance_rgb(block_colors[pSelectors[i]]);
         
         RG_ETC1_ASSERT(actual_error == m_best_solution.m_error);
      }
#endif      
      
      m_pResult->m_error = m_best_solution.m_error;

      m_pResult->m_block_color_unscaled = m_best_solution.m_coords.m_unscaled_color;
      m_pResult->m_block_color4 = m_best_solution.m_coords.m_color4;
      
      m_pResult->m_block_inten_table = m_best_solution.m_coords.m_inten_table;
      memcpy(m_pResult->m_pSelectors, pSelectors, n);
      m_pResult->m_n = n;

      return true;
   }

   void etc1_optimizer::init(const params& p, results& r)
   {
      // This version is hardcoded for 8 pixel subblocks.
      RG_ETC1_ASSERT(p.m_num_src_pixels == 8);
      
      m_pParams = &p;
      m_pResult = &r;
                  
      const uint n = 8;
      
      m_limit = m_pParams->m_use_color4 ? 15 : 31;

      vec3F avg_color(0.0f);

      for (uint i = 0; i < n; i++)
      {
         const color_quad_u8& c = m_pParams->m_pSrc_pixels[i];
         const vec3F fc(c.r, c.g, c.b);

         avg_color += fc;

         m_luma[i] = static_cast<uint16>(c.r + c.g + c.b);
         m_sorted_luma[0][i] = i;
      }
      avg_color *= (1.0f / static_cast<float>(n));
      m_avg_color = avg_color;

      m_br = rg_etc1::clamp<int>(static_cast<uint>(m_avg_color[0] * m_limit / 255.0f + .5f), 0, m_limit);
      m_bg = rg_etc1::clamp<int>(static_cast<uint>(m_avg_color[1] * m_limit / 255.0f + .5f), 0, m_limit);
      m_bb = rg_etc1::clamp<int>(static_cast<uint>(m_avg_color[2] * m_limit / 255.0f + .5f), 0, m_limit);

      if (m_pParams->m_quality <= cMediumQuality)
      {
         m_pSorted_luma_indices = indirect_radix_sort(n, m_sorted_luma[0], m_sorted_luma[1], m_luma, 0, sizeof(m_luma[0]), false);
         m_pSorted_luma = m_sorted_luma[0];
         if (m_pSorted_luma_indices == m_sorted_luma[0])
            m_pSorted_luma = m_sorted_luma[1];
      
         for (uint i = 0; i < n; i++)
            m_pSorted_luma[i] = m_luma[m_pSorted_luma_indices[i]];
      }
      
      m_best_solution.m_coords.clear();
      m_best_solution.m_valid = false;
      m_best_solution.m_error = cUINT64_MAX;
   }

   bool etc1_optimizer::evaluate_solution(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution)
   {
      trial_solution.m_valid = false;

      if (m_pParams->m_constrain_against_base_color5)
      {
         const int dr = coords.m_unscaled_color.r - m_pParams->m_base_color5.r;
         const int dg = coords.m_unscaled_color.g - m_pParams->m_base_color5.g;
         const int db = coords.m_unscaled_color.b - m_pParams->m_base_color5.b;

         if ((rg_etc1::minimum(dr, dg, db) < cETC1ColorDeltaMin) || (rg_etc1::maximum(dr, dg, db) > cETC1ColorDeltaMax))
            return false;
      }

      const color_quad_u8 base_color(coords.get_scaled_color());
      
      const uint n = 8;
            
      trial_solution.m_error = cUINT64_MAX;
            
      for (uint inten_table = 0; inten_table < cETC1IntenModifierValues; inten_table++)
      {
         const int* pInten_table = g_etc1_inten_tables[inten_table];

         color_quad_u8 block_colors[4];
         for (uint s = 0; s < 4; s++)
         {
            const int yd = pInten_table[s];
            block_colors[s].set(base_color.r + yd, base_color.g + yd, base_color.b + yd, 0);
         }
         
         uint64 total_error = 0;
         
         const color_quad_u8* pSrc_pixels = m_pParams->m_pSrc_pixels;
         for (uint c = 0; c < n; c++)
         {
            const color_quad_u8& src_pixel = *pSrc_pixels++;
            
            uint best_selector_index = 0;
            uint best_error = rg_etc1::square(src_pixel.r - block_colors[0].r) + rg_etc1::square(src_pixel.g - block_colors[0].g) + rg_etc1::square(src_pixel.b - block_colors[0].b);

            uint trial_error = rg_etc1::square(src_pixel.r - block_colors[1].r) + rg_etc1::square(src_pixel.g - block_colors[1].g) + rg_etc1::square(src_pixel.b - block_colors[1].b);
            if (trial_error < best_error)
            {
               best_error = trial_error;
               best_selector_index = 1;
            }

            trial_error = rg_etc1::square(src_pixel.r - block_colors[2].r) + rg_etc1::square(src_pixel.g - block_colors[2].g) + rg_etc1::square(src_pixel.b - block_colors[2].b);
            if (trial_error < best_error)
            {
               best_error = trial_error;
               best_selector_index = 2;
            }

            trial_error = rg_etc1::square(src_pixel.r - block_colors[3].r) + rg_etc1::square(src_pixel.g - block_colors[3].g) + rg_etc1::square(src_pixel.b - block_colors[3].b);
            if (trial_error < best_error)
            {
               best_error = trial_error;
               best_selector_index = 3;
            }

            m_temp_selectors[c] = static_cast<uint8>(best_selector_index);

            total_error += best_error;
            if (total_error >= trial_solution.m_error)
               break;
         }
         
         if (total_error < trial_solution.m_error)
         {
            trial_solution.m_error = total_error;
            trial_solution.m_coords.m_inten_table = inten_table;
            memcpy(trial_solution.m_selectors, m_temp_selectors, 8);
            trial_solution.m_valid = true;
         }
      }
      trial_solution.m_coords.m_unscaled_color = coords.m_unscaled_color;
      trial_solution.m_coords.m_color4 = m_pParams->m_use_color4;

      bool success = false;
      if (pBest_solution)
      {
         if (trial_solution.m_error < pBest_solution->m_error)
         {
            *pBest_solution = trial_solution;
            success = true;
         }
      }

      return success;
   }

   bool etc1_optimizer::evaluate_solution_fast(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution)
   {
      if (m_pParams->m_constrain_against_base_color5)
      {
         const int dr = coords.m_unscaled_color.r - m_pParams->m_base_color5.r;
         const int dg = coords.m_unscaled_color.g - m_pParams->m_base_color5.g;
         const int db = coords.m_unscaled_color.b - m_pParams->m_base_color5.b;

         if ((rg_etc1::minimum(dr, dg, db) < cETC1ColorDeltaMin) || (rg_etc1::maximum(dr, dg, db) > cETC1ColorDeltaMax))
         {
            trial_solution.m_valid = false;
            return false;
         }
      }

      const color_quad_u8 base_color(coords.get_scaled_color());

      const uint n = 8;
      
      trial_solution.m_error = cUINT64_MAX;

      for (int inten_table = cETC1IntenModifierValues - 1; inten_table >= 0; --inten_table)
      {
         const int* pInten_table = g_etc1_inten_tables[inten_table];

         uint block_inten[4];
         color_quad_u8 block_colors[4];
         for (uint s = 0; s < 4; s++)
         {
            const int yd = pInten_table[s];
            color_quad_u8 block_color(base_color.r + yd, base_color.g + yd, base_color.b + yd, 0);
            block_colors[s] = block_color;
            block_inten[s] = block_color.r + block_color.g + block_color.b;
         }

         // evaluate_solution_fast() enforces/assumesd a total ordering of the input colors along the intensity (1,1,1) axis to more quickly classify the inputs to selectors.
         // The inputs colors have been presorted along the projection onto this axis, and ETC1 block colors are always ordered along the intensity axis, so this classification is fast.
         // 0   1   2   3
         //   01  12  23
         const uint block_inten_midpoints[3] = { block_inten[0] + block_inten[1], block_inten[1] + block_inten[2], block_inten[2] + block_inten[3] };

         uint64 total_error = 0;
         const color_quad_u8* pSrc_pixels = m_pParams->m_pSrc_pixels;
         if ((m_pSorted_luma[n - 1] * 2) < block_inten_midpoints[0])
         {
            if (block_inten[0] > m_pSorted_luma[n - 1])
            {
               const uint min_error = labs(block_inten[0] - m_pSorted_luma[n - 1]);
               if (min_error >= trial_solution.m_error)
                  continue;
            }

            memset(&m_temp_selectors[0], 0, n);

            for (uint c = 0; c < n; c++)
               total_error += block_colors[0].squared_distance_rgb(pSrc_pixels[c]);
         }
         else if ((m_pSorted_luma[0] * 2) >= block_inten_midpoints[2])
         {
            if (m_pSorted_luma[0] > block_inten[3])
            {
               const uint min_error = labs(m_pSorted_luma[0] - block_inten[3]);
               if (min_error >= trial_solution.m_error)
                  continue;
            }

            memset(&m_temp_selectors[0], 3, n);

            for (uint c = 0; c < n; c++)
               total_error += block_colors[3].squared_distance_rgb(pSrc_pixels[c]);
         }
         else
         {
            uint cur_selector = 0, c;
            for (c = 0; c < n; c++)
            {
               const uint y = m_pSorted_luma[c];
               while ((y * 2) >= block_inten_midpoints[cur_selector])
                  if (++cur_selector > 2)
                     goto done;
               const uint sorted_pixel_index = m_pSorted_luma_indices[c];
               m_temp_selectors[sorted_pixel_index] = static_cast<uint8>(cur_selector);
               total_error += block_colors[cur_selector].squared_distance_rgb(pSrc_pixels[sorted_pixel_index]);
            }
done:
            while (c < n)
            {
               const uint sorted_pixel_index = m_pSorted_luma_indices[c];
               m_temp_selectors[sorted_pixel_index] = 3;
               total_error += block_colors[3].squared_distance_rgb(pSrc_pixels[sorted_pixel_index]);
               ++c;
            }
         }

         if (total_error < trial_solution.m_error)
         {
            trial_solution.m_error = total_error;
            trial_solution.m_coords.m_inten_table = inten_table;
            memcpy(trial_solution.m_selectors, m_temp_selectors, n);
            trial_solution.m_valid = true;
            if (!total_error)
               break;
         }
      }
      trial_solution.m_coords.m_unscaled_color = coords.m_unscaled_color;
      trial_solution.m_coords.m_color4 = m_pParams->m_use_color4;
      
      bool success = false;
      if (pBest_solution)
      {
         if (trial_solution.m_error < pBest_solution->m_error)
         {
            *pBest_solution = trial_solution;
            success = true;
         }
      }

      return success;
   }
         
   static uint etc1_decode_value(uint diff, uint inten, uint selector, uint packed_c)
   {
      const uint limit = diff ? 32 : 16; limit;
      RG_ETC1_ASSERT((diff < 2) && (inten < 8) && (selector < 4) && (packed_c < limit));
      int c;
      if (diff)
         c = (packed_c >> 2) | (packed_c << 3);
      else 
         c = packed_c | (packed_c << 4);
      c += g_etc1_inten_tables[inten][selector];
      c = rg_etc1::clamp<int>(c, 0, 255);
      return c;
   }

   static inline int mul_8bit(int a, int b) { int t = a*b + 128; return (t + (t >> 8)) >> 8; }

   void pack_etc1_block_init()
   {
      for (uint diff = 0; diff < 2; diff++)
      {
         const uint limit = diff ? 32 : 16;

         for (uint inten = 0; inten < 8; inten++)
         {
            for (uint selector = 0; selector < 4; selector++)
            {
               const uint inverse_table_index = diff + (inten << 1) + (selector << 4);
               for (uint color = 0; color < 256; color++)
               {
                  uint best_error = cUINT32_MAX, best_packed_c = 0;
                  for (uint packed_c = 0; packed_c < limit; packed_c++)
                  {
                     int v = etc1_decode_value(diff, inten, selector, packed_c);
                     uint err = labs(v - static_cast<int>(color));
                     if (err < best_error)
                     {
                        best_error = err;
                        best_packed_c = packed_c;
                        if (!best_error) 
                           break;
                     }
                  }
                  RG_ETC1_ASSERT(best_error <= 255);
                  g_etc1_inverse_lookup[inverse_table_index][color] = static_cast<uint16>(best_packed_c | (best_error << 8));
               }
            }
         }
      }
      
      uint expand5[32];
      for(int i = 0; i < 32; i++)
         expand5[i] = (i << 3) | (i >> 2);

      for(int i = 0; i < 256 + 16; i++)
      {
         int v = clamp<int>(i - 8, 0, 255);
         g_quant5_tab[i] = static_cast<uint8>(expand5[mul_8bit(v,31)]);
      }
   }

   // Packs solid color blocks efficiently using a set of small precomputed tables.
   // For random 888 inputs, MSE results are better than Erricson's ETC1 packer in "slow" mode ~9.5% of the time, is slightly worse only ~.01% of the time, and is equal the rest of the time.
   static uint64 pack_etc1_block_solid_color(etc1_block& block, const uint8* pColor, etc1_pack_params& pack_params)
   {
      pack_params;
      RG_ETC1_ASSERT(g_etc1_inverse_lookup[0][255]);
            
      static uint s_next_comp[4] = { 1, 2, 0, 1 };
            
      uint best_error = cUINT32_MAX, best_i = 0;
      int best_x = 0, best_packed_c1 = 0, best_packed_c2 = 0;

      // For each possible 8-bit value, there is a precomputed list of diff/inten/selector configurations that allow that 8-bit value to be encoded with no error.
      for (uint i = 0; i < 3; i++)
      {
         const uint c1 = pColor[s_next_comp[i]], c2 = pColor[s_next_comp[i + 1]];

         const int delta_range = 1;
         for (int delta = -delta_range; delta <= delta_range; delta++)
         {
            const int c_plus_delta = rg_etc1::clamp<int>(pColor[i] + delta, 0, 255);

            const uint16* pTable;
            if (!c_plus_delta)
               pTable = g_color8_to_etc_block_config_0_255[0];
            else if (c_plus_delta == 255)
               pTable = g_color8_to_etc_block_config_0_255[1];
            else
               pTable = g_color8_to_etc_block_config_1_to_254[c_plus_delta - 1];

            do
            {
               const uint x = *pTable++;

#ifdef RG_ETC1_BUILD_DEBUG
               const uint diff = x & 1;
               const uint inten = (x >> 1) & 7;
               const uint selector = (x >> 4) & 3;
               const uint p0 = (x >> 8) & 255;
               RG_ETC1_ASSERT(etc1_decode_value(diff, inten, selector, p0) == (uint)c_plus_delta);
#endif

               const uint16* pInverse_table = g_etc1_inverse_lookup[x & 0xFF];
               uint16 p1 = pInverse_table[c1];
               uint16 p2 = pInverse_table[c2];
               const uint trial_error = rg_etc1::square(c_plus_delta - pColor[i]) + rg_etc1::square(p1 >> 8) + rg_etc1::square(p2 >> 8);
               if (trial_error < best_error)
               {
                  best_error = trial_error;
                  best_x = x;
                  best_packed_c1 = p1 & 0xFF;
                  best_packed_c2 = p2 & 0xFF;
                  best_i = i;
                  if (!best_error)
                     goto found_perfect_match;
               }
            } while (*pTable != 0xFFFF);
         }
      }
found_perfect_match:

      const uint diff = best_x & 1;
      const uint inten = (best_x >> 1) & 7;

      block.m_bytes[3] = static_cast<uint8>(((inten | (inten << 3)) << 2) | (diff << 1));
                        
      const uint etc1_selector = g_selector_index_to_etc1[(best_x >> 4) & 3];
      *reinterpret_cast<uint16*>(&block.m_bytes[4]) = (etc1_selector & 2) ? 0xFFFF : 0;
      *reinterpret_cast<uint16*>(&block.m_bytes[6]) = (etc1_selector & 1) ? 0xFFFF : 0;

      const uint best_packed_c0 = (best_x >> 8) & 255;
      if (diff)
      {
         block.m_bytes[best_i] = static_cast<uint8>(best_packed_c0 << 3);
         block.m_bytes[s_next_comp[best_i]] = static_cast<uint8>(best_packed_c1 << 3);
         block.m_bytes[s_next_comp[best_i+1]] = static_cast<uint8>(best_packed_c2 << 3);
      }
      else
      {
         block.m_bytes[best_i] = static_cast<uint8>(best_packed_c0 | (best_packed_c0 << 4));
         block.m_bytes[s_next_comp[best_i]] = static_cast<uint8>(best_packed_c1 | (best_packed_c1 << 4));
         block.m_bytes[s_next_comp[best_i+1]] = static_cast<uint8>(best_packed_c2 | (best_packed_c2 << 4));
      }

      return best_error;
   }
      
   static uint pack_etc1_block_solid_color_constrained(
      etc1_optimizer::results& results, 
      uint num_colors, const uint8* pColor, 
      etc1_pack_params& pack_params, 
      bool use_diff,
      const color_quad_u8* pBase_color5_unscaled)
   {
      RG_ETC1_ASSERT(g_etc1_inverse_lookup[0][255]);

      pack_params;
      static uint s_next_comp[4] = { 1, 2, 0, 1 };

      uint best_error = cUINT32_MAX, best_i = 0;
      int best_x = 0, best_packed_c1 = 0, best_packed_c2 = 0;

      // For each possible 8-bit value, there is a precomputed list of diff/inten/selector configurations that allow that 8-bit value to be encoded with no error.
      for (uint i = 0; i < 3; i++)
      {
         const uint c1 = pColor[s_next_comp[i]], c2 = pColor[s_next_comp[i + 1]];

         const int delta_range = 1;
         for (int delta = -delta_range; delta <= delta_range; delta++)
         {
            const int c_plus_delta = rg_etc1::clamp<int>(pColor[i] + delta, 0, 255);

            const uint16* pTable;
            if (!c_plus_delta)
               pTable = g_color8_to_etc_block_config_0_255[0];
            else if (c_plus_delta == 255)
               pTable = g_color8_to_etc_block_config_0_255[1];
            else
               pTable = g_color8_to_etc_block_config_1_to_254[c_plus_delta - 1];

            do
            {
               const uint x = *pTable++;
               const uint diff = x & 1;
               if (static_cast<uint>(use_diff) != diff)
               {
                  if (*pTable == 0xFFFF)
                     break;
                  continue;
               }

               if ((diff) && (pBase_color5_unscaled))
               {
                  const int p0 = (x >> 8) & 255;
                  int delta = p0 - static_cast<int>(pBase_color5_unscaled->c[i]);
                  if ((delta < cETC1ColorDeltaMin) || (delta > cETC1ColorDeltaMax))
                  {
                     if (*pTable == 0xFFFF)
                        break;
                     continue;
                  }
               }

#ifdef RG_ETC1_BUILD_DEBUG
               {
                  const uint inten = (x >> 1) & 7;
                  const uint selector = (x >> 4) & 3;
                  const uint p0 = (x >> 8) & 255;
                  RG_ETC1_ASSERT(etc1_decode_value(diff, inten, selector, p0) == (uint)c_plus_delta);
               }
#endif

               const uint16* pInverse_table = g_etc1_inverse_lookup[x & 0xFF];
               uint16 p1 = pInverse_table[c1];
               uint16 p2 = pInverse_table[c2];

               if ((diff) && (pBase_color5_unscaled))
               {
                  int delta1 = (p1 & 0xFF) - static_cast<int>(pBase_color5_unscaled->c[s_next_comp[i]]);
                  int delta2 = (p2 & 0xFF) - static_cast<int>(pBase_color5_unscaled->c[s_next_comp[i + 1]]);
                  if ((delta1 < cETC1ColorDeltaMin) || (delta1 > cETC1ColorDeltaMax) || (delta2 < cETC1ColorDeltaMin) || (delta2 > cETC1ColorDeltaMax))
                  {
                     if (*pTable == 0xFFFF)
                        break;
                     continue;
                  }
               }

               const uint trial_error = rg_etc1::square(c_plus_delta - pColor[i]) + rg_etc1::square(p1 >> 8) + rg_etc1::square(p2 >> 8);
               if (trial_error < best_error)
               {
                  best_error = trial_error;
                  best_x = x;
                  best_packed_c1 = p1 & 0xFF;
                  best_packed_c2 = p2 & 0xFF;
                  best_i = i;
                  if (!best_error)
                     goto found_perfect_match;
               }
            } while (*pTable != 0xFFFF);
         }
      }
found_perfect_match:

      if (best_error == cUINT32_MAX)
         return best_error;

      best_error *= num_colors;

      results.m_n = num_colors;
      results.m_block_color4 = !(best_x & 1);
      results.m_block_inten_table = (best_x >> 1) & 7;
      memset(results.m_pSelectors, (best_x >> 4) & 3, num_colors);

      const uint best_packed_c0 = (best_x >> 8) & 255;
      results.m_block_color_unscaled[best_i] = static_cast<uint8>(best_packed_c0);
      results.m_block_color_unscaled[s_next_comp[best_i]] = static_cast<uint8>(best_packed_c1);
      results.m_block_color_unscaled[s_next_comp[best_i + 1]] = static_cast<uint8>(best_packed_c2);
      results.m_error = best_error;
      
      return best_error;
   }

   // Function originally from RYG's public domain real-time DXT1 compressor, modified for 555.
   static void dither_block_555(color_quad_u8* dest, const color_quad_u8* block)
   {
      int err[8],*ep1 = err,*ep2 = err+4;
      uint8 *quant = g_quant5_tab+8;

      memset(dest, 0xFF, sizeof(color_quad_u8)*16);

      // process channels seperately
      for(int ch=0;ch<3;ch++)
      {
         uint8* bp = (uint8*)block;
         uint8* dp = (uint8*)dest;

         bp += ch; dp += ch;

         memset(err,0, sizeof(err));
         for(int y = 0; y < 4; y++)
         {
            // pixel 0
            dp[ 0] = quant[bp[ 0] + ((3*ep2[1] + 5*ep2[0]) >> 4)];
            ep1[0] = bp[ 0] - dp[ 0];

            // pixel 1
            dp[ 4] = quant[bp[ 4] + ((7*ep1[0] + 3*ep2[2] + 5*ep2[1] + ep2[0]) >> 4)];
            ep1[1] = bp[ 4] - dp[ 4];

            // pixel 2
            dp[ 8] = quant[bp[ 8] + ((7*ep1[1] + 3*ep2[3] + 5*ep2[2] + ep2[1]) >> 4)];
            ep1[2] = bp[ 8] - dp[ 8];

            // pixel 3
            dp[12] = quant[bp[12] + ((7*ep1[2] + 5*ep2[3] + ep2[2]) >> 4)];
            ep1[3] = bp[12] - dp[12];

            // advance to next line
            int* tmp = ep1; ep1 = ep2; ep2 = tmp;
            bp += 16;
            dp += 16;
         }
      }
   }

   unsigned int pack_etc1_block(void* pETC1_block, const unsigned int* pSrc_pixels_rgba, etc1_pack_params& pack_params)
   {
      const color_quad_u8* pSrc_pixels = reinterpret_cast<const color_quad_u8*>(pSrc_pixels_rgba);
      etc1_block& dst_block = *static_cast<etc1_block*>(pETC1_block);

#ifdef RG_ETC1_BUILD_DEBUG
      // Ensure all alpha values are 0xFF.
      for (uint i = 0; i < 16; i++)
      {
         RG_ETC1_ASSERT(pSrc_pixels[i].a == 255);
      }
#endif

      color_quad_u8 src_pixel0(pSrc_pixels[0]);

      // Check for solid block.
      const uint32 first_pixel_u32 = pSrc_pixels->m_u32;
      int r;
      for (r = 15; r >= 1; --r)
         if (pSrc_pixels[r].m_u32 != first_pixel_u32)
            break;
      if (!r)
         return static_cast<unsigned int>(16 * pack_etc1_block_solid_color(dst_block, &pSrc_pixels[0].r, pack_params));
      
      color_quad_u8 dithered_pixels[16];
      if (pack_params.m_dithering)
      {
         dither_block_555(dithered_pixels, pSrc_pixels);
         pSrc_pixels = dithered_pixels;
      }

      etc1_optimizer optimizer;

      uint64 best_error = cUINT64_MAX;
      uint best_flip = false, best_use_color4 = false;
      
      uint8 best_selectors[2][8];
      etc1_optimizer::results best_results[2];
      for (uint i = 0; i < 2; i++)
      {
         best_results[i].m_n = 8;
         best_results[i].m_pSelectors = best_selectors[i];
      }
      
      uint8 selectors[3][8];
      etc1_optimizer::results results[3];
      
      for (uint i = 0; i < 3; i++)
      {
         results[i].m_n = 8;
         results[i].m_pSelectors = selectors[i];
      }
            
      color_quad_u8 subblock_pixels[8];

      etc1_optimizer::params params(pack_params);
      params.m_num_src_pixels = 8;
      params.m_pSrc_pixels = subblock_pixels;

      for (uint flip = 0; flip < 2; flip++)
      {
         for (uint use_color4 = 0; use_color4 < 2; use_color4++)
         {
            uint64 trial_error = 0;

            uint subblock;
            for (subblock = 0; subblock < 2; subblock++)
            {
               if (flip)
                  memcpy(subblock_pixels, pSrc_pixels + subblock * 8, sizeof(color_quad_u8) * 8);
               else
               {
                  const color_quad_u8* pSrc_col = pSrc_pixels + subblock * 2;
                  subblock_pixels[0] = pSrc_col[0]; subblock_pixels[1] = pSrc_col[4]; subblock_pixels[2] = pSrc_col[8]; subblock_pixels[3] = pSrc_col[12];
                  subblock_pixels[4] = pSrc_col[1]; subblock_pixels[5] = pSrc_col[5]; subblock_pixels[6] = pSrc_col[9]; subblock_pixels[7] = pSrc_col[13];
               }

               results[2].m_error = cUINT64_MAX;
               if ((params.m_quality >= cMediumQuality) && ((subblock) || (use_color4)))
               {
                  const uint32 subblock_pixel0_u32 = subblock_pixels[0].m_u32;
                  for (r = 7; r >= 1; --r)
                     if (subblock_pixels[r].m_u32 != subblock_pixel0_u32)
                        break;
                  if (!r)
                  {
                     pack_etc1_block_solid_color_constrained(results[2], 8, &subblock_pixels[0].r, pack_params, !use_color4, (subblock && !use_color4) ? &results[0].m_block_color_unscaled : NULL);
                  }
               }

               params.m_use_color4 = (use_color4 != 0);
               params.m_constrain_against_base_color5 = false;

               if ((!use_color4) && (subblock))
               {
                  params.m_constrain_against_base_color5 = true;
                  params.m_base_color5 = results[0].m_block_color_unscaled;
               }
                              
               if (params.m_quality == cHighQuality)
               {
                  static const int s_scan_delta_0_to_4[] = { -4, -3, -2, -1, 0, 1, 2, 3, 4 };
                  params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_0_to_4);
                  params.m_pScan_deltas = s_scan_delta_0_to_4;
               }
               else if (params.m_quality == cMediumQuality)
               {
                  static const int s_scan_delta_0_to_1[] = { -1, 0, 1 };
                  params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_0_to_1);
                  params.m_pScan_deltas = s_scan_delta_0_to_1;
               }
               else
               {
                  static const int s_scan_delta_0[] = { 0 };
                  params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_0);
                  params.m_pScan_deltas = s_scan_delta_0;
               }
               
               optimizer.init(params, results[subblock]);
               if (!optimizer.compute())
                  break;
                              
               if (params.m_quality >= cMediumQuality)
               {
                  // TODO: Fix fairly arbitrary/unrefined thresholds that control how far away to scan for potentially better solutions.
                  const uint refinement_error_thresh0 = 3000;
                  const uint refinement_error_thresh1 = 6000;
                  if (results[subblock].m_error > refinement_error_thresh0)
                  {
                     if (params.m_quality == cMediumQuality)
                     {
                        static const int s_scan_delta_2_to_3[] = { -3, -2, 2, 3 };
                        params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_2_to_3);
                        params.m_pScan_deltas = s_scan_delta_2_to_3;
                     }
                     else
                     {
                        static const int s_scan_delta_5_to_5[] = { -5, 5 };
                        static const int s_scan_delta_5_to_8[] = { -8, -7, -6, -5, 5, 6, 7, 8 };
                        if (results[subblock].m_error > refinement_error_thresh1)
                        {
                           params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_5_to_8);
                           params.m_pScan_deltas = s_scan_delta_5_to_8;
                        }
                        else
                        {
                           params.m_scan_delta_size = RG_ETC1_ARRAY_SIZE(s_scan_delta_5_to_5);
                           params.m_pScan_deltas = s_scan_delta_5_to_5;
                        }
                     }

                     if (!optimizer.compute())
                        break;
                  }

                  if (results[2].m_error < results[subblock].m_error)
                     results[subblock] = results[2];
               }
                            
               trial_error += results[subblock].m_error;
               if (trial_error >= best_error)
                  break;
            }

            if (subblock < 2)
               continue;

            best_error = trial_error;
            best_results[0] = results[0];
            best_results[1] = results[1];
            best_flip = flip;
            best_use_color4 = use_color4;
            
         } // use_color4

      } // flip

      int dr = best_results[1].m_block_color_unscaled.r - best_results[0].m_block_color_unscaled.r;
      int dg = best_results[1].m_block_color_unscaled.g - best_results[0].m_block_color_unscaled.g;
      int db = best_results[1].m_block_color_unscaled.b - best_results[0].m_block_color_unscaled.b;
      RG_ETC1_ASSERT(best_use_color4 || (rg_etc1::minimum(dr, dg, db) >= cETC1ColorDeltaMin) && (rg_etc1::maximum(dr, dg, db) <= cETC1ColorDeltaMax));
           
      if (best_use_color4)
      {
         dst_block.m_bytes[0] = static_cast<uint8>(best_results[1].m_block_color_unscaled.r | (best_results[0].m_block_color_unscaled.r << 4));
         dst_block.m_bytes[1] = static_cast<uint8>(best_results[1].m_block_color_unscaled.g | (best_results[0].m_block_color_unscaled.g << 4));
         dst_block.m_bytes[2] = static_cast<uint8>(best_results[1].m_block_color_unscaled.b | (best_results[0].m_block_color_unscaled.b << 4));
      }
      else
      {
         if (dr < 0) dr += 8; dst_block.m_bytes[0] = static_cast<uint8>((best_results[0].m_block_color_unscaled.r << 3) | dr);
         if (dg < 0) dg += 8; dst_block.m_bytes[1] = static_cast<uint8>((best_results[0].m_block_color_unscaled.g << 3) | dg);
         if (db < 0) db += 8; dst_block.m_bytes[2] = static_cast<uint8>((best_results[0].m_block_color_unscaled.b << 3) | db);
      }
      
      dst_block.m_bytes[3] = static_cast<uint8>( (best_results[1].m_block_inten_table << 2) | (best_results[0].m_block_inten_table << 5) | ((~best_use_color4 & 1) << 1) | best_flip );
      
      uint selector0 = 0, selector1 = 0;
      if (best_flip)
      {
         // flipped:
         // { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 },               
         // { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } 
         //
         // { 0, 2 }, { 1, 2 }, { 2, 2 }, { 3, 2 },
         // { 0, 3 }, { 1, 3 }, { 2, 3 }, { 3, 3 }
         const uint8* pSelectors0 = best_results[0].m_pSelectors;
         const uint8* pSelectors1 = best_results[1].m_pSelectors;
         for (int x = 3; x >= 0; --x)
         {
            uint b;
            b = g_selector_index_to_etc1[pSelectors1[4 + x]];
            selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

            b = g_selector_index_to_etc1[pSelectors1[x]];
            selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

            b = g_selector_index_to_etc1[pSelectors0[4 + x]];
            selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

            b = g_selector_index_to_etc1[pSelectors0[x]];
            selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);
         }
      }
      else
      {
         // non-flipped:
         // { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 },
         // { 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 }
         //
         // { 2, 0 }, { 2, 1 }, { 2, 2 }, { 2, 3 },
         // { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }
         for (int subblock = 1; subblock >= 0; --subblock)
         {
            const uint8* pSelectors = best_results[subblock].m_pSelectors + 4;
            for (uint i = 0; i < 2; i++)
            {
               uint b;
               b = g_selector_index_to_etc1[pSelectors[3]];
               selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

               b = g_selector_index_to_etc1[pSelectors[2]];
               selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

               b = g_selector_index_to_etc1[pSelectors[1]];
               selector0 = (selector0 << 1) | (b & 1); selector1 = (selector1 << 1) | (b >> 1);

               b = g_selector_index_to_etc1[pSelectors[0]];
               selector0 = (selector0 << 1) | (b & 1);selector1 = (selector1 << 1) | (b >> 1);

               pSelectors -= 4;
            }
         }
      }
                  
      dst_block.m_bytes[4] = static_cast<uint8>(selector1 >> 8); dst_block.m_bytes[5] = static_cast<uint8>(selector1 & 0xFF);
      dst_block.m_bytes[6] = static_cast<uint8>(selector0 >> 8); dst_block.m_bytes[7] = static_cast<uint8>(selector0 & 0xFF);

      return static_cast<unsigned int>(best_error);
   }

} // namespace rg_etc1
