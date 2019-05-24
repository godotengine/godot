SamplerState       g_sSamp : register(s0);

uniform Texture1D <float4> g_tTex1df4 : register(t0);
Texture1D <int4>   g_tTex1di4;
Texture1D <uint4>  g_tTex1du4;

Texture2D <float4> g_tTex2df4;
Texture2D <int4>   g_tTex2di4;
Texture2D <uint4>  g_tTex2du4;

Texture3D <float4> g_tTex3df4;
Texture3D <int4>   g_tTex3di4;
Texture3D <uint4>  g_tTex3du4;

TextureCube <float4> g_tTexcdf4;
TextureCube <int4>   g_tTexcdi4;
TextureCube <uint4>  g_tTexcdu4;

Texture1DArray <float4> g_tTex1df4a;
Texture1DArray <int4>   g_tTex1di4a;
Texture1DArray <uint4>  g_tTex1du4a;

Texture2DArray <float4> g_tTex2df4a;
Texture2DArray <int4>   g_tTex2di4a;
Texture2DArray <uint4>  g_tTex2du4a;

TextureCubeArray <float4> g_tTexcdf4a;
TextureCubeArray <int4>   g_tTexcdi4a;
TextureCubeArray <uint4>  g_tTexcdu4a;

Texture2DMS <float4> g_tTex2dmsf4;
Texture2DMS <int4>   g_tTex2dmsi4;
Texture2DMS <uint4>  g_tTex2dmsu4;

Texture2DMSArray <float4> g_tTex2dmsf4a;
Texture2DMSArray <int4>   g_tTex2dmsi4a;
Texture2DMSArray <uint4>  g_tTex2dmsu4a;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   uint MipLevel;
   uint WidthU;
   uint HeightU;
   uint ElementsU;
   uint DepthU;
   uint NumberOfLevelsU;
   uint NumberOfSamplesU;

   float WidthF;
   float HeightF;
   float ElementsF;
   float DepthF;
   float NumberOfLevelsF;
   float NumberOfSamplesF;

   // 1D, float tx, uint params
   g_tTex1df4 . GetDimensions(WidthU);
   g_tTex1df4 . GetDimensions(6, WidthU, NumberOfLevelsU);

   // 1D, int, uint params
   g_tTex1di4 . GetDimensions(WidthU);
   g_tTex1di4 . GetDimensions(6, WidthU, NumberOfLevelsU);

   // 1D, uint, uint params
   g_tTex1du4 . GetDimensions(WidthU);
   g_tTex1du4 . GetDimensions(6, WidthU, NumberOfLevelsU);

   // 1DArray, float tx, uint params
   g_tTex1df4a . GetDimensions(WidthU, ElementsU);
   g_tTex1df4a . GetDimensions(6, WidthU, ElementsU, NumberOfLevelsU);

   // 1DArray, int, uint params
   g_tTex1di4a . GetDimensions(WidthU, ElementsU);
   g_tTex1di4a . GetDimensions(6, WidthU, ElementsU, NumberOfLevelsU);

   // 1DArray, uint, uint params
   g_tTex1du4a . GetDimensions(WidthU, ElementsU);
   g_tTex1du4a . GetDimensions(6, WidthU, ElementsU, NumberOfLevelsU);

   // 2D, float tx, uint params
   g_tTex2df4 . GetDimensions(WidthU, HeightU);
   g_tTex2df4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // 2D, int, uint params
   g_tTex2di4 . GetDimensions(WidthU, HeightU);
   g_tTex2di4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // 2D, uint, uint params
   g_tTex2du4 . GetDimensions(WidthU, HeightU);
   g_tTex2du4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // 2Darray, float tx, uint params
   g_tTex2df4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTex2df4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // 2Darray, int, uint params
   g_tTex2di4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTex2di4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // 2Darray, uint, uint params
   g_tTex2du4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTex2du4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // 3D, float tx, uint params
   g_tTex3df4 . GetDimensions(WidthU, HeightU, DepthU);
   g_tTex3df4 . GetDimensions(6, WidthU, HeightU, DepthU, NumberOfLevelsU);

   // 3D, int, uint params
   g_tTex3di4 . GetDimensions(WidthU, HeightU, DepthU);
   g_tTex3di4 . GetDimensions(6, WidthU, HeightU, DepthU, NumberOfLevelsU);

   // 3D, uint, uint params
   g_tTex3du4 . GetDimensions(WidthU, HeightU, DepthU);
   g_tTex3du4 . GetDimensions(6, WidthU, HeightU, DepthU, NumberOfLevelsU);

   // Cube, float tx, uint params
   g_tTexcdf4 . GetDimensions(WidthU, HeightU);
   g_tTexcdf4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // Cube, int, uint params
   g_tTexcdi4 . GetDimensions(WidthU, HeightU);
   g_tTexcdi4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // Cube, uint, uint params
   g_tTexcdu4 . GetDimensions(WidthU, HeightU);
   g_tTexcdu4 . GetDimensions(6, WidthU, HeightU, NumberOfLevelsU);

   // Cubearray, float tx, uint params
   g_tTexcdf4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTexcdf4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // Cubearray, int, uint params
   g_tTexcdi4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTexcdi4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // Cubearray, uint, uint params
   g_tTexcdu4a . GetDimensions(WidthU, HeightU, ElementsU);
   g_tTexcdu4a . GetDimensions(6, WidthU, HeightU, ElementsU, NumberOfLevelsU);

   // 2DMS, float tx, uint params
   g_tTex2dmsf4 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);

   // 2DMS, int tx, uint params
   g_tTex2dmsi4 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);

   // 2DMS, uint tx, uint params
   g_tTex2dmsu4 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);

   // 2DMSArray, float tx, uint params
   g_tTex2dmsf4a . GetDimensions(WidthU, HeightU, ElementsU, NumberOfSamplesU);

   // 2DMSArray, int tx, uint params
   g_tTex2dmsi4a . GetDimensions(WidthU, HeightU, ElementsU, NumberOfSamplesU);

   // 2DMSArray, uint tx, uint params
   g_tTex2dmsu4a . GetDimensions(WidthU, HeightU, ElementsU, NumberOfSamplesU);

   // TODO: ***************************************************
   // Change this to 1 to enable float overloads when the HLSL
   // function overload resolution is fixed.
#define OVERLOAD_FIX 0

   // TODO: enable when function overload resolution rules are fixed
#if OVERLOAD_FIX
   // 1D, float tx, float params
   g_tTex1df4 . GetDimensions(WidthF);
   g_tTex1df4 . GetDimensions(6, WidthF, NumberOfLevelsF);

   // 1D, int, float params
   g_tTex1di4 . GetDimensions(WidthF);
   g_tTex1di4 . GetDimensions(6, WidthF, NumberOfLevelsF);

   // 1D, uint, float params
   g_tTex1du4 . GetDimensions(WidthF);
   g_tTex1du4 . GetDimensions(6, WidthF, NumberOfLevelsF);

   // 1DArray, float tx, float params
   g_tTex1df4a . GetDimensions(WidthF, ElementsF);
   g_tTex1df4a . GetDimensions(6, WidthF, ElementsF, NumberOfLevelsF);

   // 1DArray, int, float params
   g_tTex1di4a . GetDimensions(WidthF, ElementsF);
   g_tTex1di4a . GetDimensions(6, WidthF, ElementsF, NumberOfLevelsF);

   // 1DArray, uint, float params
   g_tTex1du4a . GetDimensions(WidthF, ElementsF);
   g_tTex1du4a . GetDimensions(6, WidthF, ElementsF, NumberOfLevelsF);

   // 2D, float tx, float params
   g_tTex2df4 . GetDimensions(WidthF, HeightF);
   g_tTex2df4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // 2D, int, float params
   g_tTex2di4 . GetDimensions(WidthF, HeightF);
   g_tTex2di4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // 2D, uint, float params
   g_tTex2du4 . GetDimensions(WidthF, HeightF);
   g_tTex2du4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // 2Darray, float tx, float params
   g_tTex2df4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTex2df4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // 2Darray, int, float params
   g_tTex2di4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTex2di4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // 2Darray, uint, float params
   g_tTex2du4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTex2du4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // 3D, float tx, float params
   g_tTex3df4 . GetDimensions(WidthF, HeightF, DepthF);
   g_tTex3df4 . GetDimensions(6, WidthF, HeightF, DepthF, NumberOfLevelsF);

   // 3D, int, float params
   g_tTex3di4 . GetDimensions(WidthF, HeightF, DepthF);
   g_tTex3di4 . GetDimensions(6, WidthF, HeightF, DepthF, NumberOfLevelsF);

   // 3D, uint, float params
   g_tTex3du4 . GetDimensions(WidthF, HeightF, DepthF);
   g_tTex3du4 . GetDimensions(6, WidthF, HeightF, DepthF, NumberOfLevelsF);

   // Cube, float tx, float params
   g_tTexcdf4 . GetDimensions(WidthF, HeightF);
   g_tTexcdf4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // Cube, int, float params
   g_tTexcdi4 . GetDimensions(WidthF, HeightF);
   g_tTexcdi4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // Cube, uint, float params
   g_tTexcdu4 . GetDimensions(WidthF, HeightF);
   g_tTexcdu4 . GetDimensions(6, WidthF, HeightF, NumberOfLevelsF);

   // Cubearray, float tx, float params
   g_tTexcdf4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTexcdf4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // Cubearray, int, float params
   g_tTexcdi4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTexcdi4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // Cubearray, uint, float params
   g_tTexcdu4a . GetDimensions(WidthF, HeightF, ElementsF);
   g_tTexcdu4a . GetDimensions(6, WidthF, HeightF, ElementsF, NumberOfLevelsF);

   // 2DMS, float tx, uint params
   g_tTex2dmsf4 . GetDimensions(WidthF, HeightF, NumberOfSamplesF);

   // 2DMS, int tx, uint params
   g_tTex2dmsi4 . GetDimensions(WidthF, HeightF, NumberOfSamplesF);

   // 2DMS, uint tx, uint params
   g_tTex2dmsu4 . GetDimensions(WidthF, HeightF, NumberOfSamplesF);

   // 2DMSArray, float tx, uint params
   g_tTex2dmsf4a . GetDimensions(WidthF, HeightF, ElementsF, NumberOfSamplesF);

   // 2DMSArray, int tx, uint params
   g_tTex2dmsi4a . GetDimensions(WidthF, HeightF, ElementsF, NumberOfSamplesF);

   // 2DMSArray, uint tx, uint params
   g_tTex2dmsu4a . GetDimensions(WidthF, HeightF, ElementsF, NumberOfSamplesF);
#endif // OVERLOAD_FIX

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}
