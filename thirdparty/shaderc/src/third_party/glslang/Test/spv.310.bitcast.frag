#version 310 es

flat in mediump int   i1;
flat in lowp    ivec2 i2;
flat in mediump ivec3 i3;
flat in highp   ivec4 i4;

flat in mediump uint  u1;
flat in lowp    uvec2 u2;
flat in mediump uvec3 u3;
flat in highp   uvec4 u4;

mediump in float f1;
lowp    in vec2  f2;
mediump in vec3  f3;
highp   in vec4  f4;

void main()
{
	highp ivec4 idata = ivec4(0);
	idata.x     += floatBitsToInt(f1);
	idata.xy    += floatBitsToInt(f2);
	idata.xyz   += floatBitsToInt(f3);
	idata       += floatBitsToInt(f4);

	highp uvec4 udata = uvec4(0);
	udata.x     += floatBitsToUint(f1);
	udata.xy    += floatBitsToUint(f2);
	udata.xyz   += floatBitsToUint(f3);
	udata       += floatBitsToUint(f4);

	highp vec4 fdata = vec4(0.0);
	fdata.x     += intBitsToFloat(i1);
	fdata.xy    += intBitsToFloat(i2);
	fdata.xyz   += intBitsToFloat(i3);
	fdata       += intBitsToFloat(i4);
    fdata.x     += uintBitsToFloat(u1);
	fdata.xy    += uintBitsToFloat(u2);
	fdata.xyz   += uintBitsToFloat(u3);
	fdata       += uintBitsToFloat(u4);
}