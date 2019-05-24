#version 450

flat in int   i1;
flat in ivec2 i2;
flat in ivec3 i3;
flat in ivec4 i4;

flat in uint  u1;
flat in uvec2 u2;
flat in uvec3 u3;
flat in uvec4 u4;

in float f1;
in vec2  f2;
in vec3  f3;
in vec4  f4;

out vec4 fragColor;

void main()
{
	ivec4 idata = ivec4(0);
	idata.x     += floatBitsToInt(f1);
	idata.xy    += floatBitsToInt(f2);
	idata.xyz   += floatBitsToInt(f3);
	idata       += floatBitsToInt(f4);

	uvec4 udata = uvec4(0);
	udata.x     += floatBitsToUint(f1);
	udata.xy    += floatBitsToUint(f2);
	udata.xyz   += floatBitsToUint(f3);
	udata       += floatBitsToUint(f4);

	vec4 fdata = vec4(0.0);
	fdata.x     += intBitsToFloat(i1);
	fdata.xy    += intBitsToFloat(i2);
	fdata.xyz   += intBitsToFloat(i3);
	fdata       += intBitsToFloat(i4);
    fdata.x     += uintBitsToFloat(u1);
	fdata.xy    += uintBitsToFloat(u2);
	fdata.xyz   += uintBitsToFloat(u3);
	fdata       += uintBitsToFloat(u4);

    fragColor = (idata == udata) ? fdata : fdata + vec4(0.2);
}