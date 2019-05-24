#version 300 es
in uvec2 badu;  // ERROR
flat in uvec2 t;
in highp float f;
in highp vec2 tc;
in bool bad;    // ERROR
uniform uvec4 v;
uniform int i;
uniform bool b;

out uvec4 c;

uniform lowp usampler2D usampler;

void main()
{
    int count = 1;

    uint u = t.y + 3u;
    const uint cu1error = 0xFFFFFFFF;  // ERROR
    const uint cu1 = 0xFFFFFFFFU;
    const uint cu2 = -1u;              // 0xFFFFFFFF
    const uint cu3 = 1U;
    const uint cu4error = 1;           // ERROR
    const uint cu4 = 1u;

    if (cu1 == cu2)
        count *= 2;  // done
    if (cu3 == cu4)
        count *= 3;  // done
    if (cu2 == cu3)
        count *= 5;  // not done

    const uint cushiftediierror = 0xFFFFFFFF  >> 10;   // ERROR
    const  int  cshiftedii      = 0xFFFFFFFF  >> 10;
    const uint cushiftedui      = 0xFFFFFFFFu >> 10;
    const uint cushiftediuerror = 0xFFFFFFFF  >> 10u;  // ERROR
    const  int  cshiftediu      = 0xFFFFFFFF  >> 10u;
    const uint cushifteduu      = 0xFFFFFFFFu >> 10u;

    if (cshiftedii == cshiftediu)
        count *= 7;  // done
    if (cushiftedui == cushifteduu)
        count *= 11; // done
    if (cshiftedii == int(cushiftedui))
        count *= 13; // not done

    uint shiftediierror = 0xFFFFFFFF  >> 10;   // ERROR
     int shiftedii      = 0xFFFFFFFF  >> 10;
    uint shiftedui      = 0xFFFFFFFFu >> 10;
    uint shiftediuerror = 0xFFFFFFFF  >> 10u;  // ERROR
     int shiftediu      = 0xFFFFFFFF  >> 10u;
    uint shifteduu      = 0xFFFFFFFFu >> 10u;

    if (shiftedii == shiftediu)
        c = texture(usampler, tc);
    if (shiftedui == shifteduu)
        c = texture(usampler, tc + float(1u));
    if (shiftedii == int(shiftedui))
        c = texture(usampler, tc - vec2(2u));

    if (t.x > 4u) {
        float af = float(u);
        bool ab = bool(u);
        int ai = int(u);

        c += uvec4(uint(af), uint(ab), uint(ai), count);
    }

    const uint cmask1 = 0x0A1u;
    const uint cmask2 = 0xA10u;
    const uint cmask3 = cmask1 << 4;
    const uint cmask4 = 0xAB1u;

    if (cmask3 == cmask2)
        count *= 17;  // done

    if ((cmask3 & cmask1) != 0u)
        count *= 19; // not done

    if ((cmask1 | cmask3) == cmask4)
        count *= 23; // done

    if ((cmask1 ^ cmask4) == 0xA10u)
        count *= 27; // done

    uint mask1 = 0x0A1u;
    uint mask2 = 0xA10u;
    uint mask3 = mask1 << 4;
    uint mask4 = 0xAB1u;

    if (mask3 == mask2)
        count *= 100;

    if ((mask3 & mask1) != 0u)
        count *= 101;

    if ((mask1 | mask3) == mask4)
        count *= 102;

    if ((mask1 ^ mask4) == 0xA10u)
        count *= 103;

    c += uvec4(count);	
}
