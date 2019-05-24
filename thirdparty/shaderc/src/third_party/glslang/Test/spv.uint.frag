#version 310 es
precision mediump float;
flat in uvec2 t;
in float f;
in vec2 tc;

flat in uvec4 v;
flat in int i;
bool b;

out uvec4 c;

uniform mediump usampler2D usampler;

void main()
{
    int count = 1;

    uint u = t.y + 3u;
    const uint cu1 = 0xFFFFFFFFU;
    const uint cu2 = -1u;              // 0xFFFFFFFF
    const uint cu3 = 1U;
    const uint cu4 = 1u;

    if (cu1 == cu2)
        count *= 2;  // 2
    if (cu3 == cu4)
        count *= 3;  // 6
    if (cu2 == cu3)
        count *= 5;  // not done

    const  int  cshiftedii      = 0xFFFFFFFF  >> 10;
    const uint cushiftedui      = 0xFFFFFFFFu >> 10;
    const  int  cshiftediu      = 0xFFFFFFFF  >> 10u;
    const uint cushifteduu      = 0xFFFFFFFFu >> 10u;

    if (cshiftedii == cshiftediu)
        count *= 7;  // 42
    if (cushiftedui == cushifteduu)
        count *= 11; // 462
    if (cshiftedii == int(cushiftedui))
        count *= 13; // not done

     int shiftedii      = 0xFFFFFFFF  >> 10;
    uint shiftedui      = 0xFFFFFFFFu >> 10;
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
        count *= 17;  // 7854

    if ((cmask3 & cmask1) != 0u)
        count *= 19; // not done

    if ((cmask1 | cmask3) == cmask4)
        count *= 23; // 180642

    if ((cmask1 ^ cmask4) == 0xA10u)
        count *= 27; // 4877334

    uint mask1 = 0x0A1u;
    uint mask2 = 0xA10u;
    uint mask3 = mask1 << 4;
    uint mask4 = 0xAB1u;

    if (mask3 == mask2)
        count *= 2;  // 9754668

    if ((mask3 & mask1) != 0u)
        count *= 3;  // not done

    if ((mask1 | mask3) == mask4)
        count *= 5;  // 48773340

    if ((mask1 ^ mask4) == 0xA10u)
        count *= 7;  // 341413380

    c += uvec4(count);

    #define UINT_MAX  4294967295u
    c.x += UINT_MAX;
}
