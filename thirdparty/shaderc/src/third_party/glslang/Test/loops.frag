#version 130
uniform vec4 bigColor;
uniform vec4 bigColor1_1;
uniform vec4 bigColor1_2;
uniform vec4 bigColor1_3;
uniform vec4 bigColor2;
uniform vec4 bigColor3;
uniform vec4 bigColor4;
uniform vec4 bigColor5;
uniform vec4 bigColor6;
uniform vec4 bigColor7;
uniform vec4 bigColor8;

varying vec4 BaseColor;

uniform float d;
uniform float d2;
uniform float d3;
uniform float d4;
uniform float d5;
uniform float d6;
uniform float d7;
uniform float d8;
uniform float d9;
uniform float d10;
uniform float d11;
uniform float d12;
uniform float d13;
uniform float d14;
uniform float d15;
uniform float d16;
uniform float d17;
uniform float d18;
uniform float d19;
uniform float d20;
uniform float d21;
uniform float d22;
uniform float d23;
uniform float d24;
uniform float d25;
uniform float d26;
uniform float d27;
uniform float d28;
uniform float d29;
uniform float d30;
uniform float d31;
uniform float d32;
uniform float d33;
uniform float d34;

uniform int Count;

void main()
{
    vec4 color = BaseColor;

    // Not a real loop
    while (true) {
        if (color.x < 0.33) {
            color += vec4(0.33);
            break;
        }
        if (color.x < 0.66) {
            color += vec4(0.66);
            break;
        }

        color += vec4(0.33);
        break;
    }

    // While
    while (color.x < d) {
        color += bigColor;
    }

    // While (latchy)
    while (color.z < d) {
        color += bigColor1_1;
        if (color.w < d)
            continue;

        color += bigColor1_1;
    }

    // While (constant)
    while (color.x < 42.0) {
        ++color;
    }

    // While (complicated-conditional)
    while (color.w < d2 && color.y < d3) {
        color += bigColor1_2;
    }

    // While (multi-exit)
    while (color.z < d3) {
        color += bigColor1_3;
        if (color.y < d4)
            break;
        color += bigColor1_3;
    }

    // For (dynamic)
    for (int i = 0; i < Count; ++i) {
        color += bigColor2;
    }

    // Do while
    do {
        color += bigColor3;
    } while (color.x < d2);

    // For (static)
    for (int i = 0; i < 42; ++i) {
        color.z += d3;
    }

    // For (static) flow-control
    for (int i = 0; i < 100; ++i) {
        if (color.z < 20.0)
            color.x++;
        else
            color.y++;
        if (color.w < 20.0)
            if (color.z > color.y)
                0;              // do nothing
    }

    // For (static) flow-control with latch merge
    for (int i = 0; i < 120; ++i) {
        if (color.z < 20.0)
            color.x++;
        else
            color.y++;
    }

    // For (static) latchy
    for (int i = 0; i < 42; ++i) {
        color.z += d3;
        if (color.x < d4)
            continue;
        ++color.w;
    }

    // For (static) multi-exit
    for (int i = 0; i < 42; ++i) {
        color.z += d3;
        if (color.x < d4)
            break;
        ++color.w;
    }

    // Latchy
    do {
        color += bigColor4;
        if (color.x < d4)
            continue;
        if (color.y < d4)
            color.y += d4;
        else
            color.x += d4;
    } while (color.z < d4);

    // Do while flow control
    do {
        color += bigColor5;
        if (color.y < d5)
            color.y += d5;
    } while (color.x < d5);

    // If then loop
    if (color.x < d6) {
        while (color.y < d6)
            color += bigColor6;
    } else {
        while (color.z < d6)
            color.z += bigColor6.z;
    }

    // If then multi-exit
    if (color.x < d6) {
        while (color.y < d6) {
            color += bigColor6;
            if (d7 < 1.0)
                break;
        }

    } else {
        while (color.z < d6)
            color.z += bigColor6.z;
    }


    // Multi-exit
    do {
       if (d7 < 0.0)
           break;

       color += bigColor7;

       if (d7 < 1.0) {
           color.z++;
           break;
       }

       color += BaseColor;

    } while (true);


    // Multi-exit2
    do {
        // invariant conditional break at the top of the loop. This could be a
        // situation where unswitching the loop has no real increases in code
        // size.
       if (d8 < 0.0)
           break;

       color += bigColor7;

       if (d8 < 1.0) {
           color.z++;
           if (d8 < 2.0) {
               color.y++;
           } else {
               color.x++;
           }
           break;
       }

       color += BaseColor;

    } while (color.z < d8);

    // Deep exit
    while (color.w < d9) {
        if (d9 > d8) {
            if (color.x <= d7) {
                if (color.z == 5.0)
                    color.w++;
                else
                    break;
            }
        }

    }

    // No end loop-back.
    while (color.z < d10) {
        color.y++;
        if (color.y < d11) {
            color.z++;
            if (color.w < d12)
                color.w++;
            else
                color.x++;
            continue;
        }

        color++;
        break;
    }

    // Multi-continue
    while (color.x < 10.0) {
        color += bigColor8;

        if (color.z < d8)
            if (color.w < d6)
                continue;

        color.y += bigColor8.x;
    }

    color++;
    gl_FragColor = color;

    // Early Return
    while (color.x < d14) {
        if (color.y < d15) {
            return;
        }
        else
            color++;
    }

    color++;

    while (color.w < d16) {
        color.w++;
    }


    // While (complicated-conditional)
    while (color.w < d2 && color.y < d3) {
        color += bigColor1_2;
        if (color.z < d3)
            return;
    }


    do {
        if (color.y < d18)
            return;
        color++;
    } while (color.x < d17);

    // Early Discard
    while (color.y < d16) {
        if (color.w < d16) {
            discard;
        } else
            color++;
    }

    color++;

    gl_FragColor = color;
}
