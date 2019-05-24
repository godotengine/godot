#version 140

in float d;
in vec4 bigColor, smallColor;
in vec4 otherColor;

in float c;

in float threshhold;
in float threshhold2;
in float threshhold3;

in float minimum;

in vec4 BaseColor;

bool b;

void main()
{
    vec4 color = BaseColor;
    vec4 color2;

    color2 = otherColor;

    if (c > d)
        color += bigColor;
    else
        color += smallColor;

    if (color.z < minimum)
        return;

    color.z++;

    if (color.z > threshhold)
        discard;

    color++;

    // Two path, different rest
    if (color.w > threshhold2) {
        if (color.z > threshhold2)
            return;
        else if (b)
            color.z++;
        else {
            if (color.x < minimum) {
                discard;
            } else {
                color++;
            }
        }
    } else {
        if (b)
            discard;
        else
            return;
    }


    // // Two path, shared rest
    // if (color.w > threshhold2) {
    //     if (color.z > threshhold2)
    //         return;
    //     else if (b)
    //         color++;
    //     else {
    //         if (color.x < minimum) {
    //             discard;
    //         } else {
    //             color++;
    //         }
    //     }
    // } else {
    //     if (b)
    //         discard;
    //     else
    //         return;
    // }


    // // One path
    // if (color.w > threshhold2) {
    //     if (color.z > threshhold2)
    //         return;
    //     else {
    //         if (color.x < minimum) {
    //             discard;
    //         } else {
    //             color++;
    //         }
    //     }
    // } else {
    //     if (b)
    //         discard;
    //     else
    //         return;
    // }

    gl_FragColor = color * color2;
}
