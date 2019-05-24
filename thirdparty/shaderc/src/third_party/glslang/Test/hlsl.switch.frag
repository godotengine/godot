float4 PixelShaderFunction(float4 input, int c, int d) : COLOR0
{
    switch(c)
    {
    }

    switch(c)
    {
    default:
    }

    switch (c) {
    case 1:
        ++input;
        break;
    case 2:
        --input;
        break;
    }

    [branch] switch (c) {
    case 1:
        ++input;
        break;
    case 2:
        switch (d) {
        case 2:
            input += 2.0;
            break;
        case 3:
            input += 3.0;
            break;
        }
        break;
    default:
        input += 4.0;
    }

    switch (c) {
    case 1:
    }

    switch (c) {
    case 1:
    case 2:
    case 3:
        ++input;
        break;
    case 4:
    case 5:
        --input;
    }

    return input;
}
