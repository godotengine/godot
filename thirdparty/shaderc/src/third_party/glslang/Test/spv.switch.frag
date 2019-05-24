#version 310 es
precision mediump float;
flat in int c, d;
in float x;
out float color;
in vec4 v;

vec4 foo1(vec4 v1, vec4 v2, int i1)
{
    switch (i1)
    {
    case 0:
        return v1;
    case 2:
    case 1:
        return v2;
    case 3:
        return v1 * v2;
    }

    return vec4(0.0);
}

vec4 foo2(vec4 v1, vec4 v2, int i1)
{
    switch (i1)
    {
    case 0:
        return v1;
    case 2:
        return vec4(1.0);
    case 1:
        return v2;
    case 3:
        return v1 * v2;
    }

    return vec4(0.0);
}

void main()
{
    float f;
    int a[2];
    int local = c;

    switch(++local)
    {
    }

    switch (c) {
    case 1:
        f = sin(x);
        break;
    case 2:
        f = cos(x);
        break;
    default:
        f = tan(x);
    }

    switch (c) {
    case 1:
        f += sin(x);
    case 2:
        f += cos(x);
        break;
    default:
        f += tan(x);
    }

    switch (c) {
    case 1:
        f += sin(x);
        break;
    case 2:
        f += cos(x);
        break;
    }

    switch (c) {
    case 1:
        f += sin(x);
        break;
    case 2:
        switch (d) {
        case 1:
            f += x * x * x;
            break;
        case 2:
            f += x * x;
            break;
        }
        break;
    default:
        f += tan(x);
    }

    for (int i = 0; i < 10; ++i) {
        switch (c) {
        case 1:
            f += sin(x);
            for (int j = 20; j < 30; ++j) {
                ++f;
                if (f < 100.2)
                    break;
            }
            break;
        case 2:
            f += cos(x);
            break;
            break;
        default:
            f += tan(x);
        }

        if (f < 3.43)
            break;
    }

    switch (c) {
    case 1:
        f += sin(x);
        break;
    case 2:
        // test no statements at end
    }

    color = f + float(local);

    color += foo1(v,v,c).y;
    color += foo2(v,v,c).z;

    switch (c) {
    case 0: break;
    default:
    }

    switch (c) {
    default:
    }
}
