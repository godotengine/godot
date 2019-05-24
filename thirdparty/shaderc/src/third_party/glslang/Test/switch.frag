#version 300 es
precision highp float;
uniform int c, d;
in highp float x;

void main()
{
    float f;
    int a[2];

    switch(f) {      // ERROR
    }

    switch(a) {      // ERROR
    }

    switch(c)
    {
    }

    switch(c)        // WARNING, not enough stuff after last label
    {
    case 2:
    }

    switch(c)
    {
        f = sin(x);   // ERRROR
    case 2:
        f = cos(x);
        break;
    }

    switch (c) {
    default:
        break;
    case 1:
        f = sin(x);
        break;
    case 2:
        f = cos(x);
        break;
    default:           // ERROR, 2nd default
        f = tan(x);
    }

    switch (c) {
    case 1:
        f = sin(x);
        break;
    case 2:
        switch (d) {
        case 1:
            f = x * x * x;
            break;
        case 2:
            f = x * x;
            break;
        }
        break;
    default:
        f = tan(x);
    case 1:           // ERROR, 2nd 'case 1'
        break;
    case 3.8:         // ERROR, non-int
        break;
    case c:           // ERROR, non-constant
        break;       
    }

    switch (c) {      // a no-error normal switch
    case 1:
        f = sin(x);
        break;
    case 2:
        switch (d) {
        case 1:
            f = x * x * x;
            break;
        case 2:
            f = x * x;
            break;
        }
        break;
    default:
        f = tan(x);
    }

    break;            // ERROR

    switch (c) {
    case 1:
        f = sin(x);
        break;
    case 2:
        switch (d) {
        case 1:
            {
                case 4:        // ERROR
                    break;
            }
            f = x * x * x;
            if (c < d) {
                case 2:         // ERROR
                    f = x * x;
            }
            if (d < c)
                case 3:         // ERROR
            break;
        }
        break;
    case 4:
        f = tan(x);
        if (f < 0.0)
            default:            // ERROR
                break;
    }

    case 5:  // ERROR
    default: // ERROR

    switch (0) {
        default:
        int onlyInSwitch = 0;
    }
    onlyInSwitch;   // ERROR
   
    switch (0) {
        default:
            int x;  // WARNING (was "no statement" ERROR, but spec. changed because unclear what a statement is)
    }

    switch (c) {
    case 1:
    {
        int nestedX;
        break;
    }
    case 2:
        nestedX;    // ERROR
        int nestedZ;
        float a;    // okay, hiding outer 'a'
        break;
    case 3:
        int linearZ;
        break;
        break;
    case 4:
        int linearY = linearZ;
        break;
    case 5:         // okay that branch bypassed an initializer
        const int linearC = 4;
        break;
    case 6:         // okay that branch bypassed an initializer
        linearC;
    }
    nestedZ;        // ERROR, no longer in scope
}
