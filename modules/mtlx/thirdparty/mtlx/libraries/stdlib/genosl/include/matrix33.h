// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
//
// MaterialX specification (c) 2017 Lucasfilm Ltd.
// http://www.materialx.org/

#pragma once
#define MATRIX33_H


struct matrix33
{
    matrix m;
};

int isValidAs33(matrix m44)
{
    return m44[0][3] == 0 && 
           m44[1][3] == 0 &&
           m44[2][3] == 0 &&
           m44[3][0] == 0 &&
           m44[3][1] == 0 &&
           m44[3][2] == 0 &&
           m44[3][3] == 1;
}

matrix matrix33To44 (matrix33 m33)
{
    return m33.m;
}

// Convert an arbitrary m44 to m33 by removing the translation
//QUESTION: should we check if it's valid to represent the 4x4 as a 3x3?
matrix33 matrix44To33 (matrix m44)
{
    matrix33 m33;
    m33.m = m44;
    m33.m[0][3] = 0;
    m33.m[1][3] = 0;
    m33.m[2][3] = 0;
    m33.m[3][0] = 0;
    m33.m[3][1] = 0;
    m33.m[3][2] = 0;
    m33.m[3][3] = 1;

    return m33;
}

matrix33 __operator__neg__(matrix33 a)
{
    matrix33 m33;
    m33.m = -a.m;
    return m33;
}


matrix33 __operator__mul__(int a, matrix33 b)
{
    matrix33 m33;
    m33.m = a * b.m;
    return m33;
}

matrix33 __operator__mul__(float a, matrix33 b)
{
    matrix33 m33;
    m33.m = a * b.m;
    return m33;
}

matrix33 __operator__mul__(matrix33 a, int b)
{
    matrix33 m33;
    m33.m = a.m * b;
    return m33;
}

matrix33 __operator__mul__(matrix33 a, float b)
{
    matrix33 m33;
    m33.m = a.m * b;
    return m33;
}

matrix33 __operator__mul__(matrix33 a, matrix33 b)
{
    matrix33 m33;
    m33.m = a.m * b.m;
    return m33;
}

matrix33 __operator__div__(int a, matrix33 b)
{
    matrix33 m33;
    m33.m = a / b.m;
    return m33;
}

matrix33 __operator__div__(float a, matrix33 b)
{
    matrix33 m33;
    m33.m = a / b.m;
    return m33;
}

matrix33 __operator__div__(matrix33 a, int b)
{
    matrix33 m33;
    m33.m = a.m / b;
    return m33;
}

matrix33 __operator__div__(matrix33 a, float b)
{
    matrix33 m33;
    m33.m = a.m / b;
    return m33;
}

matrix33 __operator__div__(matrix33 a, matrix33 b)
{
    matrix33 m33;
    m33.m = a.m / b.m;
    return m33;
}

int __operator__eq__(matrix33 a, matrix33 b)
{
    return a.m == b.m;
}

int __operator__ne__(matrix33 a, matrix33 b)
{
    return a.m != b.m;
}

float determinant (matrix33 a)
{
    return determinant(a.m);
}

matrix33 transpose(matrix33 a)
{
    matrix33 m33;
    m33.m = transpose(a.m);
    return m33;
}

point transform(matrix33 a, point b)
{
    return transform(a.m, b);
}

vector transform(matrix33 a, vector b)
{
    return transform(a.m, b);
}

normal transform(matrix33 a, normal b)
{
    return transform(a.m, b);
}



