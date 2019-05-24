#version 450

precision highp float;

struct U {
    mat2 m;
};

struct T {
    mat2 m;
};

struct S {
    T t;
    U u;
};

void main()
{
    S s1 = S(T(mat2(1.0)), U(mat2(1.0)));
    S s2 = S(T(mat2(1.0)), U(mat2(1.0)));
}
