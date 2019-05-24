#version 120

int a;
int b, c;
int d1 = 1;
int e2 = 2, f;
int g, h3 = 3;

int i4[4];
int j, k5[5];
int m6[6], m7[7];
int n8[8], p;

int ii4[4] = int[](1, 2, 3, 4);
int ij, ik5[5] = int[](5, 6, 7, 8, 9);
int im2[2] = int[](10, 11), im3[3] = int[](12, 13, 14);
int in8[4] = int[](21, 22, 23, 24), ip;

void vi4[4] = int[](1, 2, 3, 4);
void vj, vk5[5] = int[](5, 6, 7, 8, 9);
void vm2[2] = int[](10, 11), vm3[3] = int[](12, 13, 14);
void vn8[4] = int[](21, 22, 23, 24), vp;

const int cii4[4] = int[](1, 2, 3, 4);
const int cij, cik5[5] = int[](5, 6, 7, 8, 9);
const int cim2[2] = int[](10, 11), cim3[3] = int[](12, 13, 14);
const int cin8[4] = int[](21, 22, 23, 24), cip;

uniform int uii4[4] = int[](1, 2, 3, 4);
uniform int uij, uik5[5] = int[](5, 6, 7, 8, 9);
uniform int uim2[2] = int[](10, 11), uim3[3] = int[](12, 13, 14);
uniform int uin8[4] = int[](21, 22, 23, 24), uip;

int gl_vi4[4] = int[](1, 2, 3, 4);
int gl_vj, gl_vk5[5] = int[](5, 6, 7, 8, 9);
int gl_vm2[2] = int[](10, 11), gl_vm3[3] = int[](12, 13, 14);
int gl_vn8[4] = int[](21, 22, 23, 24), gl_vp;

void main()
{
    while (bool cond = b < c);
    while (int icond = b);
    while (bool gl_cond = b < c);
}

int foob__vi4[4] = int[](1, 2, 3, 4);
int foob__vj, foob__vk5[5] = int[](5, 6, 7, 8, 9);
int __foobvm2[2] = int[](10, 11), __foobvm3[3] = int[](12, 13, 14);
int foob__vn8[4] = int[](21, 22, 23, 24), foob__vp;
