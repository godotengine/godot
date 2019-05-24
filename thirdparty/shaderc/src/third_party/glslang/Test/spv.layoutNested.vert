#version 450

// should get 3 SPV types for S: no layout, 140, and 430, plus extras for interpolation or invariant differences
struct S
{
	highp uvec3 a;
	mediump mat2 b[4];
	lowp uint c;
};

layout(set = 0, binding = 0, std140) uniform Block140
{
	mediump int u;
	S s[2][3];
	mediump vec2 v;
} inst140;

layout(set = 0, binding = 1, std430) buffer Block430
{
	mediump int u;
	S s[2][3];
	mediump vec2 v;
} inst430;

S s;

// should get 5 SPV types for T: no layout, 140/row, 140/col, 430/row, and 430/col
struct T {
    mat2 m;
    int a;
};

T t;

struct Nestor {
    T nestorT;
};

layout(set = 1, binding = 0, std140) uniform Bt1
{
    layout(row_major) Nestor nt;
} Btn1;

layout(set = 1, binding = 0, std140) uniform Bt2
{
    layout(column_major) Nestor nt;
} Btn2;

layout(row_major, set = 1, binding = 0, std140) uniform Bt3
{
    layout(column_major) Nestor ntcol;
    Nestor ntrow;                         // should be row major decoration version of Nestor
} Btn3;

layout(set = 1, binding = 0, std430) buffer bBt1
{
    layout(row_major) Nestor nt;
} bBtn1;

layout(set = 1, binding = 0, std430) buffer bBt2
{
    layout(column_major) Nestor nt;
} bBtn2;

layout(set = 1, binding = 0, std430) buffer bBt3
{
    layout(row_major) Nestor ntcol;
    Nestor ntrow;                         // should be col major decoration version of Nestor
} bBtn3;

void main()
{
}

flat out S sout;
invariant out S soutinv;
