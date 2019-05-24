#version 420

struct S 
{
    vec3 color;
};

layout(location = 0) out vec3 OutColor;

flat in int u;

void GetColor1(const S i) 
{ 
    OutColor += i.color.x;
}

void GetColor2(const S i, int comp)
{ 
    OutColor += i.color[comp];
}

void GetColor3(const S i, int comp)
{ 
    OutColor += i.color[comp].x;
}

void GetColor4(const S i, int comp)
{ 
    OutColor += i.color[comp].x;
}

void GetColor5(const S i, int comp)
{ 
    OutColor += i.color;
}

void GetColor6(const S i, int comp)
{ 
    OutColor += i.color.yx[comp];
}

void GetColor7(const S i, int comp)
{ 
    OutColor.xy += i.color.yxz.yx;
}

void GetColor8(const S i, int comp)
{ 
    OutColor += i.color.yzx.yx.x.x;
}

void GetColor9(const S i, int comp)
{ 
    OutColor.zxy += i.color;
}

void GetColor10(const S i, int comp)
{ 
    OutColor.zy += i.color.xy;
}

void GetColor11(const S i, int comp)
{ 
    OutColor.zxy.yx += i.color.xy;
}

void GetColor12(const S i, int comp)
{ 
    OutColor[comp] += i.color.x;
}

void GetColor13(const S i, int comp)
{ 
    OutColor.zy[comp] += i.color.x;
}

void GetColor14(const S i, int comp)
{ 
    OutColor.zyx[comp] = i.color.x;
}

void main()
{
    S s;
    OutColor = vec3(0.0);
    GetColor1(s);
    GetColor2(s, u);
    GetColor3(s, u);
    GetColor4(s, u);
    GetColor5(s, u);
    GetColor6(s, u);
    GetColor7(s, u);
    GetColor8(s, u);
    GetColor9(s, u);
    GetColor10(s, u);
    GetColor11(s, u);
    GetColor12(s, u);
    GetColor13(s, u);
    GetColor14(s, u);
}
