#version 330

varying mat4 mvp;

in vec4 v;
in mat3 am3;
in mat4 arraym[3];

out float f;
out vec4 glPos;
//out mat4 mout[2];

void main()
{
	//needs complex output blocks to work: gl_Position = mvp * v;
    glPos = mvp * v;
	f = am3[2][1] + arraym[1][2][3];
    //mout[1] = arraym[2];
}
