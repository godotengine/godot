
Texture2D      MyTexture : register(t0);

//----------------------------------------------------------------------------------------
void TexFunc(in const Texture2D t2D, out float3 RGB)
{    
    RGB = 0;
}

//-----------------------------------------------------------------------------------
void main() 
{
    float3 final_RGB;

    TexFunc(MyTexture, final_RGB);
}
