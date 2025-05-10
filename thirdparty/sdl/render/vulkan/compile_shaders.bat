glslangValidator -D --sep main -e main -S frag --target-env vulkan1.0 --auto-sampled-textures --vn VULKAN_PixelShader_Colors -o VULKAN_PixelShader_Colors.h VULKAN_PixelShader_Colors.hlsl
glslangValidator -D --sep main -e main -S frag --target-env vulkan1.0 --auto-sampled-textures --vn VULKAN_PixelShader_Textures -o VULKAN_PixelShader_Textures.h VULKAN_PixelShader_Textures.hlsl
glslangValidator -D --sep main -e main -S frag --target-env vulkan1.0 --auto-sampled-textures --vn VULKAN_PixelShader_Advanced -o VULKAN_PixelShader_Advanced.h VULKAN_PixelShader_Advanced.hlsl

glslangValidator -D --sep mainColor -e main -S vert --iy --target-env vulkan1.0 --vn VULKAN_VertexShader -o VULKAN_VertexShader.h VULKAN_VertexShader.hlsl
