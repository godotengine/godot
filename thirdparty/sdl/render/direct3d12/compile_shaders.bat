rem This script runs for the Windows build, but also via the _xbox variant with these vars set.
rem Make sure to default to building for Windows if they're not set.
if %DXC%.==. set DXC=dxc
if %SUFFIX%.==. set SUFFIX=.h

echo Building with %DXC%
echo Suffix %SUFFIX%

cd "%~dp0"

%DXC% -E main -T ps_6_0 -Fh D3D12_PixelShader_Colors%SUFFIX% D3D12_PixelShader_Colors.hlsl
%DXC% -E main -T ps_6_0 -Fh D3D12_PixelShader_Textures%SUFFIX% D3D12_PixelShader_Textures.hlsl
%DXC% -E main -T ps_6_0 -Fh D3D12_PixelShader_Advanced%SUFFIX% D3D12_PixelShader_Advanced.hlsl

%DXC% -E mainColor -T vs_6_0 -Fh D3D12_VertexShader_Color%SUFFIX% D3D12_VertexShader.hlsl
%DXC% -E mainTexture -T vs_6_0 -Fh D3D12_VertexShader_Texture%SUFFIX% D3D12_VertexShader.hlsl
%DXC% -E mainAdvanced -T vs_6_0 -Fh D3D12_VertexShader_Advanced%SUFFIX% D3D12_VertexShader.hlsl

%DXC% -E ColorRS -T rootsig_1_1 -rootsig-define ColorRS -Fh D3D12_RootSig_Color%SUFFIX% D3D12_VertexShader.hlsl
%DXC% -E TextureRS -T rootsig_1_1 -rootsig-define TextureRS -Fh D3D12_RootSig_Texture%SUFFIX% D3D12_VertexShader.hlsl
%DXC% -E AdvancedRS -T rootsig_1_1 -rootsig-define AdvancedRS -Fh D3D12_RootSig_Advanced%SUFFIX% D3D12_VertexShader.hlsl
