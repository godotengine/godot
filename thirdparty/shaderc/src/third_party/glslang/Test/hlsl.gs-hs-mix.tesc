cbuffer UniformBlock0 : register(b0)
{
  float4x4 model_view_matrix;
  float4x4 proj_matrix;
  float4x4 model_view_proj_matrix;
  float3x3 normal_matrix;
  float3   color;
  float3   view_dir;
  float3   tess_factor;
};

// =============================================================================
// Hull Shader
// =============================================================================
struct HSInput {
  float3 PositionWS : POSITION;
  float3 NormalWS   : NORMAL;
};

struct HSOutput {
  float3 PositionWS : POSITION;
};

struct HSTrianglePatchConstant {
  float  EdgeTessFactor[3] : SV_TessFactor;
  float  InsideTessFactor  : SV_InsideTessFactor;
  float3 NormalWS[3]       : NORMAL;
};

HSTrianglePatchConstant HSPatchConstant(InputPatch<HSInput, 3> patch)
{
  float3 roundedEdgeTessFactor = tess_factor;
  float  roundedInsideTessFactor = 3;
  float  insideTessFactor = 1;

  HSTrianglePatchConstant result;

  // Edge and inside tessellation factors
  result.EdgeTessFactor[0] = roundedEdgeTessFactor.x;
  result.EdgeTessFactor[1] = roundedEdgeTessFactor.y;
  result.EdgeTessFactor[2] = roundedEdgeTessFactor.z;
  result.InsideTessFactor  = roundedInsideTessFactor;

  // Constant data
  result.NormalWS[0] = patch[0].NormalWS;
  result.NormalWS[1] = patch[1].NormalWS;
  result.NormalWS[2] = patch[2].NormalWS;

  return result;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(3)]
[patchconstantfunc("HSPatchConstant")]
HSOutput HSMain(
  InputPatch<HSInput, 3>  patch,
  uint                    id : SV_OutputControlPointID
)
{
  HSOutput output;
  output.PositionWS = patch[id].PositionWS;
  return output;
}

// =============================================================================
// Geometry Shader
// =============================================================================
struct GSVertexInput {
  float3 PositionWS : POSITION;
  float3 NormalWS   : NORMAL;
};

struct GSVertexOutput {
  float4 PositionCS : SV_POSITION;
};

[maxvertexcount(6)]
void GSMain(
  triangle GSVertexInput            input[3],
  inout LineStream<GSVertexOutput>  output
)
{

  float3 P0 = input[0].PositionWS.xyz;
  float3 P1 = input[1].PositionWS.xyz;
  float3 P2 = input[2].PositionWS.xyz;

  GSVertexOutput vertex;
  // Totally hacky...
  P0.z += 0.001;
  P1.z += 0.001;
  P2.z += 0.001;
  float4 Q0 = mul(proj_matrix, float4(P0, 1.0));
  float4 Q1 = mul(proj_matrix, float4(P1, 1.0));
  float4 Q2 = mul(proj_matrix, float4(P2, 1.0));

  // Edge 0
  vertex.PositionCS = Q0;
  output.Append(vertex);
  vertex.PositionCS = Q1;
  output.Append(vertex);
  output.RestartStrip();

  // Edge 1
  vertex.PositionCS = Q1;
  output.Append(vertex);
  vertex.PositionCS = Q2;
  output.Append(vertex);
  output.RestartStrip();

  // Edge 2
  vertex.PositionCS = Q2;
  output.Append(vertex);
  vertex.PositionCS = Q0;
  output.Append(vertex);
  output.RestartStrip();
}
