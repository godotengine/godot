float4 main() : SV_Target {
  int i = 1;
  int1 i2 = 2;
  int2 i3 = 3;
  int3 i4 = 4;
  int4 i5 = 5;

  i = dot(i, i);
  i2 = dot(i2, i2);
  i3 = dot(i3, i3);
  i4 = dot(i4, i4);
  i5 = dot(i5, i5);
  return i + i2.xxxx + i3.xyxy + i4.xyzx + i5;
}