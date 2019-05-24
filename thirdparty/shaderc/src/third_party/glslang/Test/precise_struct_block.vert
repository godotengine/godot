#version 450

struct T {
  float f1;
  float f2;
};

out B1 {precise T s; float x;} partial_precise_block;
precise out B2 {T s; float x;} all_precise_block;

float struct_member() {
  float a = 1.0;
  float b = 2.0;
  float c = 3.0;
  float d = 4.0;

  precise float result;

  T S, S2, S3;

  S2.f1 = a + 0.2;      // NoContraction
  S2.f2 = b + 0.2;      // NOT NoContraction
  S3.f1 = a + b;        // NOT NoContraction
  S = S2;               // "precise" propagated through parent object nodes
  result = S.f1 + 0.1;  // the ADD operation should be NoContraction

  return result;
}

float complex_array_struct() {
  precise float result;
  struct T1 {
    float t1_array[3];
    float t1_scalar;
  };
  struct T2 {
    T1 t1a[5];
    T1 t1b[6];
    T1 t1c[7];
  };
  struct T3 {float f; T2 t2; vec4 v; int p;};
  T3 t3[10];
  for(int i=0; i<10; i++) {
    t3[i].f = i / 3.0; // Not NoContraction
    t3[i].v = vec4(i * 1.5); // NoContraction
    t3[i].p = i + 1;
    for(int j=0; j<5; j++) {
      for(int k = 0; k<3; k++) {
        t3[i].t2.t1a[j].t1_array[k] = i * j + k; // Not NoContraction
      }
      t3[i].t2.t1a[j].t1_scalar = j * 2.0 / i; // Not NoContration
    }

    for(int j=0; j<6; j++) {
      for(int k = 0; k<3; k++) {
        t3[i].t2.t1b[j].t1_array[k] = i * j + k; // Not NoContraction
      }
      t3[i].t2.t1b[j].t1_scalar = j * 2.0 / i; // NoContraction
    }

    for(int j=0; j<6; j++) {
      for(int k = 0; k<3; k++) {
        t3[i].t2.t1c[j].t1_array[k] = i * j + k; // Not NoContraction because all operands are integers
      }
      t3[i].t2.t1c[j].t1_scalar = j * 2.0 / i; // Not NoContraction
    }
  }
  int i = 2;
  result = t3[5].t2.t1c[6].t1_array[1]
           + t3[2].t2.t1b[1].t1_scalar
           + t3[i - 1].v.xy.x; // NoContraction
  return result;
}

float out_block() {
    float a = 0.1;
    float b = 0.2;
    partial_precise_block.s.f1 = a + b; // NoContraction
    partial_precise_block.s.f2 = a - b; // NoContraction
    partial_precise_block.x = a * b; // Not NoContraction

    all_precise_block.s.f1 = a + b + 1.0; // NoContraction
    all_precise_block.s.f2 = a - b - 1.0; // NoContraction
    all_precise_block.x = a * b * 2.0; // Also NoContraction

    return a + b; // Not NoContraction
}

void main(){}
