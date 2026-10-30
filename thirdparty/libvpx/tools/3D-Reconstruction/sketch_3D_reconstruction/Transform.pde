class Transform {
  float[] inv_rot;  // inverse of rotation matrix
  PVector inv_mov;  // inverse of movement vector
  float focal;      // the focal distacne of real camera
  int w, h;         // the width and height of the frame
  float normalier;  // nomalization factor of depth
  Transform(float tx, float ty, float tz, float qx, float qy, float qz,
            float qw, float fov, int w, int h, float normalier) {
    // currently, we did not use the info of real camera's position and
    // quaternion maybe we will use it in the future when combine all frames
    float[] rot = quaternion2Mat3x3(qx, qy, qz, qw);
    inv_rot = transpose3x3(rot);
    inv_mov = new PVector(-tx, -ty, -tz);
    this.focal = 0.5f * h / tan(fov / 2.0);
    this.w = w;
    this.h = h;
    this.normalier = normalier;
  }

  PVector transform(int i, int j, float d) {
    // transfer from camera view to world view
    float z = d / normalier;
    float x = (i - w / 2.0f) * z / focal;
    float y = (j - h / 2.0f) * z / focal;
    return new PVector(x, y, z);
  }
}

// get rotation matrix by using rotation axis and angle
float[] getRotationMat3x3(float angle, float ax, float ay, float az) {
  float[] mat = new float[9];
  float c = cos(angle);
  float s = sin(angle);
  mat[0] = c + ax * ax * (1 - c);
  mat[1] = ax * ay * (1 - c) - az * s;
  mat[2] = ax * az * (1 - c) + ay * s;
  mat[3] = ay * ax * (1 - c) + az * s;
  mat[4] = c + ay * ay * (1 - c);
  mat[5] = ay * az * (1 - c) - ax * s;
  mat[6] = az * ax * (1 - c) - ay * s;
  mat[7] = az * ay * (1 - c) + ax * s;
  mat[8] = c + az * az * (1 - c);
  return mat;
}

// get rotation matrix by using quaternion
float[] quaternion2Mat3x3(float qx, float qy, float qz, float qw) {
  float[] mat = new float[9];
  mat[0] = 1 - 2 * qy * qy - 2 * qz * qz;
  mat[1] = 2 * qx * qy - 2 * qz * qw;
  mat[2] = 2 * qx * qz + 2 * qy * qw;
  mat[3] = 2 * qx * qy + 2 * qz * qw;
  mat[4] = 1 - 2 * qx * qx - 2 * qz * qz;
  mat[5] = 2 * qy * qz - 2 * qx * qw;
  mat[6] = 2 * qx * qz - 2 * qy * qw;
  mat[7] = 2 * qy * qz + 2 * qx * qw;
  mat[8] = 1 - 2 * qx * qx - 2 * qy * qy;
  return mat;
}

// tranpose a 3x3 matrix
float[] transpose3x3(float[] mat) {
  float[] Tmat = new float[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Tmat[i * 3 + j] = mat[j * 3 + i];
    }
  return Tmat;
}

// multiply a matrix with vector
PVector MatxVec3(float[] mat, PVector v) {
  float[] vec = v.array();
  float[] res = new float[3];
  for (int i = 0; i < 3; i++) {
    res[i] = 0.0f;
    for (int j = 0; j < 3; j++) {
      res[i] += mat[i * 3 + j] * vec[j];
    }
  }
  return new PVector(res[0], res[1], res[2]);
}
