// MIT License

// Copyright (c) 2019 wi-re
// Copyright 2023 The Manifold Authors.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Modified from https://github.com/wi-re/tbtSVD, removing CUDA dependence and
// approximate inverse square roots.

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <tuple>
#include <utility>

namespace {
// Constants used for calculation of Givens quaternions
inline constexpr float _gamma = 5.828427124f;   // sqrt(8)+3;
inline constexpr float _cStar = 0.923879532f;   // cos(pi/8)
inline constexpr float _sStar = 0.3826834323f;  // sin(pi/8)
// Threshold value
inline constexpr float _SVD_EPSILON = 1e-6f;
// Iteration counts for Jacobi Eigen Analysis, influences precision
inline constexpr int JACOBI_STEPS = 12;

// Helper function used to swap X with Y and Y with  X if c == true
inline void CondSwap(bool c, float& X, float& Y) {
  float Z = X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}
// Helper function used to swap X with Y and Y with -X if c == true
inline void CondNegSwap(bool c, float& X, float& Y) {
  float Z = -X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}
// A simple symmetric 3x3 Matrix class (contains no storage for (0, 1) (0, 2)
// and (1, 2)
struct Symmetric3x3 {
  float m_00 = 1.f;
  float m_10 = 0.f, m_11 = 1.f;
  float m_20 = 0.f, m_21 = 0.f, m_22 = 1.f;

  Symmetric3x3(float a11 = 1.f, float a21 = 0.f, float a22 = 1.f,
               float a31 = 0.f, float a32 = 0.f, float a33 = 1.f)
      : m_00(a11), m_10(a21), m_11(a22), m_20(a31), m_21(a32), m_22(a33) {}
  Symmetric3x3(glm::mat3 o)
      : m_00(o[0][0]),
        m_10(o[0][1]),
        m_11(o[1][1]),
        m_20(o[0][2]),
        m_21(o[1][2]),
        m_22(o[2][2]) {}
};
// Helper struct to store 2 floats to avoid OUT parameters on functions
struct Givens {
  float ch = _cStar;
  float sh = _sStar;
};
// Helper struct to store 2 Matrices to avoid OUT parameters on functions
struct QR {
  glm::mat3 Q, R;
};
// Calculates the squared norm of the vector.
inline float Dist2(glm::vec3 v) { return glm::dot(v, v); }
// For an explanation of the math see
// http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf Computing the
// Singular Value Decomposition of 3 x 3 matrices with minimal branching and
// elementary floating point operations See Algorithm 2 in reference. Given a
// matrix A this function returns the Givens quaternion (x and w component, y
// and z are 0)
inline Givens ApproximateGivensQuaternion(Symmetric3x3& A) {
  Givens g{2.f * (A.m_00 - A.m_11), A.m_10};
  bool b = _gamma * g.sh * g.sh < g.ch * g.ch;
  float w = 1.f / sqrt(fmaf(g.ch, g.ch, g.sh * g.sh));
  if (w != w) b = 0;
  return Givens{b ? w * g.ch : (float)_cStar, b ? w * g.sh : (float)_sStar};
}
// Function used to apply a Givens rotation S. Calculates the weights and
// updates the quaternion to contain the cumulative rotation
inline void JacobiConjugation(const int32_t x, const int32_t y, const int32_t z,
                              Symmetric3x3& S, glm::vec4& q) {
  auto g = ApproximateGivensQuaternion(S);
  float scale = 1.f / fmaf(g.ch, g.ch, g.sh * g.sh);
  float a = fmaf(g.ch, g.ch, -g.sh * g.sh) * scale;
  float b = 2.f * g.sh * g.ch * scale;
  Symmetric3x3 _S = S;
  // perform conjugation S = Q'*S*Q
  S.m_00 = fmaf(a, fmaf(a, _S.m_00, b * _S.m_10),
                b * (fmaf(a, _S.m_10, b * _S.m_11)));
  S.m_10 = fmaf(a, fmaf(-b, _S.m_00, a * _S.m_10),
                b * (fmaf(-b, _S.m_10, a * _S.m_11)));
  S.m_11 = fmaf(-b, fmaf(-b, _S.m_00, a * _S.m_10),
                a * (fmaf(-b, _S.m_10, a * _S.m_11)));
  S.m_20 = fmaf(a, _S.m_20, b * _S.m_21);
  S.m_21 = fmaf(-b, _S.m_20, a * _S.m_21);
  S.m_22 = _S.m_22;
  // update cumulative rotation qV
  glm::vec3 tmp = g.sh * glm::vec3(q);
  g.sh *= q[3];
  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1)) for (p,q) =
  // ((0,1),(1,2),(0,2))
  q[z] = fmaf(q[z], g.ch, g.sh);
  q[3] = fmaf(q[3], g.ch, -tmp[z]);  // w
  q[x] = fmaf(q[x], g.ch, tmp[y]);
  q[y] = fmaf(q[y], g.ch, -tmp[x]);
  // re-arrange matrix for next iteration
  _S.m_00 = S.m_11;
  _S.m_10 = S.m_21;
  _S.m_11 = S.m_22;
  _S.m_20 = S.m_10;
  _S.m_21 = S.m_20;
  _S.m_22 = S.m_00;
  S.m_00 = _S.m_00;
  S.m_10 = _S.m_10;
  S.m_11 = _S.m_11;
  S.m_20 = _S.m_20;
  S.m_21 = _S.m_21;
  S.m_22 = _S.m_22;
}
// Function used to contain the Givens permutations and the loop of the jacobi
// steps controlled by JACOBI_STEPS Returns the quaternion q containing the
// cumulative result used to reconstruct S
inline glm::mat3 JacobiEigenAnalysis(Symmetric3x3 S) {
  glm::vec4 q(0, 0, 0, 1);
  for (int32_t i = 0; i < JACOBI_STEPS; i++) {
    JacobiConjugation(0, 1, 2, S, q);
    JacobiConjugation(1, 2, 0, S, q);
    JacobiConjugation(2, 0, 1, S, q);
  }
  return glm::mat3(1.f - 2.f * (fmaf(q.y, q.y, q.z * q.z)),  //
                   2.f * fmaf(q.x, q.y, +q.w * q.z),         //
                   2.f * fmaf(q.x, q.z, -q.w * q.y),         //
                   2 * fmaf(q.x, q.y, -q.w * q.z),           //
                   1 - 2 * fmaf(q.x, q.x, q.z * q.z),        //
                   2 * fmaf(q.y, q.z, q.w * q.x),            //
                   2 * fmaf(q.x, q.z, q.w * q.y),            //
                   2 * fmaf(q.y, q.z, -q.w * q.x),           //
                   1 - 2 * fmaf(q.x, q.x, q.y * q.y));
}
// Implementation of Algorithm 3
inline void SortSingularValues(glm::mat3& B, glm::mat3& V) {
  float rho1 = Dist2(B[0]);
  float rho2 = Dist2(B[1]);
  float rho3 = Dist2(B[2]);
  bool c;
  c = rho1 < rho2;
  CondNegSwap(c, B[0][0], B[1][0]);
  CondNegSwap(c, V[0][0], V[1][0]);
  CondNegSwap(c, B[0][1], B[1][1]);
  CondNegSwap(c, V[0][1], V[1][1]);
  CondNegSwap(c, B[0][2], B[1][2]);
  CondNegSwap(c, V[0][2], V[1][2]);
  CondSwap(c, rho1, rho2);
  c = rho1 < rho3;
  CondNegSwap(c, B[0][0], B[2][0]);
  CondNegSwap(c, V[0][0], V[2][0]);
  CondNegSwap(c, B[0][1], B[2][1]);
  CondNegSwap(c, V[0][1], V[2][1]);
  CondNegSwap(c, B[0][2], B[2][2]);
  CondNegSwap(c, V[0][2], V[2][2]);
  CondSwap(c, rho1, rho3);
  c = rho2 < rho3;
  CondNegSwap(c, B[1][0], B[2][0]);
  CondNegSwap(c, V[1][0], V[2][0]);
  CondNegSwap(c, B[1][1], B[2][1]);
  CondNegSwap(c, V[1][1], V[2][1]);
  CondNegSwap(c, B[1][2], B[2][2]);
  CondNegSwap(c, V[1][2], V[2][2]);
}
// Implementation of Algorithm 4
inline Givens QRGivensQuaternion(float a1, float a2) {
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  float epsilon = (float)_SVD_EPSILON;
  float rho = sqrt(fmaf(a1, a1, +a2 * a2));
  Givens g{fabsf(a1) + fmaxf(rho, epsilon), rho > epsilon ? a2 : 0};
  bool b = a1 < 0.f;
  CondSwap(b, g.sh, g.ch);
  float w = 1.f / sqrt(fmaf(g.ch, g.ch, g.sh * g.sh));
  g.ch *= w;
  g.sh *= w;
  return g;
}
// Implements a QR decomposition of a Matrix, see Sec 4.2
inline QR QRDecomposition(glm::mat3& B) {
  glm::mat3 Q, R;
  // first Givens rotation (ch,0,0,sh)
  auto g1 = QRGivensQuaternion(B[0][0], B[0][1]);
  auto a = fmaf(-2.f, g1.sh * g1.sh, 1.f);
  auto b = 2.f * g1.ch * g1.sh;
  // apply B = Q' * B
  R[0][0] = fmaf(a, B[0][0], b * B[0][1]);
  R[1][0] = fmaf(a, B[1][0], b * B[1][1]);
  R[2][0] = fmaf(a, B[2][0], b * B[2][1]);
  R[0][1] = fmaf(-b, B[0][0], a * B[0][1]);
  R[1][1] = fmaf(-b, B[1][0], a * B[1][1]);
  R[2][1] = fmaf(-b, B[2][0], a * B[2][1]);
  R[0][2] = B[0][2];
  R[1][2] = B[1][2];
  R[2][2] = B[2][2];
  // second Givens rotation (ch,0,-sh,0)
  auto g2 = QRGivensQuaternion(R[0][0], R[0][2]);
  a = fmaf(-2.f, g2.sh * g2.sh, 1.f);
  b = 2.f * g2.ch * g2.sh;
  // apply B = Q' * B;
  B[0][0] = fmaf(a, R[0][0], b * R[0][2]);
  B[1][0] = fmaf(a, R[1][0], b * R[1][2]);
  B[2][0] = fmaf(a, R[2][0], b * R[2][2]);
  B[0][1] = R[0][1];
  B[1][1] = R[1][1];
  B[2][1] = R[2][1];
  B[0][2] = fmaf(-b, R[0][0], a * R[0][2]);
  B[1][2] = fmaf(-b, R[1][0], a * R[1][2]);
  B[2][2] = fmaf(-b, R[2][0], a * R[2][2]);
  // third Givens rotation (ch,sh,0,0)
  auto g3 = QRGivensQuaternion(B[1][1], B[1][2]);
  a = fmaf(-2.f, g3.sh * g3.sh, 1.f);
  b = 2.f * g3.ch * g3.sh;
  // R is now set to desired value
  R[0][0] = B[0][0];
  R[1][0] = B[1][0];
  R[2][0] = B[2][0];
  R[0][1] = fmaf(a, B[0][1], b * B[0][2]);
  R[1][1] = fmaf(a, B[1][1], b * B[1][2]);
  R[2][1] = fmaf(a, B[2][1], b * B[2][2]);
  R[0][2] = fmaf(-b, B[0][1], a * B[0][2]);
  R[1][2] = fmaf(-b, B[1][1], a * B[1][2]);
  R[2][2] = fmaf(-b, B[2][1], a * B[2][2]);
  // construct the cumulative rotation Q=Q1 * Q2 * Q3
  // the number of floating point operations for three quaternion
  // multiplications is more or less comparable to the explicit form of the
  // joined matrix. certainly more memory-efficient!
  auto sh12 = 2.f * fmaf(g1.sh, g1.sh, -0.5f);
  auto sh22 = 2.f * fmaf(g2.sh, g2.sh, -0.5f);
  auto sh32 = 2.f * fmaf(g3.sh, g3.sh, -0.5f);
  Q[0][0] = sh12 * sh22;
  Q[1][0] = fmaf(4.f * g2.ch * g3.ch, sh12 * g2.sh * g3.sh,
                 2.f * g1.ch * g1.sh * sh32);
  Q[2][0] = fmaf(4.f * g1.ch * g3.ch, g1.sh * g3.sh,
                 -2.f * g2.ch * sh12 * g2.sh * sh32);

  Q[0][1] = -2.f * g1.ch * g1.sh * sh22;
  Q[1][1] =
      fmaf(-8.f * g1.ch * g2.ch * g3.ch, g1.sh * g2.sh * g3.sh, sh12 * sh32);
  Q[2][1] = fmaf(
      -2.f * g3.ch, g3.sh,
      4.f * g1.sh * fmaf(g3.ch * g1.sh, g3.sh, g1.ch * g2.ch * g2.sh * sh32));

  Q[0][2] = 2.f * g2.ch * g2.sh;
  Q[1][2] = -2.f * g3.ch * sh22 * g3.sh;
  Q[2][2] = sh22 * sh32;
  return QR{Q, R};
}
}  // namespace

namespace manifold {
/** @addtogroup Connections
 *  @{
 */

/**
 * The three matrices of a Singular Value Decomposition.
 */
struct SVDSet {
  glm::mat3 U, S, V;
};

/**
 * Returns the Singular Value Decomposition of A: A = U * S * glm::transpose(V).
 *
 * @param A The matrix to decompose.
 */
inline SVDSet SVD(glm::mat3 A) {
  glm::mat3 V = JacobiEigenAnalysis(glm::transpose(A) * A);
  auto B = A * V;
  SortSingularValues(B, V);
  QR qr = QRDecomposition(B);
  return SVDSet{qr.Q, qr.R, V};
}

/**
 * Returns the largest singular value of A.
 *
 * @param A The matrix to measure.
 */
inline float SpectralNorm(glm::mat3 A) {
  SVDSet usv = SVD(A);
  return usv.S[0][0];
}
/** @} */
}  // namespace manifold
