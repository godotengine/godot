// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_FITTING_H
#define NV_MATH_FITTING_H

#include "Vector.h"
#include "Plane.h"

namespace nv
{
    namespace Fit
    {
        Vector3 computeCentroid(int n, const Vector3 * points);
        Vector3 computeCentroid(int n, const Vector3 * points, const float * weights, const Vector3 & metric);

        Vector4 computeCentroid(int n, const Vector4 * points);
        Vector4 computeCentroid(int n, const Vector4 * points, const float * weights, const Vector4 & metric);

        Vector3 computeCovariance(int n, const Vector3 * points, float * covariance);
        Vector3 computeCovariance(int n, const Vector3 * points, const float * weights, const Vector3 & metric, float * covariance);

        Vector4 computeCovariance(int n, const Vector4 * points, float * covariance);
        Vector4 computeCovariance(int n, const Vector4 * points, const float * weights, const Vector4 & metric, float * covariance);

        Vector3 computePrincipalComponent_PowerMethod(int n, const Vector3 * points);
        Vector3 computePrincipalComponent_PowerMethod(int n, const Vector3 * points, const float * weights, const Vector3 & metric);

        Vector3 computePrincipalComponent_EigenSolver(int n, const Vector3 * points);
        Vector3 computePrincipalComponent_EigenSolver(int n, const Vector3 * points, const float * weights, const Vector3 & metric);

		Vector4 computePrincipalComponent_EigenSolver(int n, const Vector4 * points);
        Vector4 computePrincipalComponent_EigenSolver(int n, const Vector4 * points, const float * weights, const Vector4 & metric);

        Vector3 computePrincipalComponent_SVD(int n, const Vector3 * points);
        Vector4 computePrincipalComponent_SVD(int n, const Vector4 * points);

        Plane bestPlane(int n, const Vector3 * points);
        bool isPlanar(int n, const Vector3 * points, float epsilon = NV_EPSILON);

        bool eigenSolveSymmetric3(const float matrix[6], float eigenValues[3], Vector3 eigenVectors[3]);
        bool eigenSolveSymmetric4(const float matrix[10], float eigenValues[4], Vector4 eigenVectors[4]);

        // Returns number of clusters [1-4].
        int compute4Means(int n, const Vector3 * points, const float * weights, const Vector3 & metric, Vector3 * cluster);
    }

} // nv namespace

#endif // NV_MATH_FITTING_H
