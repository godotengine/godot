// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "motion_derivative.h"
#include "../../common/sys/regression.h"

#include <random>

namespace embree
{
  struct motion_derivative_regression_test : public RegressionTest
  {
    motion_derivative_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }

    std::mt19937_64 rng;

    inline float rand01()
    {
      std::uniform_real_distribution<float> dist(0.f, 1.f);
      return dist(rng);
    }

    bool valid(float v)
    {
      if (std::isnan(v) && !std::isfinite(v))
      {
        //printf("value not valid %f\n", v);
        return false;
      }
      return true;
    }

    inline Vec3fa random_axis()
    {
      float x = 2.f * rand01() - 1.f;
      float y = 2.f * rand01() - 1.f;
      float z = x*x+y*y > 0.995 ? 0.f : std::sqrt(1 - x*x - y*y);
      Vec3fa axis(x, y, z);
      return normalize(axis);
    }

    inline float random_scale()
    {
      float xi = 20.f * rand01() - 10.f;
      if (xi < 0) xi -= 0.001f;
      if (xi >= 0) xi += 0.001f;
      return xi;
    }

    inline AffineSpace3fa random_quaternion_decomposition()
    {
      Vec3fa T(20.f * rand01() - 10.f, 20.f * rand01() - 10.f, 20.f * rand01() - 10.f);
      Quaternion3f q = Quaternion3f::rotate(random_axis(), 4.f * M_PI * rand01() - 2.f * M_PI);
      AffineSpace3fa S(one);
      S.p = Vec3fa(20.f * rand01() - 10.f, 20.f * rand01() - 10.f, 20.f * rand01() - 10.f);
      S.l.vx.x = random_scale();
      S.l.vy.y = random_scale();
      S.l.vz.z = random_scale();
      S.l.vy.x = 2.f * rand01() - 1.f;
      S.l.vz.x = 2.f * rand01() - 1.f;
      S.l.vz.y = 2.f * rand01() - 1.f;
      return quaternionDecomposition(T, q, S);
    }

    bool testMDC(MotionDerivativeCoefficients const& mdc)
    {
      if (!valid(mdc.theta)) return false;
      for (int i = 0; i < 3*8*7; ++i)
      {
        if(!valid(mdc.coeffs[i])) return false;
      }

      unsigned int maxNumRoots = 32;
      float roots[32];
      for (int j = 0; j < 32; ++j)
      {
        Vec3fa p0(20.f * rand01() - 10.f, 20.f * rand01() - 10.f, 20.f * rand01() - 10.f);
        Vec3fa p1(20.f * rand01() - 10.f, 20.f * rand01() - 10.f, 20.f * rand01() - 10.f);
        float offset = 20.f * rand01() - 10.f;

        for (int dim = 0; dim < 3; ++dim) {
          MotionDerivative md(mdc, dim, p0, rand01() > 0.5f ? p1 : p0);
          float tmin = 2.f * rand01() - 1.f;
          float tmax = tmin + rand01();
          Interval1f interval(tmin, tmax);
          unsigned int num_roots = md.findRoots(interval, offset, roots, maxNumRoots);
          if (num_roots >= maxNumRoots) {
            return false;
          }
          for (unsigned int i = 0; i < min(num_roots, maxNumRoots); ++i) {
            if(!valid(roots[i])) return false;
            if(!(roots[i] >= tmin && roots[i] <= tmax)) return false;
          }
        }
      }

      return true;
    }

    bool run ()
    {
      bool passed = true;
#if 0
      std::random_device device;
      size_t seed = device();
#else
      // tests fails for this seed if MOTION_DERIVATIVE_ROOT_EPSILON is 1e-6f instead of 1e-4f
      size_t seed = 2973843361;
#endif
      rng.seed(seed);

      ////////////////////////////////////////////////////////////////////////////////
      // test root solver
      ////////////////////////////////////////////////////////////////////////////////
      unsigned int maxNumRoots = 32;
      float roots[32];
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
            return (t - 1.0) * (t + 1.0);
        }};
        Interval1f I(-2.f, 2.f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 2) return false;
        if (abs(roots[0]+1.f) > 1e6f) return false;
        if (abs(roots[1]-1.f) > 1e6f) return false;
      }
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return (t - 1.0f) * (t + 1.0f) * (t + 0.99999f);
        }};
        Interval1f I(-2.f, 2.f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 2) return false;
      }
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return (t - 1.0f) * (t + 1.0f) * (t + 0.9999f);
        }};
        Interval1f I(-2.f, 2.f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 3) return false;
      }
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return (-7.831048f) + (70.619041f) + (-116.007454f) * t
            + ((-198.235809f) + ( 411.193054f) * t + (-160.020020f) * t * t) * cos(5.438017f * t)
            + ((-320.571472f) + ( 786.181946f) * t + (-559.993164f) * t * t) * sin(5.438017f * t);
        }};
        Interval1f I(0.5, 1.4f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 3) return false;
      }
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return sin(t) * cos(t-1.0f) * 50 * (t-2.0f) * (t-4.0f);
        }};
        Interval1f I(0.0, 7.0f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 7) return false;
      }
      {
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return sin(t) * cos(t-1) * 0.0000001f * (t-2.0f) * (t-4.0f);
        }};
        Interval1f I(0.0, 7.0f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 7) return false;
      }
      {
        // all roots are of the form n*PI/10 for n = 0, 1, 2, ...
        // the function values vary from 0 to 3000
        // in a plot everything from 0 to 0.5 looks like constant zero
        struct EvalFunc { Interval1f operator()(Interval1f const& t) const {
          return sin(10.0f * t) * 100.0f * t*t*t*t*t;
        }};
        Interval1f I(0.0, 2.0f);
        unsigned int numRoots = 0;
        MotionDerivative::findRoots(EvalFunc(), I, numRoots, roots, maxNumRoots);
        if (numRoots != 7) return false;
        for (int i = 0; i < 7; ++i) {
          if (abs(roots[i]-(i*M_PI)/10.f) > 1e-6f)
            return false;
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // test motion derivative coefficients
      ////////////////////////////////////////////////////////////////////////////////
      const size_t numTests = 64;

      // test completely random transformations
      for (int j = 0; j < numTests; ++j)
      {
        AffineSpace3fa M0 = random_quaternion_decomposition();
        AffineSpace3fa M1 = random_quaternion_decomposition();
        if (!testMDC(MotionDerivativeCoefficients(M0, M1))) return false;
        if (!testMDC(MotionDerivativeCoefficients(M0, M0))) return false;
      }

      // test transformations with slightly different rotation angles
      for (int j = 0; j < numTests; ++j)
      {
        Vec3fa axis = random_axis();
        float angle = 2.f * M_PI * (2.f * rand01() - 1.f);
        AffineSpace3fa S(one);
        Vec3fa T(0.f);

        for (int k = 0; k < 16; ++k)
        {
          float eps = pow(0.1f, k);
          Quaternion3f q0 = Quaternion3f::rotate(axis, angle);
          Quaternion3f q1 = Quaternion3f::rotate(axis, angle + (rand01() > 0.5f ? -1.f : 1.f) * k * eps);
          AffineSpace3fa M0 = quaternionDecomposition(T, q0, S);
          AffineSpace3fa M1 = quaternionDecomposition(T, q1, S);
          if (!testMDC(MotionDerivativeCoefficients(M0, M1))) return false;
          if (!testMDC(MotionDerivativeCoefficients(M0, M0))) return false;
        }
      }


      return passed;
    }
  };

  motion_derivative_regression_test motion_derivative_ilter_regression("motion_derivative_regression");
}
