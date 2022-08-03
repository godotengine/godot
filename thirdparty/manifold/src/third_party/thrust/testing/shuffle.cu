#include <thrust/detail/config.h>

#include <map>
#include <limits>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <unittest/unittest.h>

// Functions for performing statistical tests of randomness
// From NIST-Statistical-Test-Suite
// Licence:
//  "This software was developed at the National Institute of Standards and
//  Technology by employees of the Federal Government in the course of their
//  official duties. Pursuant to title 17 Section 105 of the United States Code
//  this software is not subject to copyright protection and is in the public
//  domain. The NIST Statistical Test Suite is an experimental system. NIST
//  assumes no responsibility whatsoever for its use by other parties, and makes
//  no guarantees, expressed or implied, about its quality, reliability, or any
//  other characteristic. We would appreciate acknowledgment if the software is
//  used."
class CephesFunctions {
public:
  static double cephes_igamc(double a, double x) {
    double ans, ax, c, yc, r, t, y, z;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;

    if ((x <= 0) || (a <= 0))
      return (1.0);

    if ((x < 1.0) || (x < a))
      return (1.e0 - cephes_igam(a, x));

    ax = a * log(x) - x - cephes_lgam(a);

    if (ax < -MAXLOG) {
      printf("igamc: UNDERFLOW\n");
      return 0.0;
    }
    ax = exp(ax);

    /* continued fraction */
    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    do {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      yc = y * c;
      pk = pkm1 * z - pkm2 * yc;
      qk = qkm1 * z - qkm2 * yc;
      if (qk != 0) {
        r = pk / qk;
        t = fabs((ans - r) / r);
        ans = r;
      } else
        t = 1.0;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (fabs(pk) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
    } while (t > MACHEP);

    return ans * ax;
  }

private:
  static constexpr double rel_error = 1E-12;

  static constexpr double MACHEP = 1.11022302462515654042E-16;  // 2**-53
  static constexpr double MAXLOG = 7.09782712893383996732224E2; // log(MAXNUM)
  static constexpr double MAXNUM = 1.7976931348623158E308; // 2**1024*(1-MACHEP)
  static constexpr double PI = 3.14159265358979323846;

  static constexpr double big = 4.503599627370496e15;
  static constexpr double biginv = 2.22044604925031308085e-16;

  static int sgngam;

  static double cephes_igam(double a, double x) {
    double ans, ax, c, r;

    if ((x <= 0) || (a <= 0))
      return 0.0;

    if ((x > 1.0) && (x > a))
      return 1.e0 - cephes_igamc(a, x);

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ax = a * log(x) - x - cephes_lgam(a);
    if (ax < -MAXLOG) {
      printf("igam: UNDERFLOW\n");
      return 0.0;
    }
    ax = exp(ax);

    /* power series */
    r = a;
    c = 1.0;
    ans = 1.0;

    do {
      r += 1.0;
      c *= x / r;
      ans += c;
    } while (c / ans > MACHEP);

    return ans * ax / a;
  }

  /* A[]: Stirling's formula expansion of log gamma
   * B[], C[]: log gamma function between 2 and 3
   */
  static constexpr double A[] = {
      0.000811614167470508488140545910738410384510643780,
      -0.000595061904284301438315674115386855191900394857,
      0.000793650340457716942620114419781884862459264696,
      -0.002777777777300996942672073330982129846233874559,
      0.083333333333333189929525985917280195280909538269};
  static constexpr double B[] = {
      -1378.251525691208598800585605204105377197265625,
      -38801.631513463784358464181423187255859375,
      -331612.9927388711948879063129425048828125,
      -1162370.97492762305773794651031494140625,
      -1721737.00820839661173522472381591796875,
      -853555.66424576542340219020843505859375};
  static constexpr double C[] = {
      -351.8157014365234545039129443466663360595703125,
      -17064.21066518811494461260735988616943359375,
      -220528.59055385444662533700466156005859375,
      -1139334.44367982516996562480926513671875,
      -2532523.07177582941949367523193359375,
      -2018891.4143353276886045932769775390625};

  static constexpr double MAXLGM = 2.556348e305;

  /* Logarithm of gamma function */
  static double cephes_lgam(double x) {
    double p, q, u, w, z;
    int i;

    sgngam = 1;

    if (x < -34.0) {
      q = -x;
      w = cephes_lgam(q); /* note this modifies sgngam! */
      p = floor(q);
      if (p == q) {
      lgsing:
        goto loverf;
      }
      i = (int)p;
      if ((i & 1) == 0)
        sgngam = -1;
      else
        sgngam = 1;
      z = q - p;
      if (z > 0.5) {
        p += 1.0;
        z = p - q;
      }
      z = q * sin(PI * z);
      if (z == 0.0)
        goto lgsing;
      /*      z = log(PI) - log( z ) - w;*/
      z = log(PI) - log(z) - w;
      return z;
    }

    if (x < 13.0) {
      z = 1.0;
      p = 0.0;
      u = x;
      while (u >= 3.0) {
        p -= 1.0;
        u = x + p;
        z *= u;
      }
      while (u < 2.0) {
        if (u == 0.0)
          goto lgsing;
        z /= u;
        p += 1.0;
        u = x + p;
      }
      if (z < 0.0) {
        sgngam = -1;
        z = -z;
      } else
        sgngam = 1;
      if (u == 2.0)
        return (log(z));
      p -= 2.0;
      x = x + p;
      p = x * cephes_polevl(x, B, 5) /
          cephes_p1evl(x, C, 6);

      return log(z) + p;
    }

    if (x > MAXLGM) {
    loverf:
      printf("lgam: OVERFLOW\n");

      return sgngam * MAXNUM;
    }

    q = (x - 0.5) * log(x) - x + log(sqrt(2 * PI));
    if (x > 1.0e8)
      return q;

    p = 1.0 / (x * x);
    if (x >= 1000.0)
      q +=
          ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p +
           0.0833333333333333333333) /
          x;
    else
      q += cephes_polevl(p, A, 4) / x;

    return q;
  }

  static double cephes_polevl(double x, const double *coef, int N) {
    const double *p = coef;
    double ans = *p++;
    int i = N;
    do
      ans = ans * x + *p++;
    while (--i);

    return ans;
  }

  static double cephes_p1evl(double x, const double *coef, int N) {
    const double *p = coef;
    double ans = x + *p++;
    int i = N - 1;

    do
      ans = ans * x + *p++;
    while (--i);

    return ans;
  }

  static double cephes_erf(double x) {
    static const double two_sqrtpi = 1.128379167095512574;
    double sum = x, term = x, xsqr = x * x;
    int j = 1;

    if (fabs(x) > 2.2)
      return 1.0 - cephes_erfc(x);

    do {
      term *= xsqr / j;
      sum -= term / (2 * j + 1);
      j++;
      term *= xsqr / j;
      sum += term / (2 * j + 1);
      j++;
    } while (fabs(term) / sum > rel_error);

    return two_sqrtpi * sum;
  }

  static double cephes_erfc(double x) {
    static const double one_sqrtpi = 0.564189583547756287;
    double a = 1, b = x, c = x, d = x * x + 0.5;
    double q1, q2 = b / d, n = 1.0, t;

    if (fabs(x) < 2.2)
      return 1.0 - cephes_erf(x);
    if (x < 0)
      return 2.0 - cephes_erfc(-x);

    do {
      t = a * n + b * x;
      a = b;
      b = t;
      t = c * n + d * x;
      c = d;
      d = t;
      n += 0.5;
      q1 = q2;
      q2 = b / d;
    } while (fabs(q1 - q2) / q2 > rel_error);

    return one_sqrtpi * exp(-x * x) * q2;
  }

  static double cephes_normal(double x) {
    double arg, result, sqrt2 = 1.414213562373095048801688724209698078569672;

    if (x > 0) {
      arg = x / sqrt2;
      result = 0.5 * (1 + erf(arg));
    } else {
      arg = -x / sqrt2;
      result = 0.5 * (1 - erf(arg));
    }

    return (result);
  }
};
int CephesFunctions::sgngam = 0;
constexpr double CephesFunctions::A[];
constexpr double CephesFunctions::B[];
constexpr double CephesFunctions::C[];

template <typename Vector>
void TestShuffleSimple() {
  Vector data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 2;
  data[3] = 3;
  data[4] = 4;
  Vector shuffled(data.begin(), data.end());
  thrust::default_random_engine g(2);
  thrust::shuffle(shuffled.begin(), shuffled.end(), g);
  thrust::sort(shuffled.begin(), shuffled.end());
  // Check all of our data is present
  // This only tests for strange conditions like duplicated elements
  ASSERT_EQUAL(shuffled, data);
}
DECLARE_VECTOR_UNITTEST(TestShuffleSimple);

template <typename Vector>
void TestShuffleCopySimple() {
  Vector data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 2;
  data[3] = 3;
  data[4] = 4;
  Vector shuffled(5);
  thrust::default_random_engine g(2);
  thrust::shuffle_copy(data.begin(), data.end(), shuffled.begin(), g);
  g.seed(2);
  thrust::shuffle(data.begin(), data.end(), g);
  ASSERT_EQUAL(shuffled, data);
}
DECLARE_VECTOR_UNITTEST(TestShuffleCopySimple);

template <typename T>
void TestHostDeviceIdentical(size_t m) {
  thrust::host_vector<T> host_result(m);
  thrust::host_vector<T> device_result(m);
  thrust::sequence(host_result.begin(), host_result.end(), T{});
  thrust::sequence(device_result.begin(), device_result.end(), T{});

  thrust::default_random_engine host_g(183);
  thrust::default_random_engine device_g(183);

  thrust::shuffle(host_result.begin(), host_result.end(), host_g);
  thrust::shuffle(device_result.begin(), device_result.end(), device_g);

  ASSERT_EQUAL(device_result, host_result);
}
DECLARE_VARIABLE_UNITTEST(TestHostDeviceIdentical);

template <typename T>
void TestFunctionIsBijection(size_t m) {
  thrust::default_random_engine host_g(0xD5);
  thrust::default_random_engine device_g(0xD5);

  thrust::system::detail::generic::feistel_bijection host_f(m, host_g);
  thrust::system::detail::generic::feistel_bijection device_f(m, device_g);

  if (static_cast<double>(host_f.nearest_power_of_two()) >= static_cast<double>(std::numeric_limits<T>::max()) || m == 0) {
    return;
  }

  thrust::host_vector<T> host_result(host_f.nearest_power_of_two());
  thrust::host_vector<T> device_result(device_f.nearest_power_of_two());
  thrust::sequence(host_result.begin(), host_result.end(), T{});
  thrust::sequence(device_result.begin(), device_result.end(), T{});

  thrust::transform(host_result.begin(), host_result.end(), host_result.begin(),
                    host_f);
  thrust::transform(device_result.begin(), device_result.end(),
                    device_result.begin(), device_f);

  ASSERT_EQUAL(host_result, device_result);

  thrust::sort(host_result.begin(), host_result.end());
  // Assert all values were generated exactly once
  for (uint64_t i = 0; i < m; i++) {
    ASSERT_EQUAL((uint64_t)host_result[i], i);
  }
}
DECLARE_VARIABLE_UNITTEST(TestFunctionIsBijection);

void TestBijectionLength() {
  thrust::default_random_engine g(0xD5);

  uint64_t m = 31;
  thrust::system::detail::generic::feistel_bijection f(m, g);
  ASSERT_EQUAL(f.nearest_power_of_two(), uint64_t(32));

  m = 32;
  f = thrust::system::detail::generic::feistel_bijection(m, g);
  ASSERT_EQUAL(f.nearest_power_of_two(), uint64_t(32));

  m = 1;
  f = thrust::system::detail::generic::feistel_bijection(m, g);
  ASSERT_EQUAL(f.nearest_power_of_two(), uint64_t(16));
}
DECLARE_UNITTEST(TestBijectionLength);

// Individual input keys should be permuted to output locations with uniform
// probability. Perform chi-squared test with confidence 99.9%.
template <typename Vector>
void TestShuffleKeyPosition() {
  typedef typename Vector::value_type T;
  size_t m = 20;
  size_t num_samples = 100;
  thrust::host_vector<size_t> index_sum(m, 0);
  thrust::host_vector<T> sequence(m);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));

  thrust::default_random_engine g(0xD5);
  for (size_t i = 0; i < num_samples; i++) {
    Vector shuffled(sequence.begin(), sequence.end());
    thrust::shuffle(shuffled.begin(), shuffled.end(), g);
    thrust::host_vector<T> tmp(shuffled.begin(), shuffled.end());

    for (auto j = 0ull; j < m; j++) {
      index_sum[tmp[j]] += j;
    }
  }

  double expected_average_position = static_cast<double>(m - 1) / 2;
  double chi_squared = 0.0;
  for (auto j = 0ull; j < m; j++) {
    double average_position = static_cast<double>(index_sum[j]) / num_samples;
    chi_squared += std::pow(expected_average_position - average_position, 2) /
                   expected_average_position;
  }
  // Tabulated chi-squared critical value for m-1=19 degrees of freedom
  // and 99.9% confidence
  double confidence_threshold = 43.82;
  ASSERT_LESS(chi_squared, confidence_threshold);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleKeyPosition);

struct vector_compare {
  template <typename VectorT>
  bool operator()(const VectorT &a, const VectorT &b) const {
    for (auto i = 0ull; i < a.size(); i++) {
      if (a[i] < b[i])
        return true;
      if (a[i] > b[i])
        return false;
    }
    return false;
  }
};

// Brute force check permutations are uniformly distributed on small input
// Uses a chi-squared test indicating 99% confidence the output is uniformly
// random
template <typename Vector>
void TestShuffleUniformPermutation() {
  typedef typename Vector::value_type T;

  size_t m = 5;
  size_t num_samples = 1000;
  size_t total_permutations = 1 * 2 * 3 * 4 * 5;
  std::map<thrust::host_vector<T>, size_t, vector_compare> permutation_counts;
  Vector sequence(m);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));
  thrust::default_random_engine g(0xD5);
  for (auto i = 0ull; i < num_samples; i++) {
    thrust::shuffle(sequence.begin(), sequence.end(), g);
    thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
    permutation_counts[tmp]++;
  }

  ASSERT_EQUAL(permutation_counts.size(), total_permutations);

  double chi_squared = 0.0;
  double expected_count = static_cast<double>(num_samples) / total_permutations;
  for (auto kv : permutation_counts) {
    chi_squared += std::pow(expected_count - kv.second, 2) / expected_count;
  }
  double p_score = CephesFunctions::cephes_igamc(
      (double)(total_permutations - 1) / 2.0, chi_squared / 2.0);
  ASSERT_GREATER(p_score, 0.01);
}
DECLARE_VECTOR_UNITTEST(TestShuffleUniformPermutation);

template <typename Vector>
void TestShuffleEvenSpacingBetweenOccurances() {
  typedef typename Vector::value_type T;
  const uint64_t shuffle_size = 10;
  const uint64_t num_samples = 1000;

  thrust::host_vector<T> h_results;
  Vector sequence(shuffle_size);
  thrust::sequence(sequence.begin(), sequence.end(), 0);
  thrust::default_random_engine g(0xD6);
  for (auto i = 0ull; i < num_samples; i++) {
    thrust::shuffle(sequence.begin(), sequence.end(), g);
    thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
    h_results.insert(h_results.end(), sequence.begin(), sequence.end());
  }

  std::vector<std::vector<std::vector<uint64_t>>> distance_between(
      num_samples, std::vector<std::vector<uint64_t>>(
                       num_samples, std::vector<uint64_t>(shuffle_size, 0)));

  for (uint64_t sample = 0; sample < num_samples; sample++) {
    for (uint64_t i = 0; i < shuffle_size - 1; i++) {
      for (uint64_t j = 1; j < shuffle_size - i; j++) {
        T val_1 = h_results[sample * shuffle_size + i];
        T val_2 = h_results[sample * shuffle_size + i + j];
        distance_between[val_1][val_2][j]++;
        distance_between[val_2][val_1][shuffle_size - j]++;
      }
    }
  }

  const double expected_occurances = (double)num_samples / (shuffle_size - 1);
  for (uint64_t val_1 = 0; val_1 < shuffle_size; val_1++) {
    for (uint64_t val_2 = val_1 + 1; val_2 < shuffle_size; val_2++) {
      double chi_squared = 0.0;
      auto &distances = distance_between[val_1][val_2];
      for (uint64_t i = 1; i < shuffle_size; i++) {
        chi_squared += std::pow((double)distances[i] - expected_occurances, 2) /
                       expected_occurances;
      }

      double p_score = CephesFunctions::cephes_igamc(
          (double)(shuffle_size - 2) / 2.0, chi_squared / 2.0);
      ASSERT_GREATER(p_score, 0.01);
    }
  }
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleEvenSpacingBetweenOccurances);

template <typename Vector>
void TestShuffleEvenDistribution() {
  typedef typename Vector::value_type T;
  const uint64_t shuffle_sizes[] = {10, 100, 500};
  thrust::default_random_engine g(0xD5);
  for (auto shuffle_size : shuffle_sizes) {
    if(shuffle_size > (uint64_t)std::numeric_limits<T>::max())
      continue;
    const uint64_t num_samples = shuffle_size == 500 ? 1000 : 200;

    std::vector<uint64_t> counts(shuffle_size * shuffle_size, 0);
    Vector sequence(shuffle_size);
    for (auto i = 0ull; i < num_samples; i++) {
      thrust::sequence(sequence.begin(), sequence.end(), 0);
      thrust::shuffle(sequence.begin(), sequence.end(), g);
      thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
      for (uint64_t j = 0; j < shuffle_size; j++) {
        assert(j < tmp.size());
        counts.at(j * shuffle_size + tmp[j])++;
      }
    }

    const double expected_occurances = (double)num_samples / shuffle_size;
    for (uint64_t i = 0; i < shuffle_size; i++) {
      double chi_squared_pos = 0.0;
      double chi_squared_num = 0.0;
      for (uint64_t j = 0; j < shuffle_size; j++) {
        auto count_pos = counts.at(i * shuffle_size + j);
        auto count_num = counts.at(j * shuffle_size + i);
        chi_squared_pos +=
            pow((double)count_pos - expected_occurances, 2) / expected_occurances;
        chi_squared_num +=
            pow((double)count_num - expected_occurances, 2) / expected_occurances;
      }

      double p_score_pos = CephesFunctions::cephes_igamc(
          (double)(shuffle_size - 1) / 2.0, chi_squared_pos / 2.0);
      ASSERT_GREATER(p_score_pos, 0.001 / (double)shuffle_size);

      double p_score_num = CephesFunctions::cephes_igamc(
          (double)(shuffle_size - 1) / 2.0, chi_squared_num / 2.0);
      ASSERT_GREATER(p_score_num, 0.001 / (double)shuffle_size);
    }
  }
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleEvenDistribution);
