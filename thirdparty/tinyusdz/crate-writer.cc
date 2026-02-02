#include <limits>
#include <cstdint>
#include <cstring>

#include "crate-writer.hh"
#include "value-types.hh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// TODO: Use std:: version for C++17
#include "nonstd/optional.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace tinyusdz {
namespace crate {

namespace {

struct FltBit {
  union {
    float f;
    uint32_t i;
  };
};

struct DblBit {
  union {
    double f;
    uint64_t i;
  };
};

template<typename T>
bool Compare(const T &lhs, const T &rhs) {
  return lhs == rhs;
}
 
// Use bitfield comparison for floating-point value,
// This may give slightly different result compared to pxrUSD implementation
// (which uses `==` for floating point comparison)
template<>
bool Compare(const double &lhs, const double &rhs) {
  DblBit a; a.f = lhs;
  DblBit b; b.f = rhs;

  return (a.i == b.i);
}

template<>
bool Compare(const float &lhs, const float &rhs) {
  FltBit a; a.f = lhs;
  FltBit b; b.f = rhs;

  return (a.i == b.i);
}

} // namespace

// IsExactlyRepresented in pxrUSD.
template<typename Tfrom, typename Tto>
nonstd::optional<Tto> TryExactlyRepresentable(const Tfrom &from) {
  // NOTE: pxrUSD uses lowest() for minval, not min()
  Tfrom minval = static_cast<Tfrom>(std::numeric_limits<Tto>::lowest());
  Tfrom maxval = static_cast<Tfrom>(std::numeric_limits<Tto>::max());

  if (from < minval) {
    return nonstd::nullopt;
  }

  if (from > maxval) {
    return nonstd::nullopt;
  }

  // Identity check.
  Tto vt = static_cast<Tto>(from);
  if (Compare(static_cast<Tfrom>(vt), from)) {
    return vt;
  }

  return nonstd::nullopt;
}

// NOTE `Inline` payload is 6bytes.
//
// - Inlineable value
//   - double as float format
//   - (u)int64 as (u)int32
//   - vector as int8 x N (n = 2, 3 or 4)
//   - Diagonal matrix as int8 x N  (n = 2, 3 or 4)
//   - empty dictionary

inline nonstd::optional<uint32_t> TryEncodeInline(double v) {
  uint32_t dst;

  nonstd::optional<float> f = TryExactlyRepresentable<double, float>(v);

  if (f) {
    memcpy(&dst, &f.value(), sizeof(float));
    return dst;
  }

  return nonstd::nullopt;
}

inline nonstd::optional<uint32_t> TryEncodeInline(uint64_t v) {
  uint32_t dst;

  nonstd::optional<uint32_t> f = TryExactlyRepresentable<uint64_t, uint32_t>(v);

  if (f) {
    dst = f.value();
    return dst;
  }

  return nonstd::nullopt;
}

inline nonstd::optional<uint32_t> TryEncodeInline(int64_t v) {
  uint32_t dst;

  nonstd::optional<int32_t> f = TryExactlyRepresentable<int64_t, int32_t>(v);

  if (f) {
    memcpy(&dst, &f.value(), sizeof(int32_t));
    return dst;
  }

  return nonstd::nullopt;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::vector3f v) {
  uint32_t dst;

  // Check if each component of the vector can be represented by int8.
  std::array<int8_t, 3> ivec;
  for (size_t i = 0; i < 3; i++) {
    if (auto f = TryExactlyRepresentable<float, int8_t>(v[i])) {
      ivec[i] = f.value();
    } else {
      return nonstd::nullopt;
    }
  }

  memcpy(&dst, &ivec[0], sizeof(ivec));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::vector3d v) {
  uint32_t dst;

  // Check if each component of the vector can be represented by int8.
  std::array<int8_t, 3> ivec;
  for (size_t i = 0; i < 3; i++) {
    if (auto f = TryExactlyRepresentable<double, int8_t>(v[i])) {
      ivec[i] = f.value();
    } else {
      return nonstd::nullopt;
    }
  }

  memcpy(&dst, &ivec[0], sizeof(ivec));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::color4f v) {
  uint32_t dst;

  // Check if each component of the vector can be represented by int8.
  std::array<int8_t, 4> ivec;
  for (size_t i = 0; i < 4; i++) {
    if (auto f = TryExactlyRepresentable<float, int8_t>(v[i])) {
      ivec[i] = f.value();
    } else {
      return nonstd::nullopt;
    }
  }

  memcpy(&dst, &ivec[0], sizeof(ivec));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::color4d v) {
  uint32_t dst;

  // Check if each component of the vector can be represented by int8.
  std::array<int8_t, 4> ivec;
  for (size_t i = 0; i < 4; i++) {
    if (auto f = TryExactlyRepresentable<double, int8_t>(v[i])) {
      ivec[i] = f.value();
    } else {
      return nonstd::nullopt;
    }
  }

  memcpy(&dst, &ivec[0], sizeof(ivec));
  return dst;
}

//
// TODO: Implement more TryEncodeInline for scalar value types
//

inline nonstd::optional<uint32_t> TryEncodeInline(value::matrix2d v) {
  uint32_t dst;

  // Check if a matrix is a diagonal matrix and its diagonal component can be represented by int8.
  std::array<int8_t, 2> diag;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      if (i == j) {
        // diag
        if (auto f = TryExactlyRepresentable<double, int8_t>(v.m[i][j])) {
          diag[i] = f.value();
        } else {
          return nonstd::nullopt;
        }
      } else {
        if (!Compare(v.m[i][j], 0.0)) {
          return nonstd::nullopt;
        }
      }
    }
  }

  memcpy(&dst, &diag[0], sizeof(diag));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::matrix3d v) {
  uint32_t dst;

  // Check if a matrix is a diagonal matrix and its diagonal component can be represented by int8.
  std::array<int8_t, 3> diag;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      if (i == j) {
        // diag
        if (auto f = TryExactlyRepresentable<double, int8_t>(v.m[i][j])) {
          diag[i] = f.value();
        } else {
          return nonstd::nullopt;
        }
      } else {
        if (!Compare(v.m[i][j], 0.0)) {
          return nonstd::nullopt;
        }
      }
    }
  }

  memcpy(&dst, &diag[0], sizeof(diag));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::matrix4d v) {
  uint32_t dst;

  // Check if a matrix is a diagonal matrix and its diagonal component can be represented by int8.
  std::array<int8_t, 4> diag;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      if (i == j) {
        // diag
        if (auto f = TryExactlyRepresentable<double, int8_t>(v.m[i][j])) {
          diag[i] = f.value();
        } else {
          return nonstd::nullopt;
        }
      } else {
        if (!Compare(v.m[i][j], 0.0)) {
          return nonstd::nullopt;
        }
      }
    }
  }

  memcpy(&dst, &diag[0], sizeof(diag));
  return dst;
}

inline nonstd::optional<uint32_t> TryEncodeInline(value::dict v) {
  uint32_t dst{0};

  if (v.empty()) {
    return dst;
  }

  return nonstd::nullopt;
}

} // namespace crate 
} // namespace tinyusdz
