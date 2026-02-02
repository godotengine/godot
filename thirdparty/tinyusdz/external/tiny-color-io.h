#ifndef TINY_COLOR_IO_H_
#define TINY_COLOR_IO_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tinycolorio {

template <typename T>
class LUT3D {
 public:
  LUT3D() : x_dim_(0), y_dim_(0), z_dim_(0) {}

  ~LUT3D() {}

  void create(size_t x_dim, size_t y_dim, size_t z_dim) {
    size_t len = x_dim * y_dim * z_dim;
    data_.resize(3 * len);

    x_dim_ = x_dim;
    y_dim_ = y_dim;
    z_dim_ = z_dim;
  }

  void set(size_t x, size_t y, size_t z, const T val[3]) {
    if ((x < x_dim_) && (y < y_dim_) && (z < z_dim_)) {
      size_t idx = (x_dim_ * y_dim_) * z + x_dim_ * y + x;
      data_[3 * idx + 0] = val[0];
      data_[3 * idx + 1] = val[1];
      data_[3 * idx + 2] = val[2];
    }
  }

  void set(size_t x, size_t y, size_t z, const T r_val, const T g_val,
           const T b_val) {
    if ((x < x_dim_) && (y < y_dim_) && (z < z_dim_)) {
      size_t idx = (x_dim_ * y_dim_) * z + x_dim_ * y + x;
      data_[3 * idx + 0] = r_val;
      data_[3 * idx + 1] = g_val;
      data_[3 * idx + 2] = b_val;
    }
  }

  void get(size_t x, size_t y, size_t z, T val[3]) const {
    if ((x < x_dim_) && (y < y_dim_) && (z < z_dim_)) {
      size_t idx = (x_dim_ * y_dim_) * z + x_dim_ * y + x;
      val[0] = data_[3 * idx + 0];
      val[1] = data_[3 * idx + 1];
      val[2] = data_[3 * idx + 2];
    }
  }

  size_t x_dim() { return x_dim_; }

  size_t y_dim() { return y_dim_; }

  size_t z_dim() { return z_dim_; }

  size_t x_dim_;
  size_t y_dim_;
  size_t z_dim_;

  std::vector<T> data_;  // RGB
};

using LUT3Df = LUT3D<float>;

///
/// Loads SPI3D LUT data(ASCII) from a file
///
/// @param[in] filename spi3d LUT filename.
/// @param[out] lut 3D LUT table.
/// @param[out] err Error message(when failed to load a LUT).
/// @return true upon succes.
///
bool LoadSPI3DFromFile(const std::string &filename, LUT3Df *lut,
                       std::string *err = nullptr);

///
/// Loads SPI3D LUT data(ASCII) from memory
///
/// @param[in] addr Buffer address
/// @param[in] sz Buffer size
/// @param[out] lut 3D LUT table.
/// @param[out] err Error message(when failed to load a LUT).
/// @return true upon succes.
///
bool LoadSPI3DFromMemory(const uint8_t *addr, const size_t sz, LUT3Df *lut,
                       std::string *err = nullptr);

}  // namespace tinycolorio

#endif  // TINY_COLOR_IO_H_

#ifdef TINY_COLOR_IO_IMPLEMENTATION

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace tinycolorio {

bool LoadSPI3DFromFile(const std::string &filename, LUT3Df *lut,
                       std::string *err) {
  std::ifstream ifs(filename);

  if (!ifs) {
    if (err) {
      (*err) = "Failed to open file : " + filename;
    }
    return false;
  }

  // header
  std::string line;
  std::getline(ifs, line);

  std::string buf;

  // lower string.
  for (auto c : line) {
    buf.push_back(static_cast<char>(std::tolower(c)));
  }
  // std::transform(line.begin(), line.end(), buf.begin(), std::tolower);

  if (buf.find("spilut") == std::string::npos) {
    if (err) {
      (*err) = "Not a SPILUT format. header = " + line;
    }
    return false;
  }

  // ignore 2nd line(assuming 3 3)
  std::getline(ifs, line);

  // lut size
  std::getline(ifs, line);

  int x_size = 0, y_size = 0, z_size = 0;
  if (3 != sscanf(line.c_str(), "%d %d %d", &x_size, &y_size, &z_size)) {
    if (err) {
      (*err) = "Error while reading lut size";
    }
    return false;
  }

  lut->create(size_t(x_size), size_t(y_size), size_t(z_size));

  int read_count = x_size * y_size * z_size;

  while (ifs.good() && read_count > 0) {
    std::getline(ifs, line);

    int x_idx = 0, y_idx = 0, z_idx = 0;
    float r_value = 0.0f, g_value = 0.0f, b_value = 0.0f;

    if (sscanf(line.c_str(), "%d %d %d %f %f %f", &x_idx, &y_idx, &z_idx,
               &r_value, &g_value, &b_value) == 6) {
      lut->set(size_t(x_idx), size_t(y_idx), size_t(z_idx), r_value, g_value,
               b_value);
    }
  }

  return true;
}

}  // namespace tinycolorio

#endif  // TINY_COLOR_IO_IMPLEMENTATION
