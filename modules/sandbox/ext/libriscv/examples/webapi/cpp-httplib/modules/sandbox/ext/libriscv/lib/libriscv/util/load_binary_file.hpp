#pragma once

#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

inline std::vector<uint8_t> load_binary_file(const std::string& filename) {
  size_t size = 0;

  FILE* f = fopen(filename.c_str(), "rb");
  if (f == NULL) throw std::runtime_error("Could not open file: " + filename);

  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);

  std::vector<uint8_t> result(size);

  if (size != fread(result.data(), 1, size, f)) {
    fclose(f);
    throw std::runtime_error("Error when reading from file: " + filename);
  }

  fclose(f);

  return result;
}
