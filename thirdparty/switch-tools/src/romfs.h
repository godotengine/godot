#pragma once

#include "filepath.h"

size_t build_romfs_by_paths(char *dir, char *out_fn);

size_t build_romfs_by_path_into_file(char *dir, FILE *f_out, off_t base_offset);