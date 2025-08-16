/*
LodePNG pngdetail

Copyright (c) 2005-2025 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

//g++ pngdetail.cpp lodepng_util.cpp lodepng.cpp -ansi -pedantic -Wall -Wextra -o pngdetail -O3


/*
Utility program that shows a lot of information in the console about a PNG file,
including color type, text chunks, the names and sizes of all chunks in the
image, exif, and ICC profile, etc...
*/

#include "lodepng.h"
#include "lodepng_util.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <inttypes.h>

void showHelp() {
  std::cout << "pngdetail by Lode Vandevenne" << std::endl;
  std::cout << "version: " << LODEPNG_VERSION_STRING << std::endl;
  std::cout << "Shows detailed information about a PNG image, its compression and possible corruptions.\n"
               "Usage: pngdetail [filename] [options]...\n"
               "Without options shows a default set of stats. With options, shows only selected options.\n"
               "E.g. 'pngdetail image.png -plc' to show png info, palette info and chunks\n"
               "Options:\n"
               "-o: show header summary on one line\n"
               "-H: show header info\n"
               "-p: show PNG file info\n"
               "-a: analyze errors or warnings\n"
               "-e: show exif metadata (if the PNG has an eXIf chunk)\n"
               "-E: show exif metadata hex bytes\n"
               "-i: show ICC profile details (if any)\n"
               "-I: show ICC profile hex bytes\n"
               "--format=<format>: display mode for -I:\n"
               "    hex: print bytes in hex\n"
               "    mix: print printable bytes as ASCII characters, hex for others\n"
               "    bin: dump as binary in terminal\n"
               "-l: show palette (if any)\n"
               "-s: show color statistics\n"
               "-r: render the PNG image in terminal (with --mode and --size)\n"
               "--size=<width>: render width for -r\n"
               "--mode=<mode>: render mode for -r:\n"
               "    ascii:   Letters ROYLGTCABVMF indicate hue (L=lime, T=turquoise, A=azure, F=fuchsia, ...).\n"
               "    hex:     CSS hex notation for every pixel.\n"
               "    hex16:   Like hex but shows 16 bits values per channel.\n"
               "    palette: Shows palette index of each pixel, only for palette images.\n"
               "--size=<width>: render width (not used by hex, hex16 or palette):\n"
               "-c: show PNG chunks\n"
               "-C: show PNG chunks (alternate format)\n"
               "-f: show PNG filters\n"
               "-z: show Zlib info\n"
               "-v: be more verbose\n"
               "-t: expand long texts\n"
               "-x: print most integer numbers in hexadecimal (includes e.g. year, num unique colors, ...)\n"
               "-?, --help, -h: show this help" << std::endl;
}

enum RenderMode {
  RM_ASCII,
  RM_HEX, // CSS
  RM_HEX16,
  RM_PAL // palette indices (only rendered if image is palette based)
};

// for displaying ICC profile
enum HexFormat {
  HF_HEX,
  HF_MIX, // hex and ascii
  HF_BIN // bytes as binary data dump
};

struct Options {
  bool verbose;
  bool expand_long_texts;
  bool show_one_line_summary; //show filesize, pixels and color type on single line
  bool show_header; // show only info from the IHDR chunk
  bool show_errors;
  bool show_icc_details; // show ICC color profile details
  bool show_icc_hex; // show ICC color profile in full
  bool show_exif;
  bool show_exif_hex;
  bool show_color_stats;
  bool show_png_info; //show things like filesize, width, height, palette size, ...
  bool show_palette; //show all palette values
  bool show_palette_pixels; //show palette indices of pixels

  HexFormat hexformat;

  bool show_render;
  RenderMode rendermode;
  int rendersize;

  bool show_chunks; //show the PNG chunk names and their lengths
  bool show_chunks2; //alternate form to print chunks
  bool show_filters; //show the PNG filter of each scanline (not supported for interlaced PNGs currently)
  bool show_zlib_info; //show basic zlib info
  bool use_hex; //show some sizes or positions in hexadecimal

  Options() : verbose(false), expand_long_texts(false),
              show_one_line_summary(false), show_header(false), show_errors(false),
              show_icc_details(false), show_icc_hex(false), show_exif(false), show_exif_hex(false),
              show_color_stats(false), show_png_info(false),
              show_palette(false), show_palette_pixels(false),
              hexformat(HF_MIX), show_render(false), rendermode(RM_ASCII), rendersize(80),
              show_chunks(false), show_chunks2(false), show_filters(false),
              show_zlib_info(false), use_hex(false) {
  }
};

unsigned inspect_chunk_by_name(const unsigned char* data, const unsigned char* end,
                               lodepng::State& state, const char type[5]) {
  const unsigned char* p = lodepng_chunk_find_const(data, end, type);
  if(!p) return 0; // not found, but this is not considered an error
  return lodepng_inspect_chunk(&state, p - data, data, end - data);
}

// Lazy loads the raw file, inspected header or entire image as needed
struct Data {
  std::string filename;
  std::vector<unsigned char> buffer;
  std::vector<unsigned char> pixels; // 16-bit
  unsigned w;
  unsigned h;
  lodepng::State state;
  unsigned error;
  bool inspected;
  bool is_png; // not is_icc or is_exif
  bool is_icc; // the file is a raw icc file, not a PNG, only options -i and -I are useful
  bool is_exif; // the file is a raw exif file, not a PNG, only option -e is useful

  Data(const std::string& filename) : filename(filename), error(0), inspected(false), is_png(false), is_icc(false), is_exif(false) {}

  // Load the file if not already loaded
  void loadFile() {
    if(buffer.empty()) {
      error = lodepng::load_file(buffer, filename); //load the image file with given filename
    } else {
      error = 0; // for reloadpixels, reset error if file was already successfully loaded
    }
  }

  // is PNG according to the file signature
  bool isPng() {
    if(buffer.size() < 8) return false;
    return buffer[0] == 137 && buffer[1] == 80 && buffer[2] == 78 && buffer[3] == 71
        && buffer[4] == 13 && buffer[5] == 10 && buffer[6] == 26 && buffer[7] == 10;
  }

  // is probably an ICC profile instead of a PNG image
  bool isIcc() {
    if(isPng()) return false;
    if(buffer.size() < 128) return false;
    size_t size = (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + buffer[3];
    if(size != buffer.size()) return false;
    if(buffer[36] != 'a') return false;
    if(buffer[37] != 'c') return false;
    if(buffer[38] != 's') return false;
    if(buffer[39] != 'p') return false;
    return true;
  }
  // is probably an EXIF file instead of a PNG image
  bool isExif() {
    if(buffer.size() < 8) return false;
    if(buffer[0] == 'M' && buffer[1] == 'M' && buffer[2] == 0 && buffer[3] == 42) return true;
    if(buffer[0] == 'I' && buffer[1] == 'I' && buffer[2] == 42 && buffer[3] == 0) return true;
    return false;
  }

  // Load header info (plus a few more nearby light chunks) if not already loaded, and the file if needed
  void loadInspect() {
    if(inspected) return;
    inspected = true;
    loadFile();
    if(error) return;
    is_png = is_icc = is_exif = false;
    if(isIcc()) {
      lodepng_set_icc(&state.info_png, "<none>", &buffer[0], buffer.size());
      is_icc = true;
    } else if(isExif()) {
      lodepng_set_exif(&state.info_png, &buffer[0], buffer.size());
      is_exif = true;
    } else {
      is_png = true;
      const unsigned char* data = &buffer[0];
      error = lodepng_inspect(&w, &h, &state, data, buffer.size());
      if(error) return;
      // end before first IDAT chunk: do not parse more than first part of file for all this.
      const unsigned char* end = lodepng_chunk_find_const(data, data + buffer.size(), "IDAT");
      if(!end) end = data + buffer.size(); // no IDAT, invalid PNG but extract info anyway
      error = inspect_chunk_by_name(data, end, state, "PLTE");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "tRNS");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "cHRM");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "gAMA");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "sBIT");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "bKGD");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "hIST");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "pHYs");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "iCCP");
      if(error) return;
      error = inspect_chunk_by_name(data, end, state, "eXIf");
      if(error) return;
    }
  }

  // Load the pixels if not already loaded, and the file if needed
  void loadPixels() {
    if(pixels.empty()) reloadPixels();
  }

  void reloadPixels() {
    loadFile();
    if(error) return;
    inspected = true;
    state.info_raw.colortype = LCT_RGBA;
    state.info_raw.bitdepth = 16;
    pixels.clear();
    error = lodepng::decode(pixels, w, h, state, buffer);
  }
};

std::string colorTypeString(LodePNGColorType type) {
  std::string name;
  switch(type) {
    case LCT_GREY: name = "grey"; break;
    case LCT_RGB: name = "RGB"; break;
    case LCT_PALETTE: name = "palette"; break;
    case LCT_GREY_ALPHA: name = "grey+alpha"; break;
    case LCT_RGBA: name = "RGBA"; break;
    default: name = "invalid"; break;
  }
  std::stringstream ss;
  ss << type << " (" << name << ")";
  return ss.str();
}

template<typename T>
T strtoval(const std::string& s) {
  std::istringstream sstream(s);
  T val;
  sstream >> val;
  return val;
}


/*
Display the names and sizes of all chunks in the PNG file.
*/
void displayChunkNames(Data& data, const Options& options) {
  data.loadInspect();
  if(!data.is_png) return;
  data.loadFile();
  if(data.error) return;
  std::cout << (options.use_hex ? std::hex: std::dec);
  const std::vector<unsigned char>& buffer = data.buffer;
  std::vector<std::string> names;
  std::vector<size_t> sizes;
  unsigned error = lodepng::getChunkInfo(names, sizes, buffer);
  if(error) {
    if(!names.empty() && names.back() == "IEND" && sizes.back() == 0) {
      std::cout << "Corruption or superfluous data detected after the IEND chunk" << std::endl;
    } else {
      std::cout << "Error while identifying chunks. Listing identified chunks anyway." << std::endl;
    }
  }

  if(options.show_chunks2) {
    std::cout << "Chunk types: ";
    for(size_t i = 0; i < names.size(); i++) std::cout << names[i] << " ";
    std::cout << std::endl;
    std::cout << "Chunk sizes: ";
    for(size_t i = 0; i < sizes.size(); i++) std::cout << sizes[i] << " ";
    std::cout << std::endl;
  } else {
    std::cout << "Chunks (type: lengths):";
    std::string last_type;
    for(size_t i = 0; i < names.size(); i++) {
      if(last_type != names[i]) {
        std::cout << std::endl;
        std::cout << " " << names[i] << ": ";
      }
      last_type = names[i];

      std::cout << sizes[i] << " ";
    }
    std::cout << std::endl;
  }

  std::map<std::string, bool> typedict;
  for(size_t i = 0; i < names.size(); i++) {
    typedict[names[i]] = true;
  }

  if(!error) {
    if(!typedict["IHDR"]) std::cout << "Error: no IHDR chunk" << std::endl;
    if(!typedict["IDAT"]) std::cout << "Error: no IDAT chunk" << std::endl;
    if(!typedict["IEND"]) std::cout << "Error: no IEND chunk" << std::endl;
  }
}

void RGBtoHSL(unsigned char r, unsigned char g, unsigned char b, unsigned char* h, unsigned char* s, unsigned char* l) {
  int cmax = std::max<int>(r, std::max<int>(g, b));
  int cmin = std::min<int>(r, std::min<int>(g, b));
  if(cmin == cmax) {
    *h = *s = 0;
    *l = r;
  } else {
    int sum = cmin + cmax;
    int diff = cmax - cmin;
    *l = sum / 2;
    *s = 255 * diff / ((*l < 128) ? sum : (512 - sum));
    int hi = (r == cmax) ? (255 * (g - b) / diff) : ((g == cmax) ? (512 + 255 * (b - r) / diff) : (1024 + 255 * (r - g) / diff));
    *h = ((hi / 6) & 255);
  }
}

/*
HCT: Hue, Chroma, Tone: returns a linear combination between a pure hue and a greyscale value.
*) Chroma: The linear combination factor: 255 for pure hue, 0 for pure greyscale
*) Tone: greyscale to mix with: 0 = black (shade), 255 = white (tint), in between = grey (tone)
*/
void RGBtoHCT(unsigned char r, unsigned char g, unsigned char b, unsigned char* h, unsigned char* c, unsigned char* t) {
  int cmax = std::max<int>(r, std::max<int>(g, b));
  int cmin = std::min<int>(r, std::min<int>(g, b));
  RGBtoHSL(r, g, b, h, c, t);
  *c = cmax - cmin;
  *t = *c == 255 ? 0 : 255 * cmin / (255 + cmin - cmax);
}

// add 32 to get small letter instead of capital
char HueToLetter(int h) {
  char hl = 'R';
  // 12 unique hue letters for 30 degree increment hues.
  if(h < 11 || h >= 244) hl = 'R';  // red
  else if(h >= 11 && h < 32) hl = 'O';  // orange
  else if(h >= 32 && h < 53) hl = 'Y';  // yellow
  else if(h >= 53 && h < 74) hl = 'L';  // lime (officialy "chartreuse" but c is for cyan)
  else if(h >= 74 && h < 96) hl = 'G';  // green
  else if(h >= 96 && h < 117) hl = 'T';  // turquoise (officially "spring green" but that name overlaps green)
  else if(h >= 117 && h < 138) hl = 'C';  // cyan
  else if(h >= 138 && h < 159) hl = 'A';  // azure
  else if(h >= 159 && h < 181) hl = 'B';  // blue
  else if(h >= 181 && h < 202) hl = 'V';  // violet
  else if(h >= 202 && h < 223) hl = 'M';  // magenta
  else if(h >= 223 && h < 244) hl = 'F';  // fuchsia (officially "rose" but r is for red)
  return hl;
}

char lightnessToLetter(int l) {
  int c = ' ';
  if(l < 16) c = ' ';
  else if(l < 48) c = '.';
  else if(l < 80) c = ':';
  else if(l < 112) c = '-';
  else if(l < 144) c = '!';
  else if(l < 176) c = '*';
  else if(l < 208) c = '+'; // The + looks denser than the * in a terminal...
  else if(l < 240) c = '=';
  else c = '#';
  return c;
}

// Both v and result are assumed in range 0-255
// range is the size of an individual bucket. A value in roughly range [-range/2, range/2) can get added to v.
// E.g. if there are 12 hue letters, give 255/12 = 21 as range
static inline int applyDither(int v, int range, int x, int y, bool wrap) {
  // ordered dithering pattern; ranges from 0-15, so multiply with 17 to have 0-255
  static const int pattern[16] = {0,8,2,10,
                                  12,4,14,6,
                                  3,11,1,9,
                                  15,7,13,5};
  int d = pattern[(x & 3) + 4 * (y & 3)] * 17 - 128; // range: -128 to 127
  if(wrap) return (v + d * range / 256) & 255;
  else return std::max(0, std::min(255, v + d * range / 256));
}

// x and y are to use for dithering
// inverted inverts black and white, for in case black text on white background is used (by default it assumes white text on black background)
char RGBtoLetter(unsigned char r, unsigned char g, unsigned char b, unsigned char a, unsigned x, unsigned y, bool dither = true, bool inverted = false) {
  if(a < 255) {
    r = a * r / 255;
    g = a * g / 255;
    b = a * b / 255;
  }

  if(dither) {
    unsigned char h, c, t;
    RGBtoHCT(r, g, b, &h, &c, &t);
    int l = (std::max(std::max(r, g), b) + std::min(std::min(r, g), b)) / 2;
    if(inverted) {
      l = 255 - l;
      t = 255 - t;
    }
    if(applyDither(c, 254, x, y, false) >= 128) {
      char letter = HueToLetter(applyDither(h, 21, x, y, true));
      bool smallcaps = applyDither(l, 64, x+2, y+2, false) < 80;
      return letter + (smallcaps ? 32 : 0);
    }
    else return lightnessToLetter(applyDither(l, 31, x, y, false));
  } else {
    unsigned char h, s, l;
    RGBtoHSL(r, g, b, &h, &s, &l);
    if(inverted) l = 255 - l;

    char hl = HueToLetter(h);
    char c = ' ';
    if(l < 24 || l > 232 || s < 64) {
      c = lightnessToLetter(l);
    } else {
      if(l < 128) c = hl + 32;
      else c = hl;
    }
    return c;
  }
}

std::vector<unsigned char> rescale(const std::vector<unsigned char>& in,
                                   int w0, int h0, int w1, int h1, bool smooth) {
  int numchannels = in.size() / (w0 * h0);
  std::vector<unsigned char> out(w1 * h1 * numchannels);
  if(smooth) {
    // box filter.
    std::vector<unsigned char> temp(w1 * h0 * numchannels);
    for(int c = 0; c < numchannels; c++) {
      for(int x = 0; x < w1; x++) {
        float xaf = x * 1.0 * w0 / w1;
        float xbf = (x + 1.0) * w0 / w1;
        int xa = (int)xaf;
        int xb = (int)xbf;
        double norm = 1.0 / (xbf - xaf);
        xaf -= std::floor(xaf);
        xbf -= std::floor(xbf);
        for(int y = 0; y < h0; y++) {
          int index1 = x * numchannels + y * w1 * numchannels;
          double val = 0;
          for(int x0 = xa; x0 <= xb && x0 < w0; x0++) {
            int index0 = x0 * numchannels + y * w0 * numchannels;
            double v = 1;
            if(x0 == xa) v -= xaf;
            if(x0 == xb) v -= (1 - xbf);
            val += v * in[index0 + c];
          }
          temp[index1 + c] = val * norm;
        }
      }
      for(int y = 0; y < h1; y++) {
        float yaf = y * 1.0 * h0 / h1;
        float ybf = (y + 1.0) * h0 / h1;
        int ya = (int)yaf;
        int yb = (int)ybf;
        double norm = 1.0 / (ybf - yaf);
        yaf -= std::floor(yaf);
        ybf -= std::floor(ybf);
        for(int x = 0; x < w1; x++) {
          int index1 = x * numchannels + y * w1 * numchannels;
          double val = 0;
          for(int y0 = ya; y0 <= yb && y0 < h0; y0++) {
            int index0 = x * numchannels + y0 * w1 * numchannels;
            double v = 1;
            if(y0 == ya) v -= yaf;
            if(y0 == yb) v -= (1 - ybf);
            val += v * temp[index0 + c];
          }
          out[index1 + c] = val * norm;
        }
      }
    }
  } else {
    for(int y = 0; y < h1; y++) {
      int y0 = (int)((y + 0.5) * h0 / h1 - 0.5);
      for (int x = 0; x < w1; x++) {
        int x0 = (int)((x + 0.5) * w0 / w1 - 0.5);
        int index0 = x0 * numchannels + y0 * w0 * numchannels;
        int index1 = x * numchannels + y * w1 * numchannels;
        for (int c = 0; c < numchannels; c++) {
          out[index1 + c] = in[index0 + c];
        }
      }
    }
  }
  return out;
}

/*
Show ASCII art preview of the image
Image is given in 16-bit big endian (because only one format, with the max possible precision, is used throughout this)
*/
void displayAsciiArt(const std::vector<unsigned char>& image, unsigned w, unsigned h, unsigned asciiw) {
  const std::vector<unsigned char>* imagep = &image;
  std::vector<unsigned char> image2;
  if(asciiw < w) {
    unsigned w2 = asciiw;
    unsigned h2 = h * w2 / w;
    image2 = rescale(image, w, h, w2, h2, true);
    imagep = &image2;
    w = w2;
    h = h2;
  }
  if(w > 0 && h > 0) {
    std::cout << "ASCII Art Preview: " << std::endl;
    unsigned h2 = 1 + ((h - 1) * 4) / 7; //compensate for non-square characters in terminal
    std::cout << '+';
    for(unsigned x = 0; x < w; x++) std::cout << '-';
    std::cout << '+' << std::endl;
    for(unsigned y = 0; y < h2; y++) {
      std::cout << "|";
      unsigned y2 = y * h / h2;
      for(unsigned x = 0; x < w; x++) {
        int r = (*imagep)[y2 * w * 8 + x * 8 + 0];
        int g = (*imagep)[y2 * w * 8 + x * 8 + 2];
        int b = (*imagep)[y2 * w * 8 + x * 8 + 4];
        int a = (*imagep)[y2 * w * 8 + x * 8 + 6];
        char symbol = RGBtoLetter(r, g, b, a, x, y, true, false);
        std::cout << (char)symbol;
      }
      std::cout << "|";
      std::cout << std::endl;
    }
    std::cout << '+';
    for(unsigned x = 0; x < w; x++) std::cout << '-';
    std::cout << '+' << std::endl;
  }
}

//sixteen: print 16 bits per pixel
//alpha: print alpha channel
//input image ALWAYS given in 16-bit per channel RGBA
void displayColorsHex(const std::vector<unsigned char>& image, unsigned w, unsigned h, bool sixteen) {
  std::ios_base::fmtflags flags = std::cout.flags();

  if(w > 0 && h > 0) {
    std::cout << "Colors (CSS RGBA hex format):" << std::endl;

    for(unsigned y = 0; y < h; y++) {
      std::cout.flags(flags); //print line numbers in hex or dec whatever it originally was
      std::cout << y << ":";
      for(unsigned x = 0; x < w; x++) {
        size_t index = y * w * 8 + x * 8;
        if (sixteen) {
          int r = image[index + 0] * 256 + image[index + 1];
          int g = image[index + 2] * 256 + image[index + 3];
          int b = image[index + 4] * 256 + image[index + 5];
          int a = image[index + 6] * 256 + image[index + 7];
          std::cout << std::hex << std::setfill('0') << " #" << std::setw(4) << r << std::setw(4) << g << std::setw(4) << b << std::setw(4) << a;
        } else {
          int r = image[index + 0];
          int g = image[index + 2];
          int b = image[index + 4];
          int a = image[index + 6];
          std::cout << std::hex << std::setfill('0') << " #" << std::setw(2) << r << std::setw(2) << g << std::setw(2) << b << std::setw(2) << a;
        }
      }
      std::cout << std::endl;
    }
  }

  std::cout.flags(flags);
}


/*
Show the filtertypes of each scanline in this PNG image.
*/
void displayFilterTypes(Data& data, const Options& options) {
  std::cout << (options.use_hex ? std::hex: std::dec);
  data.loadFile();
  if(data.error) return;
  const std::vector<unsigned char>& buffer = data.buffer;
  std::vector<std::vector<unsigned char> > types;
  unsigned error = lodepng::getFilterTypesInterlaced(types, buffer);
  if(error) {
    std::cout << "Error getting filter types" << std::endl;
    return;
  }

  if(types.size() == 7) {
    std::cout << "Filter types (Adam7 interlaced):" << std::endl;
    for(int j = 0; j < 7; j++) {
      std::cout << " Pass " << (j + 1) << ": ";
      for(size_t i = 0; i < types[j].size(); i++) {
        std::cout << (int)(types[j][i]);
      }
      std::cout << std::endl;
    }
  } else {
    std::cout << "Filter types: ";
    for(size_t i = 0; i < types[0].size(); i++) {
      std::cout << (int)(types[0][i]);
    }
    std::cout << std::endl;
  }
}

//image type MUST be palette
void displayPalette(Data& data, const Options& options) {
  data.loadInspect();
  if(data.error) return;
  std::cout << (options.use_hex ? std::hex: std::dec);

  const LodePNGInfo& info = data.state.info_png;
  const LodePNGColorMode& color = info.color;

  std::cout << "Palette size: " << color.palettesize << std::endl;
  std::cout << "Palette colors: ";
  std::ios_base::fmtflags flags = std::cout.flags();
  std::cout << std::hex << std::setfill('0');
  for(size_t i = 0; i < color.palettesize; i++) {
    unsigned char* p = &color.palette[i * 4];
    std::cout << "#" << std::setw(2) << (int)p[0] << std::setw(2) << (int)p[1] << std::setw(2) << (int)p[2] << std::setw(2) << (int)p[3] << " ";
  }
  std::cout.flags(flags);
  std::cout << std::endl;
}

//image type MUST be palette
void displayPalettePixels(const std::vector<unsigned char>& buffer, const Options& options) {
  unsigned w, h;
  lodepng::State state;
  std::vector<unsigned char> out;
  std::cout << (options.use_hex ? std::hex: std::dec);

  state.decoder.color_convert = 0;

  lodepng::decode(out, w, h, state, buffer);

  if(state.info_png.color.colortype == LCT_PALETTE) {
    if (options.show_color_stats) {
      std::vector<size_t> count(256, 0);
      size_t outofbounds = 0;

      for(size_t i = 0; i < w * h; i++) {
        int value = lodepng::getPaletteValue(&out[0], i, state.info_raw.bitdepth);
        count[value]++;
        if(value >= (int)state.info_raw.palettesize) outofbounds++;
      }

      std::cout << "Palette count: ";
      for(size_t i = 0; i < state.info_raw.palettesize; i++) {
        std::cout << count[i] << " ";
      }
      std::cout << std::endl;

      if(outofbounds > 0) std::cout << "Out of bounds palette values: " << outofbounds << std::endl;
    }

    std::cout << "Pixel palette indices:" << std::endl;
    for(size_t i = 0; i < w * h; i++) {
      int value = lodepng::getPaletteValue(&out[0], i, state.info_raw.bitdepth);
      std::cout << value << ", ";
      if(i % w == w - 1) std::cout << std::endl;
    }
  } else {
    std::cout << "Pixel palette indices: not shown, not a palette image\n" << std::endl;
  }
}

void printZlibInfo(Data& data) {
  data.loadFile();
  if(data.error || !data.isPng()) return;
  const std::vector<unsigned char>& png = data.buffer;

  //Read literal data from all IDAT chunks
  const unsigned char *chunk, *begin, *end;
  end = &png.back() + 1;
  begin = chunk = &png.front() + 8;

  std::vector<unsigned char> compressed;

  while(chunk < end && end - chunk >= 8) {
    char type[5];
    lodepng_chunk_type(type, chunk);
    if(std::string(type).size() != 4) {
      std::cout << "invalid png" << std::endl;
      return;
    }

    if(std::string(type) == "IDAT") {
      const unsigned char* cdata = lodepng_chunk_data_const(chunk);
      unsigned clength = lodepng_chunk_length(chunk);
      if(chunk + clength + 12 > end || clength > png.size() || chunk + clength + 12 < begin) {
        std::cout << "corrupt chunk length" << std::endl;
        return;
      }
      compressed.insert(compressed.end(), cdata, cdata + clength);
    }

    chunk = lodepng_chunk_next_const(chunk, end);
  }

  if(compressed.size() >= 3) {
    int bfinal = compressed[2] & 1;
    int btype = (compressed[2] & 6) >> 1;
    if(bfinal) {
      std::cout << "zlib data stored in a single block, btype: " << btype << std::endl;
    } else {
      std::cout << "zlib data spread over multiple blocks. First block btype: " << btype << std::endl;
    }
  }

  std::cout << "zlib compressed size: " << compressed.size() << std::endl;

  //Decompress all IDAT data (if the while loop ended early, this might fail)
  std::vector<unsigned char> uncompressed;
  data.error = lodepng::decompress(uncompressed, compressed.empty() ? NULL : &compressed[0], compressed.size());
  if(data.error) return;

  std::cout << "zlib uncompressed size: " << uncompressed.size() << std::endl;
}

// returns number of unique RGBA colors in the image
// also fills unique r, g, b, a counts in the output parameters
// the input image is in 16-bit per channel color, so 8 chars per pixel
size_t countColors(const std::vector<unsigned char>& image, unsigned w, unsigned h,
    size_t* ro, size_t* go, size_t* bo, size_t* ao) {
  typedef std::pair<std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short> > RGBA;
  std::map<RGBA, size_t> rgbam;
  //std::map<uint64_t, size_t> rgbam;
  std::vector<unsigned char> rm(65536, 0);
  std::vector<unsigned char> gm(65536, 0);
  std::vector<unsigned char> bm(65536, 0);
  std::vector<unsigned char> am(65536, 0);
  for(unsigned y = 0; y < h; y++) {
    for(unsigned x = 0; x < w; x++) {
      unsigned short r = 256 * image[y * 8 * w + x * 8 + 0] + image[y * 8 * w + x * 8 + 1];
      unsigned short g = 256 * image[y * 8 * w + x * 8 + 2] + image[y * 8 * w + x * 8 + 3];
      unsigned short b = 256 * image[y * 8 * w + x * 8 + 4] + image[y * 8 * w + x * 8 + 5];
      unsigned short a = 256 * image[y * 8 * w + x * 8 + 6] + image[y * 8 * w + x * 8 + 7];
      RGBA rgba(std::make_pair(r, g), std::make_pair(b, a));
      //uint64_t rgba = (uint64_t)r + ((uint64_t)g << 16) + ((uint64_t)b << 32) + ((uint64_t)a << 48);
      rgbam[rgba]++;
      rm[r] = 1;
      gm[g] = 1;
      bm[b] = 1;
      am[a] = 1;
    }
  }
  *ro = *go = *bo = *ao = 0;
  for(size_t i = 0; i < rm.size(); i++) {
    *ro += rm[i];
    *go += gm[i];
    *bo += bm[i];
    *ao += am[i];
  }

  return rgbam.size();
}


void showError(Data& data, const Options& options) {
  std::cout << (options.use_hex ? std::hex: std::dec);
  std::string prefix = (options.use_hex ? "0x": "");
  if(!data.error) {
    std::cout << "No error" << std::endl;
  }
  std::cout << "Decoding error " << prefix << data.error << ": " << lodepng_error_text(data.error) << std::endl;
}

void loadWithErrorRecovery(Data& data, const Options& options, bool show_errors_mode) {
  (void)options;
  unsigned& error = data.error;
  lodepng::State& state = data.state;

  data.loadPixels();

  if(show_errors_mode) {
    if(!error) std::cout << "No errors or warnings" << std::endl;
    return;
  }

  // In case of checksum errors and some other ignorable errors, report it but ignore it and retry
  while(error) {
    // Not showing regular error here, is shown at end of program.
    unsigned error2 = error;
    if(error == 57) {
      showError(data, options);
      if(!show_errors_mode) std::cerr << "Ignoring the error: enabling ignore_crc" << std::endl;
      state.decoder.ignore_crc = 1;
      data.reloadPixels();
    } else if(error == 58) {
      showError(data, options);
      if(!show_errors_mode) std::cerr << "Ignoring the error: enabling ignore_adler32" << std::endl;
      state.decoder.zlibsettings.ignore_adler32 = 1;
      data.reloadPixels();
    } else if(error == 69) {
      showError(data, options);
      if(!show_errors_mode) std::cerr << "Ignoring the error: enabling ignore_critical" << std::endl;
      state.decoder.ignore_critical = 1;
      data.reloadPixels();
    } else if(error == 30 || error == 63) {
      showError(data, options);
      if(!show_errors_mode) std::cerr << "Ignoring the error: enabling ignore_end" << std::endl;
      state.decoder.ignore_end = 1;
      data.reloadPixels();
    } else {
      showError(data, options);
      if(!show_errors_mode) std::cerr << "This error is unrecoverable" << std::endl;
      break;  // other error that we cannot ignore
    }
    if(!show_errors_mode) if(error == 0) std::cerr << "Successfully ignored the error" << std::endl;
    if(error == error2) {
      if(!show_errors_mode) std::cerr << "Failed to ignore the error" << std::endl;
      break; // avoid infinite loop if ignoring did not fix the error code
    }
  }

  if(show_errors_mode) {
    if(!error) std::cout << "The error is recoverable" << std::endl;
    else std::cout << "The error is not recoverable" << std::endl;
  }
}

void showSingleLineSummary(Data& data, const Options& options) {
  data.loadInspect();
  if(data.error && data.error != 57) return; // CRC error (57) ignored here for parsing of header only
  std::cout << (options.use_hex ? std::hex: std::dec);
  std::cout << "Filesize: " << data.buffer.size() << " (" << data.buffer.size() / 1024 << "K)";
  if(data.is_icc) {
    std::cout << ", not a PNG but an ICC profile, use -i to expand ICC profile info." << std::endl;
    return;
  }
  if(data.is_exif) {
    std::cout << ", not a PNG but an EXIF file, use -e to expand EXIF file info." << std::endl;
    return;
  }

  std::cout << ", " << data.w << "x" << data.h << ", ";
  std::cout << "Color: " << colorTypeString(data.state.info_png.color.colortype) << ", " << data.state.info_png.color.bitdepth << " bit" << std::endl;
}

static unsigned getICCUint16(const unsigned char* icc, size_t size, size_t pos) {
  if (pos + 2 > size) return 0;
  return (unsigned)((icc[pos] << 8) | (icc[pos + 1]));
}

static unsigned getICCUint32(const unsigned char* icc, size_t size, size_t pos) {
  if (pos + 4 > size) return 0;
  return (unsigned)((icc[pos] << 24) | (icc[pos + 1] << 16) | (icc[pos + 2] << 8) | (icc[pos + 3] << 0));
}

static int getICCInt32(const unsigned char* icc, size_t size, size_t pos) {
  if (pos + 4 > size) return 0;
  return (int)((icc[pos] << 24) | (icc[pos + 1] << 16) | (icc[pos + 2] << 8) | (icc[pos + 3] << 0));
}

// Signed
static float getICC15Fixed16(const unsigned char* icc, size_t size, size_t pos) {
  return getICCInt32(icc, size, pos) / 65536.0;
}

// Unsigned
static float getICC16Fixed16(const unsigned char* icc, size_t size, size_t pos) {
  return getICCUint32(icc, size, pos) / 65536.0;
}

static std::string printableICCWord(const unsigned char* icc, size_t size, size_t pos) {
  if (pos + 4 > size) {
    return "out of range";
  }
  std::string result;
  for (int i = 0; i < 4; i++) {
    char c = icc[pos + i];
    result += ((c >= 32 && c < 127) ? c : '?');
  }
  return result;
}

void printICCDetails(const unsigned char* icc, size_t size, const std::string& indent) {
  // 128 for header, 4 for num tags
  if(size < 132) {
    std::cout << indent << "Invalid ICC: too small to contain header" << std::endl;
    return;
  }
  if(printableICCWord(icc, size, 36) != "acsp") {
    std::cout << indent << "Invalid ICC: does not contain signature \"acsp\"" << std::endl;
    return;
  }

  std::cout << indent << "profile size: " << getICCUint32(icc, size, 0) << std::endl;
  std::cout << indent << "CMM type: " << printableICCWord(icc, size, 4) << std::endl;

  uint32_t version = getICCUint32(icc, size, 8);
  uint16_t version_major = (version >> 24) & 255;
  uint16_t version_minor = (version >> 20) & 15;
  uint16_t version_bugfix = (version >> 16) & 15;
  std::cout << indent << "version: " << version_major << "." << version_minor
            << "." << version_bugfix << std::endl;

  std::cout << indent << "device class: " << printableICCWord(icc, size, 12) << std::endl;
  std::cout << indent << "input space: \"" << printableICCWord(icc, size, 16) << "\", ";
  std::cout << "output space: \"" << printableICCWord(icc, size, 20) << "\"" << std::endl;
  std::cout << indent;
  printf("date: %02d-%02d-%02dT%02d:%02d:%02d\n",
         getICCUint16(icc, size, 24), getICCUint16(icc, size, 26), getICCUint16(icc, size, 28),
         getICCUint16(icc, size, 30), getICCUint16(icc, size, 32), getICCUint16(icc, size, 34));
  std::cout << indent << "signature: " << printableICCWord(icc, size, 36) << std::endl;
  std::cout << indent << "platform: " << printableICCWord(icc, size, 40) << std::endl;
  std::cout << indent << "flags: " << getICCUint32(icc, size, 44) << std::endl;
  std::cout << indent << "device manufacturer: " << printableICCWord(icc, size, 48) << ", ";
  std::cout << "device model: " << printableICCWord(icc, size, 52) << ", ";
  std::cout << "device attributes: " << getICCUint32(icc, size, 56) << " "
            << getICCUint32(icc, size, 60) << std::endl;
  std::cout << indent << "rendering intent: " << getICCUint32(icc, size, 64) << std::endl;

  float pcsx = getICC15Fixed16(icc, size, 68);
  float pcsy = getICC15Fixed16(icc, size, 72);
  float pcsz = getICC15Fixed16(icc, size, 76);
  float pcsxyz = pcsx + pcsy + pcsz;
  std::cout << indent << "xyz illuminant: X:" << pcsx << ", Y:" << pcsy << ", Z:" << pcsz
            << ", xy:" << (pcsx / pcsxyz) << "," << (pcsy / pcsxyz) << std::endl;

  std::cout << indent << "creator: " << printableICCWord(icc, size, 80) << std::endl;

  // The md5 is present in v2.4 and above profiles, but it's always shown anyway. Those
  // bytes are normally all zero for older versions.
  std::cout << indent;
  printf("md5: %08x%08x%08x%08x\n", getICCUint32(icc, size, 84), getICCUint32(icc, size, 88),
                                    getICCUint32(icc, size, 92), getICCUint32(icc, size, 96));

  size_t numtags = getICCUint32(icc, size, 128);
  std::cout << indent << "num icc tags: " << numtags << std::endl;
  if(size < 128 + 4 + numtags * 12) {
    std::cout << indent << "Invalid ICC: too small to contain tag descriptions" << std::endl;
    return;
  }
  for(size_t i = 0; i < numtags; i++) {
    size_t pos = 132 + i * 12;
    std::cout << indent << "icc tag: \"" << printableICCWord(icc, size, pos) << "\"";
    size_t offset = getICCUint32(icc, size, pos + 4);
    size_t tagsize = getICCUint32(icc, size, pos + 8);
    std::cout << ", offset: " << offset << ", size: " << tagsize;
    if(offset + tagsize > size || tagsize < 4) {
      std::cout << std::endl << indent << "Invalid ICC: tag out of range" << std::endl;
      return;
    }
    std::string datatype = printableICCWord(icc, size, offset);
    std::cout << ", datatype: \"" << datatype << "\"";
    if(datatype == "XYZ ") {
      float x = getICC15Fixed16(icc, size, offset + 8);
      float y = getICC15Fixed16(icc, size, offset + 12);
      float z = getICC15Fixed16(icc, size, offset + 16);
      float xyz = x + y + z;
      std::cout << ", X:" << x << ", Y:" << y << ", Z:" << z;
      if(xyz) std::cout << ", xy:" << (x / xyz) << "," << (y / xyz);
    }
    if(datatype == "curv") {
      size_t lutsize = getICCUint32(icc, size, offset + 8);
      std::cout << ", lookup table size: " << lutsize;
      if(lutsize == 1 && offset + 14 <= size) {
        std::cout << " (gamma: " << (getICCUint16(icc, size, offset + 12) / 256.0) << ")";
      }
      if(lutsize == 0) std::cout << " (linear)";
    }
    if(datatype == "para") {
      unsigned type = getICCUint16(icc, size, offset + 8);
      float gamma = getICC15Fixed16(icc, size, offset + 12);
      int numparams = (type == 4) ? 7 : ((type >= 1 && type <= 3) ? (type + 1) : 0);
      std::cout << " type: " << type << ", gamma: " <<  gamma;
      if(numparams > 0) {
        std::cout << ", params: ";
        for(int j = 0; j < numparams; j++) {
          if(j > 0) std::cout << ", ";
          std::cout << getICC15Fixed16(icc, size, offset + 16 + j * 4);
        }
      }
    }
    if(datatype == "sf32") {
      std::cout << ":";
      for(size_t j = 8; j < tagsize; j += 4) {
        float v = getICC15Fixed16(icc, size, offset + j);
        std::cout << " " << v;
      }
    }
    if(datatype == "chrm") {
      size_t numchannels = getICCUint16(icc, size, offset + 8);
      std::cout << ": n:" << numchannels
                << " phosphor:" << getICCUint16(icc, size, offset + 10);
      for(size_t j = 0; j < numchannels; j++) {
        std::cout << " xy:" << getICC16Fixed16(icc, size, offset + 12 + j * 8)
                  << "," << getICC16Fixed16(icc, size, offset + 12 + j * 8 + 4);
      }
    }
    if(datatype == "text" || datatype == "mluc" || datatype == "desc") {
      // TODO: this is a bit of a simplification of the parse for now, e.g.
      // ignoring UTF-16, instead implicitely skipping non-ASCII bytes, and
      // potentially printing things multiple times in a row if multiple
      // variants are in desc or mluc.
      std::cout << ": ";
      for(size_t j = (datatype == "mluc" ? 28 : 8); j < tagsize; j++) {
        char c = icc[offset + j];
        if(c >= 32 && c < 127) std::cout << c;
      }
    }
    std::cout << std::endl;
  }
}

void showHex(const unsigned char* data, size_t size, const Options& options) {
  for(size_t i = 0; i < size; i++) {
    unsigned char c = data[i];
    if(options.hexformat == HF_BIN) {
      printf("%c", c);
    } else {
      if(c > 32 && c < 127 && options.hexformat == HF_MIX) printf(" %c ", c);
      else printf("%02x ", c);
      if(i % 32 == 31 && i + 1 != size) std::cout << std::endl;
    }
  }
  if(options.hexformat != HF_BIN) std::cout << std::endl;
}

void showHeaderInfo(Data& data, const Options& options) {
  data.loadInspect();
  if(data.error) return;
  std::cout << (options.use_hex ? std::hex: std::dec);

  const LodePNGInfo& info = data.state.info_png;
  const LodePNGColorMode& color = info.color;
  if(options.show_header) {
    std::cout << "Filesize: " << data.buffer.size() << " (" << data.buffer.size() / 1024 << "K)" << std::endl;
    if(data.is_icc) {
      std::cout << "Not a PNG but an ICC profile, use -i or -I for more info." << std::endl;
    } else if(data.is_exif) {
      std::cout << "Not a PNG but an EXIF file, use -e for more info." << std::endl;
    } else {
      std::cout << "Width: " << data.w << std::endl;
      std::cout << "Height: " << data.h << std::endl;
      if(options.verbose) {
        double bpp = data.buffer.size() / (double)(data.w * data.h);
        std::cout << "Compressed bpp: " << bpp << std::endl;
      }
      std::cout << "Interlace method: " << info.interlace_method << std::endl;
      if(options.verbose) {
        std::cout << "Compression method: " << info.compression_method << std::endl;
        std::cout << "Filter method: " << info.filter_method << std::endl;
      }
      std::cout << "Color type: " << colorTypeString(color.colortype) << std::endl;
      std::cout << "Bit depth: " << color.bitdepth << std::endl;
      if(options.verbose) {
        std::cout << "Bits per pixel: " << lodepng_get_bpp(&color) << std::endl;
        std::cout << "Channels per pixel: " << lodepng_get_channels(&color) << std::endl;
        std::cout << "Is greyscale type: " << lodepng_is_greyscale_type(&color) << std::endl;
        std::cout << "Can have alpha: " << lodepng_can_have_alpha(&color) << std::endl;
        std::cout << "Has color key: " << color.key_defined << std::endl;
      }
    }
  }
  if(options.show_png_info && data.is_png) {
    if (color.colortype == LCT_PALETTE) {
      std::cout << "Palette size: " << color.palettesize << std::endl;
    }
    if(color.key_defined) {
      std::cout << "Color key rgb: " << color.key_r
                << ", " << color.key_g
                << ", " << color.key_b << std::endl;
    }
    if(info.background_defined) {
      if(color.colortype == LCT_PALETTE) {
        std::cout << "Background index: " << info.background_r << std::endl;
      } else {
        std::cout << "Background rgb: " << info.background_r
                  << ", " << info.background_g
                  << ", " << info.background_b << std::endl;
      }
    }
    if(info.gama_defined) {
      std::cout << "gAMA defined: " << info.gama_gamma << " (" << (info.gama_gamma / 100000.0)
                << ", " << (100000.0 / info.gama_gamma) << ")" << std::endl;
    }
    if(info.chrm_defined) {
      std::cout << "cHRM defined: w: " << (info.chrm_white_x / 100000.0) << " " << (info.chrm_white_y / 100000.0)
                << ", r: " << (info.chrm_red_x / 100000.0) << " " << (info.chrm_red_y / 100000.0)
                << ", g: " << (info.chrm_green_x / 100000.0) << " " << (info.chrm_green_y / 100000.0)
                << ", b: " << (info.chrm_blue_x / 100000.0) << " " << (info.chrm_blue_y / 100000.0)
                << std::endl;
    }
    if(info.srgb_defined) {
      std::cout << "sRGB defined: rendering intent: " << info.srgb_intent << std::endl;
    }
    if(info.iccp_defined) {
      std::cout << "ICC profile defined (" << info.iccp_profile_size << " bytes): name: " << info.iccp_name << std::endl;
      if(options.verbose && !options.show_icc_details && !options.show_icc_hex) {
        std::cout << "Use -i or -I to show ICC profile details or hex" << std::endl;
      }
    }
    if(info.exif_defined) {
      std::cout << "EXIF metadata defined (" << info.exif_size << " bytes)" << std::endl;
      if(options.verbose && !options.show_exif && !options.show_exif_hex) {
        std::cout << "Use -e or -E to show EXIF details or hex" << std::endl;
      }
    }
  }
  if(info.iccp_defined && options.show_icc_details) {
    if(!options.show_header) {
      std::cout << "ICC profile details (" << info.iccp_profile_size << " bytes), PNG name: " << info.iccp_name << std::endl;
    }
    printICCDetails(info.iccp_profile, info.iccp_profile_size, "  ");
    std::cout << "end of ICC profile" << std::endl;
  }
  if(info.iccp_defined && options.show_icc_hex) {
    showHex(info.iccp_profile, info.iccp_profile_size, options);
  }
  if(info.exif_defined && options.show_exif_hex) {
    showHex(info.exif, info.exif_size, options);
  }

  if(options.show_png_info && data.is_png) {
    if(options.verbose) std::cout << "Physics defined: " << info.phys_defined << std::endl;
    if(info.phys_defined) {
      std::cout << "Physics: X: " << info.phys_x << ", Y: " << info.phys_y << ", unit: " << info.phys_unit << std::endl;
    }
  }
}

// shortens the text unless options.expand_long_texts is true
std::string shortenText(const std::string& text, const Options& options) {
  if(options.expand_long_texts) return text;
  size_t maxlen = 512;
  size_t maxnl = options.verbose ? 5 : 1;
  size_t numnl = 0; // amount of newlines
  for(size_t i = 0; i < text.size(); i++) {
    if(text[i] == 10) numnl++;
    if(numnl >= maxnl) {
      maxlen = i;
      break;
    }
  }

  if(text.size() < maxlen) return text;

  return text.substr(0, maxlen) + (numnl > 1 ? "\n" : "") + "... [TEXT SNIPPED! use -t to expand long text]";
}

// A bit more PNG info, which is from chunks that can come after IDAT. showHeaderInfo shows most other stuff.
void showPNGInfo(Data& data, const Options& options) {
  data.loadInspect();
  if(!data.is_png) return;
  loadWithErrorRecovery(data, options, false);
  if(data.error) return;
  std::cout << (options.use_hex ? std::hex: std::dec);

  const LodePNGInfo& info = data.state.info_png;

  if(options.verbose) std::cout << "Texts: " << info.text_num << std::endl;
  for(size_t i = 0; i < info.text_num; i++) {
    std::cout << "Text (" << (strlen(info.text_strings[i])) << " bytes): "
              << info.text_keys[i] << ": " << shortenText(info.text_strings[i], options) << std::endl;
  }
  if(options.verbose) std::cout << "International texts: " << info.itext_num << std::endl;
  for(size_t i = 0; i < info.itext_num; i++) {
    std::cout << "Text (" << (strlen(info.itext_strings[i])) << " bytes): "
              << info.itext_keys[i] << ", "
              << info.itext_langtags[i] << ", "
              << info.itext_transkeys[i] << ": "
              << shortenText(info.itext_strings[i], options) << std::endl;
  }
  if(options.verbose) std::cout << "Time defined: " << info.time_defined << std::endl;
  if(info.time_defined) {
    const LodePNGTime& time = info.time;
    printf("time: %02d-%02d-%02dT%02d:%02d:%02d\n",
           time.year, time.month, time.day, time.hour, time.minute, time.second);

  }
}

void showColorStats(Data& data, const Options& options) {
  data.loadInspect();
  if(!data.is_png) return;
  std::cout << (options.use_hex ? std::hex: std::dec);
  std::vector<unsigned char>& image = data.pixels;
  unsigned& w = data.w;
  unsigned& h = data.h;

  data.loadPixels();
  if(data.error) return;
  // TODO: move to show color stats function
  if(options.verbose) std::cout << "Num pixels: " << w * h << std::endl;
  size_t rc, gc, bc, ac;
  std::cout << "Num unique colors: " << countColors(image, w, h, &rc, &gc, &bc, &ac);
  std::cout << " (r: " << rc << ", g: " << gc << ", b: " << bc << ", a: " << ac << ")";
  std::cout << std::endl;
  if(w > 0 && h > 0) {
    double avg[4] = {0, 0, 0, 0};
    double min[4] = {999999, 999999, 999999, 999999};
    double max[4] = {0, 0, 0, 0};
    for(unsigned y = 0; y < h; y++) {
      for(unsigned x = 0; x < w; x++) {
        for(int c = 0; c < 4; c++) {
          double v = 256 * image[y * 8 * w + x * 8 + c * 2] + image[y * 8 * w + x * 8 + c * 2 + 1];
          avg[c] += v;
          min[c] = std::min(min[c], v);
          max[c] = std::max(max[c], v);
        }
      }
    }
    for(int c = 0; c < 4; c++) {
      avg[c] /= (w * h * 257.0);
      min[c] /= 257.0;
      max[c] /= 257.0;
    }
    if(options.verbose) std::cout << "Ranges shown as 0.0-255.0, even for 16-bit data:" << std::endl;
    std::cout << "Average color: " << avg[0] << ", " << avg[1] << ", " << avg[2] << ", " << avg[3] << std::endl;
    std::cout << "Color ranges: " << min[0] << "-" << max[0] << ", " << min[1] << "-" << max[1] << ", " << min[2] << "-" << max[2] << ", " << min[3] << "-" << max[3] << std::endl;
  }
}

void showErrors(Data& data, const Options& options) {
  std::cout << "Error report: " << std::endl;
  Data data2(data.filename);
  loadWithErrorRecovery(data2, options, true);
}

uint32_t readExifUint32(const unsigned char* exif, size_t size, size_t pos, bool big_endian) {
  if(pos + 4 > size) return 0;
  if(big_endian) {
    return ((uint32_t)exif[pos + 0] << 24u) | ((uint32_t)exif[pos + 1] << 16u) | ((uint32_t)exif[pos + 2] << 8u) | (uint32_t)exif[pos + 3];
  } else {
    return ((uint32_t)exif[pos + 3] << 24u) | ((uint32_t)exif[pos + 2] << 16u) | ((uint32_t)exif[pos + 1] << 8u) | (uint32_t)exif[pos + 0];
  }
}

uint32_t readExifUint16(const unsigned char* exif, size_t size, size_t pos, bool big_endian) {
  if(pos + 2 > size) return 0;
  if(big_endian) {
    return ((uint32_t)exif[pos + 0] << 8u) | (uint32_t)exif[pos + 1];
  } else {
    return ((uint32_t)exif[pos + 1] << 8u) | (uint32_t)exif[pos + 0];
  }
}


// shows all the information from 1 IFD from the exif file. If more IFDs are linked, recursively shows those too.
void showExifIFD(const unsigned char* exif, size_t size, size_t ifd_pos, bool big_endian, bool is_thumbnail, bool is_sub) {
  size_t pos = ifd_pos;
  size_t sub_ifd = 0;
  if(pos + 2 > size) {
    std::cout << "EXIF IFD out of range: " << pos << std::endl;
    return;
  }
  size_t num_entries = readExifUint16(exif, size, pos, big_endian);
  if(is_sub) {
    std::cout << "EXIF Sub-IFD at " << pos << ", num entries: " << num_entries << std::endl;
  } else if(is_thumbnail) {
    std::cout << "EXIF Thumbnail IFD at " << pos << ", num entries: " << num_entries << std::endl;
  } else {
    std::cout << "EXIF IFD at " << pos << ", num entries: " << num_entries << std::endl;
  }
  pos += 2;
  for(size_t i = 0; i < num_entries; i++) {
    if(pos + 12 > size) {
      std::cout << "EXIF IFD entry out of range: " << pos << std::endl;
      return;
    }
    uint32_t tag_number = readExifUint16(exif, size, pos, big_endian);
    uint32_t format = readExifUint16(exif, size, pos + 2, big_endian);
    uint32_t num = readExifUint32(exif, size, pos + 4, big_endian);
    uint32_t offset = readExifUint32(exif, size, pos + 8, big_endian);
    uint32_t component_size = 1;
    if(format == 3 || format == 8) component_size = 2;
    else if(format == 4 || format == 9 || format == 11) component_size = 4;
    else if(format == 5 || format == 10 || format == 12) component_size = 8;
    size_t len = num * component_size;
    if(len <= 4) offset = pos + 8; // small value is stored in the offset itself
    pos += 12;

    if(format < 1 || format > 12) {
      std::cout << "EXIF unknown entry format" << std::endl;
      return;
    }
    if(is_thumbnail) std::cout << "EXIF tag (thumbnail): ";
    else std::cout << "EXIF tag: ";
    // Only show some common tags by full name
    if(tag_number == 256) std::cout << "Umage Width";
    else if(tag_number == 257) std::cout << "Image Height";
    else if(tag_number == 259) std::cout << "Compression";
    else if(tag_number == 269) std::cout << "Document Name";
    else if(tag_number == 270) std::cout << "Image Description";
    else if(tag_number == 274) std::cout << "Orientation";
    else if(tag_number == 282) std::cout << "X Resolution";
    else if(tag_number == 283) std::cout << "Y Resolution";
    else if(tag_number == 296) std::cout << "Resolution Unit";
    else if(tag_number == 513) std::cout << "Thumbnail Offset";
    else if(tag_number == 514) std::cout << "Thumbnail Size";
    else if(tag_number == 33434) std::cout << "Exposure Time";
    else if(tag_number == 33432) std::cout << "Copyright";
    else if(tag_number == 36864) std::cout << "Exif Version";
    else if(tag_number == 37510) std::cout << "User Comment";
    else std::cout << "#" << tag_number; // tag for which we don't show a name here
    std::cout << ": ";

    if(offset + len > size) {
      std::cout << "EXIF data out of range" << std::endl;
      return;
    }
    if(len == 0) {
      std::cout << "[empty]" << std::endl;
      continue;
    }

    if(format == 1) {
      std::cout << (uint32_t)exif[offset];
    } else if(format == 2) {
      for(size_t j = 0; j < len; j++) {
        if(!exif[offset + j]) break; // NULL terminator
        std::cout << exif[offset + j];
      }
    } else if(format == 3) {
      std::cout << readExifUint16(exif, size, offset, big_endian);
    } else if(format == 4) {
      if(tag_number == 34665) sub_ifd = readExifUint32(exif, size, offset, big_endian);
      else std::cout << readExifUint32(exif, size, offset, big_endian);
    } else if(format == 5) {
      double n = readExifUint32(exif, size, offset, big_endian);
      double d = readExifUint32(exif, size, offset + 4, big_endian);
      std::cout << (n / d);
    } else if(format == 7 && len > 8 && exif[offset + 0] == 'A' && exif[offset + 1] == 'S' &&
              exif[offset + 2] == 'C' && exif[offset + 3] == 'I' && exif[offset + 4] == 'I') {
      for(size_t j = 8; j < len; j++) std::cout << exif[offset + j];
    } else if(format == 7 && len == 4 && tag_number == 36864) {
      for(size_t j = 0; j < len; j++) std::cout << exif[offset + j];
    } else if(format == 8) {
      std::cout << (int32_t)readExifUint16(exif, size, offset, big_endian);
    } else if(format == 9) {
      std::cout << (int16_t)readExifUint32(exif, size, offset, big_endian);
    } else if(format == 10) {
      double n = (int32_t)readExifUint32(exif, size, offset, big_endian);
      double d = (int32_t)readExifUint32(exif, size, offset + 4, big_endian);
      std::cout << (n / d);
    } else {
      // Formats like double not handled here
      std::cout << "[format " << format << ", len " << len << "]";
    }
    std::cout << std::endl;
  }
  if(pos + 4 > size) {
    std::cout << "EXIF IFD footer out of range" << std::endl;
    return;
  }
  size_t next_ifd = readExifUint32(exif, size, pos, big_endian);

  // The > checks are to guarantee progress rather than infinity recursion, though it does mean
  // that an odd EXIF that places later parts earlier in the file won't be supported correctly
  if(sub_ifd && sub_ifd > ifd_pos && sub_ifd != next_ifd) {
    showExifIFD(exif, size, sub_ifd, big_endian, is_thumbnail, true);
  }
  if(next_ifd && next_ifd > ifd_pos) {
    showExifIFD(exif, size, sub_ifd, big_endian, true, is_sub);
  }
}

// Shows information from the EXIF chunk in the PNG, this only shows the basics
// and some primitive values of the EXIF, it's not a complete EXIF parser but
// shows the most common tags by name to verify the EXIF chunk handling is working.
void showExif(Data& data) {
  data.loadInspect();
  if(data.error) return;
  if(!data.state.info_png.exif_defined) {
    std::cout << "No EXIF data present in this PNG image" << std::endl;
    return;
  }
  const unsigned char* exif = data.state.info_png.exif;
  size_t size = data.state.info_png.exif_size;
  if(size < 8) {
    std::cout << "EXIF size too small: " << size << std::endl;
    return;
  }
  std::cout << "EXIF size: " << size << std::endl;
  bool big_endian = exif[0] == 'M' && exif[1] == 'M' && exif[2] == 0 && exif[3] == 42;
  bool little_endian = exif[0] == 'I' && exif[1] == 'I' && exif[2] == 42 && exif[3] == 0;
  if(big_endian) {
    std::cout << "EXIF encoded using big endian (MM)" << std::endl;
  } else if(little_endian) {
    std::cout << "EXIF encoded using little endian (II)" << std::endl;
  } else {
    std::cout << "EXIF has invalid header, must start with MM or II" << std::endl;
    return;
  }
  size_t ifd = readExifUint32(exif, size, 4, big_endian);
  showExifIFD(exif, size, ifd, big_endian, false, false);
}

void showRender(Data& data, const Options& options) {
  data.loadPixels();
  if(data.error) return;
  if(options.rendermode == RM_ASCII) {
    displayAsciiArt(data.pixels, data.w, data.h, options.rendersize);
  }

  if(options.rendermode == RM_HEX) {
    displayColorsHex(data.pixels, data.w, data.h, false);
  }

  if(options.rendermode == RM_HEX16) {
    displayColorsHex(data.pixels, data.w, data.h, true);
  }

  if(options.rendermode == RM_PAL) {
    displayPalettePixels(data.buffer, options);
  }
}


void showInfos(Data& data, const Options& options) {
  if(options.show_one_line_summary) showSingleLineSummary(data, options);
  if(options.show_errors) showErrors(data, options);
  if(options.show_exif) showExif(data);
  if(options.show_header || options.show_icc_details || options.show_icc_hex || options.show_exif_hex) showHeaderInfo(data, options);
  if(options.show_color_stats) showColorStats(data, options);
  if(options.show_png_info) showPNGInfo(data, options);
  if(options.show_palette) displayPalette(data, options);
  if(options.show_chunks || options.show_chunks2) displayChunkNames(data, options);
  if(options.show_filters) displayFilterTypes(data, options);
  if(options.show_zlib_info) printZlibInfo(data);
  if(options.show_render) showRender(data, options);

  if(data.error) showError(data, options);
}

int main(int argc, char *argv[]) {
  Options options;
  bool options_chosen = false;

  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if(s.size() > 1 && s[0] == '-' && s[1] != '-') {
      // anything that chooses actual set disables the defaults
      if(s != "-x" && s != "-v" && s != "-t") options_chosen = true;
      for(size_t j = 1; j < s.size(); j++) {
        char c = s[j];
        if(c == 'o') options.show_one_line_summary = true;
        else if(c == 'H') options.show_header = true;
        else if(c == 'i') options.show_icc_details = true;
        else if(c == 'I') options.show_icc_hex = true;
        else if(c == 'e') options.show_exif = true;
        else if(c == 'E') options.show_exif_hex = true;
        else if(c == 'v') options.verbose = true;
        else if(c == 't') options.expand_long_texts = true;
        else if(c == 's') options.show_color_stats = true;
        else if(c == 'a') options.show_errors = true;
        else if(c == 'p') options.show_header = options.show_png_info = true;
        else if(c == 'r') options.show_render = true;
        else if(c == 'l') options.show_palette = true;
        else if(c == 'L') options.show_palette_pixels = true;
        else if(c == 'c') options.show_chunks = true;
        else if(c == 'C') options.show_chunks2 = true;
        else if(c == 'f') options.show_filters = true;
        else if(c == 'z') options.show_zlib_info = true;
        else if(c == 'x') {
          options.use_hex = true;
          std::cout << std::hex;
        } else {
          if(s != "--help" && c != 'h' && c != '?') std::cout << "Unknown flag: " << c << ". Use -h for help" << std::endl;
          showHelp();
          return 0;
        }

      }
    } else if(s.size() > 1 && s[0] == '-' && s[1] == '-') {
      size_t eqpos = 2;
      while(eqpos < s.size() && s[eqpos] != '=') eqpos++;
      std::string key = s.substr(2, eqpos - 2);
      std::string value = (eqpos + 1) < s.size() ? s.substr(eqpos + 1) : "";
      if(key == "help") {
        showHelp();
        return 0;
      }
      if(key == "mode") {
        if(value == "ascii") options.rendermode = RM_ASCII;
        else if(value == "hex") options.rendermode = RM_HEX;
        else if(value == "hex16") options.rendermode = RM_HEX16;
        else if(value == "palette") options.rendermode = RM_PAL;
      }
      if(key == "size") {
        int size = strtoval<int>(value);
        if(size >= 1 && size <= 4096) options.rendersize = size;
      }
      if(key == "format") {
        if(value == "mix") options.hexformat = HF_MIX;
        else if(value == "hex") options.hexformat = HF_HEX;
        else if(value == "bin") options.hexformat = HF_BIN;
      }
    }
    else filenames.push_back(s);
  }

  if(filenames.empty()) {
    std::cout << "Please provide a filename to preview" << std::endl;
    showHelp();
    return 0;
  }

  if(!options_chosen) {
    //fill in defaults
    options.show_header = true;
    options.show_png_info = true;
    options.show_chunks2 = true;
    // verbose lets individual sections show more, and in addition adds more default unlocked sections if no specific one chosen
    if(options.verbose) {
      options.show_chunks2 = false;
      options.show_chunks = true;
    }
  }

  for(size_t i = 0; i < filenames.size(); i++) {
    if(filenames.size() > 1) {
      if(i > 0 && !options.show_one_line_summary) std::cout << std::endl;
      std::cout << filenames[i] << ":";
      if(!options.show_one_line_summary) std::cout << std::endl; else std::cout << " ";
    }
    Data data(filenames[i]);
    showInfos(data, options);
  }
}

