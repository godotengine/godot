/*
LodePNG Examples

Copyright (c) 2005-2012 Lode Vandevenne

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

//Compile command for Linux:
//g++ lodepng.cpp example_sdl.cpp -lSDL2 -O3 -o showpng

/*
LodePNG SDL example
This example displays a PNG with a checkerboard pattern to show tranparency.
It requires the SDL2 library to compile and run.
If multiple filenames are given to the command line, it shows all of them.
Press any key to see next image, or esc to quit.
*/

#include "lodepng.h"

#include <iostream>
#include <SDL2/SDL.h>

int show(const std::string& caption, const unsigned char* rgba, unsigned w, unsigned h) {
  //avoid too large window size by downscaling large image
  unsigned jump = 1;
  if(w / 1024 >= jump) jump = w / 1024 + 1;
  if(h / 1024 >= jump) jump = h / 1024 + 1;

  size_t screenw = w / jump;
  size_t screenh = h / jump;
  size_t pitch = screenw * sizeof(Uint32);
  //init SDL
  SDL_Window* sdl_window;
  SDL_Renderer* sdl_renderer;
  SDL_CreateWindowAndRenderer(screenw, screenh, SDL_WINDOW_OPENGL, &sdl_window, &sdl_renderer);
  SDL_SetWindowTitle(sdl_window, caption.c_str());
  if(!sdl_window) {
    std::cout << "error, no SDL screen" << std::endl;
    return 0;
  }
  SDL_Texture* sdl_texture = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_ARGB8888,
                                               SDL_TEXTUREACCESS_STREAMING, screenw, screenh);
  std::vector<Uint32> sdl_pixels(screenw * screenh * sizeof(Uint32));

  //plot the pixels of the PNG file
  for(unsigned y = 0; y + jump - 1 < h; y += jump)
  for(unsigned x = 0; x + jump - 1 < w; x += jump) {
    //get RGBA components
    Uint32 r = rgba[4 * y * w + 4 * x + 0]; //red
    Uint32 g = rgba[4 * y * w + 4 * x + 1]; //green
    Uint32 b = rgba[4 * y * w + 4 * x + 2]; //blue
    Uint32 a = rgba[4 * y * w + 4 * x + 3]; //alpha

    //make translucency visible by placing checkerboard pattern behind image
    int checkerColor = 191 + 64 * (((x / 16) % 2) == ((y / 16) % 2));
    r = (a * r + (255 - a) * checkerColor) / 255;
    g = (a * g + (255 - a) * checkerColor) / 255;
    b = (a * b + (255 - a) * checkerColor) / 255;

    //give the color value to the pixel of the screenbuffer
    sdl_pixels[(y * screenw + x) / jump] = 65536 * r + 256 * g + b;
  }

  // render the pixels to the screen
  SDL_UpdateTexture(sdl_texture, NULL, sdl_pixels.data(), pitch);
  SDL_RenderClear(sdl_renderer);
  SDL_RenderCopy(sdl_renderer, sdl_texture, NULL, NULL);
  SDL_RenderPresent(sdl_renderer);

  //pause until you press escape and meanwhile redraw screen
  SDL_Event event;
  int done = 0;
  while(done == 0) {
    while(SDL_PollEvent(&event)) {
      if(event.type == SDL_QUIT) done = 2;
      else if(SDL_GetKeyboardState(NULL)[SDLK_ESCAPE]) done = 2;
      else if(event.type == SDL_KEYDOWN) done = 1; //press any other key for next image
    }
    SDL_Delay(5); //pause 5 ms so it consumes less processing power
  }

  SDL_Quit();
  return done == 2 ? 1 : 0;
}

/*shows image with SDL. Returns 1 if user wants to fully quit, 0 if user wants to see next image.*/
int showfile(const char* filename) {
  std::cout << "showing " << filename << std::endl;

  std::vector<unsigned char> buffer, image;
  lodepng::load_file(buffer, filename); //load the image file with given filename
  unsigned w, h;
  unsigned error = lodepng::decode(image, w, h, buffer); //decode the png

  //stop if there is an error
  if(error) {
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 0;
  }

  return show(filename, &image[0], w, h);
}

int main(int argc, char* argv[]) {
  if(argc <= 1) std::cout << "Please enter PNG file name(s) to display" << std::endl;

  for(int i = 1; i < argc; i++) {
    if(showfile(argv[i])) return 0;
  }
}
