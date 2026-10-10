/*
 *    MIT License
 *
 *    Copyright (c) 2016 Błażej Szczygieł
 * 	  Copyright (c) 2026-present The Simplicity Group & contributors
 * 	  See AUTHORS.md for more information
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include "include/decoder/VPXDecoder.h"

#include <SDL3/SDL.h>

#include <webm/file_reader.h>

#include <algorithm>
#include <cstdio>

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		std::fprintf(stderr, "Usage: %s <file.webm>\n", argv[0]);

		return 1;
	}

	FILE *file = std::fopen(argv[1], "rb");

	if (!file)
	{
		std::fprintf(stderr, "Failed to open file: %s\n", argv[1]);

		return 1;
	}

	webm::FileReader reader(file);
	WebMDemuxer demuxer(&reader);

	if (!demuxer.isOpen())
	{
		std::fprintf(stderr, "Failed to open WebM file: %s\n", argv[1]);

		return 1;
	}

	if (demuxer.getVideoCodec() == WebMDemuxer::NO_VIDEO)
	{
		std::fprintf(stderr, "File contains no supported video track\n");

		return 1;
	}

	VPXDecoder videoDecoder(demuxer, 1);

	if (!videoDecoder.isOpen())
	{
		std::fprintf(stderr, "Failed to open video decoder\n");

		return 1;
	}

	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());

		return 1;
	}

	const int width = demuxer.getWidth();
	const int height = demuxer.getHeight();

	SDL_Window *window = SDL_CreateWindow("libsimplewebm player", width, height, 0);

	if (!window)
	{
		std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());

		SDL_Quit();
		return 1;
	}

	SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);

	if (!renderer)
	{
		std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());

		SDL_DestroyWindow(window);
		SDL_Quit();

		return 1;
	}

	SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING, width, height);

	if (!texture)
	{
		std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());

		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();

		return 1;
	}

	WebMFrame videoFrame;
	VPXDecoder::Image image;

	bool running = true;
	bool haveFrame = false;

	double playbackStart = static_cast<double>(SDL_GetTicks()) / 1000.0;

	double frameTime = 0.0;

	while (running)
	{
		SDL_Event event;

		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_EVENT_QUIT)
			{
				running = false;
			}
		}

		if (!running)
		{
			break;
		}

		if (!haveFrame)
		{
			if (!demuxer.readFrame(&videoFrame, nullptr))
			{
				running = false;
				break;
			}

			if (!videoFrame.isValid())
			{
				continue;
			}

			if (!videoDecoder.decode(videoFrame))
			{
				std::fprintf(stderr, "Video decode error at %.3f s\n", videoFrame.time);

				running = false;
				break;
			}

			VPXDecoder::IMAGE_ERROR imageError = videoDecoder.getImage(image);

			if (imageError == VPXDecoder::NO_FRAME)
			{
				continue;
			}

			if (imageError != VPXDecoder::NO_ERROR)
			{
				std::fprintf(stderr, "Unsupported decoded image at %.3f s\n", videoFrame.time);

				running = false;
				break;
			}

			frameTime = videoFrame.time;
			haveFrame = true;

			//std::fprintf(stderr, "Frame: %.3f s, %dx%d, alpha=%s\n", frameTime, image.w, image.h, image.alpha ? "yes" : "no");

            //std::fprintf(stderr, "Y=%p U=%p V=%p A=%p | strides=%d/%d/%d/%d\n", static_cast<void *>(image.planes[0]), static_cast<void *>(image.planes[1]), static_cast<void *>(image.planes[2]), static_cast<void *>(image.alpha), image.linesize[0], image.linesize[1], image.linesize[2], image.alphaLinesize);

            if (image.alpha)
            {
                unsigned minAlpha = 255;
                unsigned maxAlpha = 0;
                unsigned long long sumAlpha = 0;

                for (int y = 0; y < image.h; y++)
                {
                    const unsigned char *row = image.alpha + y * image.alphaLinesize;

                    for (int x = 0; x < image.w; x++)
                    {
                        const unsigned value = row[x];

                        minAlpha = std::min(minAlpha, value);
                        maxAlpha = std::max(maxAlpha, value);
                        sumAlpha += value;
                    }
                }

                const double average = static_cast<double>(sumAlpha) / static_cast<double>(image.w * image.h);

                //std::fprintf(stderr, "Alpha range: %u-%u, average=%.2f\n", minAlpha, maxAlpha, average);
            }
            if (!SDL_UpdateYUVTexture(texture, NULL, image.planes[0], image.linesize[0], image.planes[1], image.linesize[1], image.planes[2], image.linesize[2]))
			{
				std::fprintf(stderr, "SDL_UpdateYUVTexture failed: %s\n", SDL_GetError());

				running = false;
				break;
			}
		}

		if (!running)
		{
			break;
		}

		const double currentTime = static_cast<double>(SDL_GetTicks()) / 1000.0;

		const double playbackTime = currentTime - playbackStart;

		
		if (frameTime > playbackTime)
		{
			SDL_Delay(1);
			continue;
		}

		SDL_RenderClear(renderer);

		SDL_RenderTexture(renderer, texture, NULL, NULL);

		SDL_RenderPresent(renderer);

		haveFrame = false;
	}

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);

	SDL_Quit();

	return 0;
}