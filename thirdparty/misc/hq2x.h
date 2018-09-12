#ifndef HQ2X_H
#define HQ2X_H

#include "core/typedefs.h"


uint32_t *hq2x_resize(
		const uint32_t *image,
		uint32_t width,
		uint32_t height,
		uint32_t *output,
		uint32_t trY = 0x30,
		uint32_t trU = 0x07,
		uint32_t trV = 0x06,
		uint32_t trA = 0x50,
		bool wrapX = false,
		bool wrapY = false );

#endif // HQ2X_H
