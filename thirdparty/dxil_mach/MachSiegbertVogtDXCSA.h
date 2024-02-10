// Licensed under the Mach engine license (Apache or MIT at your choosing.) For details
// and a copy of this license, see https://github.com/hexops/mach/blob/main/LICENSE
// This copyright header, and a copy of the above open source licenses, must be provided
// with any redistributions of this software.

#pragma once
#ifndef MACH_SIEGBERT_VOGT_DXCSA_H
#define MACH_SIEGBERT_VOGT_DXCSA_H

#include <stdint.h>

// Performs Mach Siegbert Vogt DX Code Signing Algorithm on a DXIL binary file / container blob.
//
// This is an open source alternative to dxil.dll's proprietary code signing.
//
// Writes the code signing secret to
// the `secret_out` out parameter.
void machSiegbertVogtDXCSA(uint8_t* p_data, uint32_t dw_size, uint32_t secret_out[4]);

#endif // MACH_SIEGBERT_VOGT_DXCSA_H
