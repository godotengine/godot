/*
ExoQuant v0.7

Copyright (c) 2004 Dennis Ranke

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "exoquant.h"
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef NULL
#define NULL (0)
#endif

#define SCALE_R 1.0f
#define SCALE_G 1.2f
#define SCALE_B 0.8f
#define SCALE_A 1.0f

exq_data *exq_init()
{
	int i;
	exq_data *pExq;

	pExq = (exq_data*)malloc(sizeof(exq_data));
	
	for(i = 0; i < EXQ_HASH_SIZE; i++)
		pExq->pHash[i] = NULL;

	pExq->numColors = 0;
	pExq->optimized = 0;
	pExq->transparency = 1;
	pExq->numBitsPerChannel = 8;

	return pExq;
}

void exq_no_transparency(exq_data *pExq)
{
	pExq->transparency = 0;
}

void exq_free(exq_data *pExq)
{
	int i;
	exq_histogram *pCur, *pNext;

	for(i = 0; i < EXQ_HASH_SIZE; i++)
		for(pCur = pExq->pHash[i]; pCur != NULL; pCur = pNext)
		{
			pNext = pCur->pNextInHash;
			free(pCur);
		}

	free(pExq);
}

static unsigned int exq_make_hash(unsigned int rgba)
{
	rgba -= (rgba >> 13) | (rgba << 19);
	rgba -= (rgba >> 13) | (rgba << 19);
	rgba -= (rgba >> 13) | (rgba << 19);
	rgba -= (rgba >> 13) | (rgba << 19);
	rgba -= (rgba >> 13) | (rgba << 19);
	rgba &= EXQ_HASH_SIZE - 1;
	return rgba;
}

void exq_feed(exq_data *pExq, unsigned char *pData, int nPixels)
{
	int i;
	unsigned int hash;
	unsigned char r, g, b, a;
	exq_histogram *pCur;
	unsigned char channelMask = 0xff00 >> pExq->numBitsPerChannel;

	for(i = 0; i < nPixels; i++)
	{
		r = *pData++; g = *pData++; b = *pData++; a = *pData++;
		hash = exq_make_hash(((unsigned int)r) | (((unsigned int)g) << 8) | (((unsigned int)b) << 16) | (((unsigned int)a) << 24));

		pCur = pExq->pHash[hash];
		while(pCur != NULL && (pCur->ored != r || pCur->ogreen != g ||
			pCur->oblue != b || pCur->oalpha != a))
			pCur = pCur->pNextInHash;

		if(pCur != NULL)
			pCur->num++;
		else
		{
			pCur = (exq_histogram*)malloc(sizeof(exq_histogram));
			pCur->pNextInHash = pExq->pHash[hash];
			pExq->pHash[hash] = pCur;
			pCur->ored = r; pCur->ogreen = g; pCur->oblue = b; pCur->oalpha = a;
			r &= channelMask; g &= channelMask; b &= channelMask;
			pCur->color.r = r / 255.0f * SCALE_R;
			pCur->color.g = g / 255.0f * SCALE_G;
			pCur->color.b = b / 255.0f * SCALE_B;
			pCur->color.a = a / 255.0f * SCALE_A;

			if(pExq->transparency)
			{
				pCur->color.r *= pCur->color.a;
				pCur->color.g *= pCur->color.a;
				pCur->color.b *= pCur->color.a;
			}

			pCur->num = 1;
			pCur->palIndex = -1;
			pCur->ditherScale.r = pCur->ditherScale.g = pCur->ditherScale.b =
				pCur->ditherScale.a = -1;
			pCur->ditherIndex[0] = pCur->ditherIndex[1] = pCur->ditherIndex[2] =
				pCur->ditherIndex[3] = -1;
		}
	}
}

void exq_quantize(exq_data *pExq, int nColors)
{
	exq_quantize_ex(pExq, nColors, 0);
}

void exq_quantize_hq(exq_data *pExq, int nColors)
{
	exq_quantize_ex(pExq, nColors, 1);
}

void exq_quantize_ex(exq_data *pExq, int nColors, int hq)
{
	int besti;
	exq_float beste;
	exq_histogram *pCur, *pNext;
	int i, j;

	if(nColors > 256)
		nColors = 256;

	if(pExq->numColors == 0)
	{
		pExq->node[0].pHistogram = NULL;
		for(i = 0; i < EXQ_HASH_SIZE; i++)
			for(pCur = pExq->pHash[i]; pCur != NULL; pCur = pCur->pNextInHash)
			{
				pCur->pNext = pExq->node[0].pHistogram;
				pExq->node[0].pHistogram = pCur;
			}
		
		exq_sum_node(&pExq->node[0]);

		pExq->numColors = 1;
	}

	for(i = pExq->numColors; i < nColors; i++)
	{
		beste = 0;
		besti = 0;
		for(j = 0; j < i; j++)
			if(pExq->node[j].vdif >= beste)
			{
				beste = pExq->node[j].vdif;
				besti = j;
			}

//		printf("node %d: %d, %f\n", besti, pExq->node[besti].num, beste);

		pCur = pExq->node[besti].pHistogram;
		pExq->node[besti].pHistogram = NULL;
		pExq->node[i].pHistogram = NULL;
		while(pCur != NULL && pCur != pExq->node[besti].pSplit)
		{
			pNext = pCur->pNext;
			pCur->pNext = pExq->node[i].pHistogram;
			pExq->node[i].pHistogram = pCur;
			pCur = pNext;
		}

		while(pCur != NULL)
		{
			pNext = pCur->pNext;
			pCur->pNext = pExq->node[besti].pHistogram;
			pExq->node[besti].pHistogram = pCur;
			pCur = pNext;
		}

		exq_sum_node(&pExq->node[besti]);
		exq_sum_node(&pExq->node[i]);

		pExq->numColors = i + 1;
		if(hq)
			exq_optimize_palette(pExq, 1);
	}

	pExq->optimized = 0;
}

exq_float exq_get_mean_error(exq_data *pExq)
{
	int i, n;
	exq_float err;

	n = 0;
	err = 0;
	for(i = 0; i < pExq->numColors; i++)
	{
		n += pExq->node[i].num;
		err += pExq->node[i].err;
	}

	return sqrt(err / n) * 256;
}

void exq_get_palette(exq_data *pExq, unsigned char *pPal, int nColors)
{
	int i, j;
	exq_float r, g, b, a;
	unsigned char channelMask = 0xff00 >> pExq->numBitsPerChannel;

	if(nColors > pExq->numColors)
		nColors = pExq->numColors;

	if(!pExq->optimized)
		exq_optimize_palette(pExq, 4);

	for(i = 0; i < nColors; i++)
	{
		r = pExq->node[i].avg.r;
		g = pExq->node[i].avg.g;
		b = pExq->node[i].avg.b;
		a = pExq->node[i].avg.a;

		if(pExq->transparency == 1 && a != 0)
		{
			r /= a; g/= a; b/= a;
		}

		pPal[0] = (unsigned char)(r / SCALE_R * 255.9f);
		pPal[1] = (unsigned char)(g / SCALE_G * 255.9f);
		pPal[2] = (unsigned char)(b / SCALE_B * 255.9f);
		pPal[3] = (unsigned char)(a / SCALE_A * 255.9f);

		for(j = 0; j < 3; j++)
			pPal[j] = (pPal[j] + (1 << (8 - pExq->numBitsPerChannel)) / 2) & channelMask;
		pPal += 4;
	}
}

void exq_set_palette(exq_data *pExq, unsigned char *pPal, int nColors)
{
	int i;

	pExq->numColors = nColors;

	for(i = 0; i < nColors; i++)
	{
		pExq->node[i].avg.r = *pPal++ * SCALE_R / 255.9f;
		pExq->node[i].avg.g = *pPal++ * SCALE_G / 255.9f;
		pExq->node[i].avg.b = *pPal++ * SCALE_B / 255.9f;
		pExq->node[i].avg.a = *pPal++ * SCALE_A / 255.9f;
	}

	pExq->optimized = 1;
}

void exq_sum_node(exq_node *pNode)
{
	int n, n2;
	exq_color fsum, fsum2, vc, tmp, tmp2, sum, sum2;
	exq_histogram *pCur;
	exq_float isqrt, nv, v;

	n = 0;
	fsum.r = fsum.g = fsum.b = fsum.a = 0;
	fsum2.r = fsum2.g = fsum2.b = fsum2.a = 0;

	for(pCur = pNode->pHistogram; pCur != NULL; pCur = pCur->pNext)
	{
		n += pCur->num;
		fsum.r += pCur->color.r * pCur->num;
		fsum.g += pCur->color.g * pCur->num;
		fsum.b += pCur->color.b * pCur->num;
		fsum.a += pCur->color.a * pCur->num;
		fsum2.r += pCur->color.r * pCur->color.r * pCur->num;
		fsum2.g += pCur->color.g * pCur->color.g * pCur->num;
		fsum2.b += pCur->color.b * pCur->color.b * pCur->num;
		fsum2.a += pCur->color.a * pCur->color.a * pCur->num;
	}
	pNode->num = n;
	if(n == 0)
	{
		pNode->vdif = 0;
		pNode->err = 0;
		return;
	}

	pNode->avg.r = fsum.r / n;
	pNode->avg.g = fsum.g / n;
	pNode->avg.b = fsum.b / n;
	pNode->avg.a = fsum.a / n;

	vc.r = fsum2.r - fsum.r * pNode->avg.r;
	vc.g = fsum2.g - fsum.g * pNode->avg.g;
	vc.b = fsum2.b - fsum.b * pNode->avg.b;
	vc.a = fsum2.a - fsum.a * pNode->avg.a;

	v = vc.r + vc.g + vc.b + vc.a;
	pNode->err = v;
	pNode->vdif = -v;

	if(vc.r > vc.g && vc.r > vc.b && vc.r > vc.a)
		exq_sort(&pNode->pHistogram, exq_sort_by_r);
	else if(vc.g > vc.b && vc.g > vc.a)
		exq_sort(&pNode->pHistogram, exq_sort_by_g);
	else if(vc.b > vc.a)
		exq_sort(&pNode->pHistogram, exq_sort_by_b);
	else
		exq_sort(&pNode->pHistogram, exq_sort_by_a);

	pNode->dir.r = pNode->dir.g = pNode->dir.b = pNode->dir.a = 0;
	for(pCur = pNode->pHistogram; pCur != NULL; pCur = pCur->pNext)
	{
		tmp.r = (pCur->color.r - pNode->avg.r) * pCur->num;
		tmp.g = (pCur->color.g - pNode->avg.g) * pCur->num;
		tmp.b = (pCur->color.b - pNode->avg.b) * pCur->num;
		tmp.a = (pCur->color.a - pNode->avg.a) * pCur->num;
		if(tmp.r * pNode->dir.r + tmp.g * pNode->dir.g +
			tmp.b * pNode->dir.b + tmp.a * pNode->dir.a < 0)
		{
			tmp.r = -tmp.r;
			tmp.g = -tmp.g;
			tmp.b = -tmp.b;
			tmp.a = -tmp.a;
		}
		pNode->dir.r += tmp.r;
		pNode->dir.g += tmp.g;
		pNode->dir.b += tmp.b;
		pNode->dir.a += tmp.a;
	}
	isqrt = 1 / sqrt(pNode->dir.r * pNode->dir.r +
		pNode->dir.g * pNode->dir.g + pNode->dir.b * pNode->dir.b +
		pNode->dir.a * pNode->dir.a);
	pNode->dir.r *= isqrt;
	pNode->dir.g *= isqrt;
	pNode->dir.b *= isqrt;
	pNode->dir.a *= isqrt;

	exq_sort_dir = pNode->dir;
	exq_sort(&pNode->pHistogram, exq_sort_by_dir);

	sum.r = sum.g = sum.b = sum.a = 0;
	sum2.r = sum2.g = sum2.b = sum2.a = 0;
	n2 = 0;
	pNode->pSplit = pNode->pHistogram;
	for(pCur = pNode->pHistogram; pCur != NULL; pCur = pCur->pNext)
	{
		if(pNode->pSplit == NULL)
			pNode->pSplit = pCur;

		n2 += pCur->num;
		sum.r += pCur->color.r * pCur->num;
		sum.g += pCur->color.g * pCur->num;
		sum.b += pCur->color.b * pCur->num;
		sum.a += pCur->color.a * pCur->num;
		sum2.r += pCur->color.r * pCur->color.r * pCur->num;
		sum2.g += pCur->color.g * pCur->color.g * pCur->num;
		sum2.b += pCur->color.b * pCur->color.b * pCur->num;
		sum2.a += pCur->color.a * pCur->color.a * pCur->num;

		if(n == n2)
			break;

		tmp.r = sum2.r - sum.r*sum.r / n2;
		tmp.g = sum2.g - sum.g*sum.g / n2;
		tmp.b = sum2.b - sum.b*sum.b / n2;
		tmp.a = sum2.a - sum.a*sum.a / n2;
		tmp2.r = (fsum2.r - sum2.r) - (fsum.r-sum.r)*(fsum.r-sum.r) / (n - n2);
		tmp2.g = (fsum2.g - sum2.g) - (fsum.g-sum.g)*(fsum.g-sum.g) / (n - n2);
		tmp2.b = (fsum2.b - sum2.b) - (fsum.b-sum.b)*(fsum.b-sum.b) / (n - n2);
		tmp2.a = (fsum2.a - sum2.a) - (fsum.a-sum.a)*(fsum.a-sum.a) / (n - n2);

		nv = tmp.r + tmp.g + tmp.b + tmp.a + tmp2.r + tmp2.g + tmp2.b + tmp2.a;
		if(-nv > pNode->vdif)
		{
			pNode->vdif = -nv;
			pNode->pSplit = NULL;
		}
	}

	if(pNode->pSplit == pNode->pHistogram)
		pNode->pSplit = pNode->pSplit->pNext;

	pNode->vdif += v;
//	printf("error sum: %f, vdif: %f\n", pNode->err, pNode->vdif);
}

void exq_optimize_palette(exq_data *pExq, int iter)
{
	int n, i, j;
	exq_histogram *pCur;

	pExq->optimized = 1;

	for(n = 0; n < iter; n++)
	{
		for(i = 0; i < pExq->numColors; i++)
			pExq->node[i].pHistogram = NULL;

		for(i = 0; i < EXQ_HASH_SIZE; i++)
			for(pCur = pExq->pHash[i]; pCur != NULL; pCur = pCur->pNextInHash)
			{
				j = exq_find_nearest_color(pExq, &pCur->color);
				pCur->pNext = pExq->node[j].pHistogram;
				pExq->node[j].pHistogram = pCur;
			}

		for(i = 0; i < pExq->numColors; i++)
			exq_sum_node(&pExq->node[i]);
	}
}

void exq_map_image(exq_data *pExq, int nPixels, unsigned char *pIn,
				   unsigned char *pOut)
{
	int i;
	exq_color c;
	exq_histogram *pHist;

	if(!pExq->optimized)
		exq_optimize_palette(pExq, 4);

	for(i = 0; i < nPixels; i++)
	{
		pHist = exq_find_histogram(pExq, pIn);
		if(pHist != NULL && pHist->palIndex != -1)
		{
			*pOut++ = (unsigned char)pHist->palIndex;
			pIn += 4;
		}
		else
		{
			c.r = *pIn++ / 255.0f * SCALE_R;
			c.g = *pIn++ / 255.0f * SCALE_G;
			c.b = *pIn++ / 255.0f * SCALE_B;
			c.a = *pIn++ / 255.0f * SCALE_A;

			if(pExq->transparency)
			{
				c.r *= c.a; c.g *= c.a; c.b *= c.a;
			}

			*pOut = exq_find_nearest_color(pExq, &c);
			if(pHist != NULL)
				pHist->palIndex = *pOut;
			pOut++;
		}
	}
}

void exq_map_image_ordered(exq_data *pExq, int width, int height,
						   unsigned char *pIn, unsigned char *pOut)
{
	exq_map_image_dither(pExq, width, height, pIn, pOut, 1);
}

void exq_map_image_random(exq_data *pExq, int nPixels,
						   unsigned char *pIn, unsigned char *pOut)
{
	exq_map_image_dither(pExq, nPixels, 1, pIn, pOut, 0);
}

void exq_map_image_dither(exq_data *pExq, int width, int height,
						  unsigned char *pIn, unsigned char *pOut, int ordered)
{
	int x, y, i, j, d;
	exq_color p, scale, tmp;
	exq_histogram *pHist;
	const exq_float dither_matrix[4] = { -0.375, 0.125, 0.375, -0.125 };

	if(!pExq->optimized)
		exq_optimize_palette(pExq, 4);

	for(y = 0; y < height; y++)
		for(x = 0; x < width; x++)
		{
			if(ordered)
				d = (x & 1) + (y & 1) * 2;
			else
				d = rand() & 3;
			pHist = exq_find_histogram(pExq, pIn);
			p.r = *pIn++ / 255.0f * SCALE_R;
			p.g = *pIn++ / 255.0f * SCALE_G;
			p.b = *pIn++ / 255.0f * SCALE_B;
			p.a = *pIn++ / 255.0f * SCALE_A;

			if(pExq->transparency)
			{
				p.r *= p.a; p.g *= p.a; p.b *= p.a;
			}

			if(pHist == NULL || pHist->ditherScale.r < 0)
			{
				i = exq_find_nearest_color(pExq, &p);
				scale.r = pExq->node[i].avg.r - p.r;
				scale.g = pExq->node[i].avg.g - p.g;
				scale.b = pExq->node[i].avg.b - p.b;
				scale.a = pExq->node[i].avg.a - p.a;
				tmp.r = p.r - scale.r / 3;
				tmp.g = p.g - scale.g / 3;
				tmp.b = p.b - scale.b / 3;
				tmp.a = p.a - scale.a / 3;
				j = exq_find_nearest_color(pExq, &tmp);
				if(i == j)
				{
					tmp.r = p.r - scale.r * 3;
					tmp.g = p.g - scale.g * 3;
					tmp.b = p.b - scale.b * 3;
					tmp.a = p.a - scale.a * 3;
					j = exq_find_nearest_color(pExq, &tmp);
				}
				if(i != j)
				{
					scale.r = (pExq->node[j].avg.r - pExq->node[i].avg.r) * 0.8f;
					scale.g = (pExq->node[j].avg.g - pExq->node[i].avg.g) * 0.8f;
					scale.b = (pExq->node[j].avg.b - pExq->node[i].avg.b) * 0.8f;
					scale.a = (pExq->node[j].avg.a - pExq->node[i].avg.a) * 0.8f;
					if(scale.r < 0) scale.r = -scale.r;
					if(scale.g < 0) scale.g = -scale.g;
					if(scale.b < 0) scale.b = -scale.b;
					if(scale.a < 0) scale.a = -scale.a;
				}
				else
					scale.r = scale.g = scale.b = scale.a = 0;

				if(pHist != NULL)
				{
					pHist->ditherScale.r = scale.r;
					pHist->ditherScale.g = scale.g;
					pHist->ditherScale.b = scale.b;
					pHist->ditherScale.a = scale.a;
				}
			}
			else
			{
				scale.r = pHist->ditherScale.r;
				scale.g = pHist->ditherScale.g;
				scale.b = pHist->ditherScale.b;
				scale.a = pHist->ditherScale.a;
			}

			if(pHist != NULL && pHist->ditherIndex[d] >= 0)
				*pOut++ = (unsigned char)pHist->ditherIndex[d];
			else
			{
				tmp.r = p.r + scale.r * dither_matrix[d];
				tmp.g = p.g + scale.g * dither_matrix[d];
				tmp.b = p.b + scale.b * dither_matrix[d];
				tmp.a = p.a + scale.a * dither_matrix[d];
				*pOut = exq_find_nearest_color(pExq, &tmp);
				if(pHist != NULL)
					pHist->ditherIndex[d] = *pOut;
				pOut++;
			}
		}
}

exq_histogram *exq_find_histogram(exq_data *pExq, unsigned char *pCol)
{
	unsigned int hash;
	int r, g, b, a;
	exq_histogram *pCur;

	r = *pCol++; g = *pCol++; b = *pCol++; a = *pCol++;
	hash = exq_make_hash(((unsigned int)r) | (((unsigned int)g) << 8) | (((unsigned int)b) << 16) | (((unsigned int)a) << 24));

	pCur = pExq->pHash[hash];
	while(pCur != NULL && (pCur->ored != r || pCur->ogreen != g ||
		pCur->oblue != b || pCur->oalpha != a))
		pCur = pCur->pNextInHash;

	return pCur;
}

unsigned char exq_find_nearest_color(exq_data *pExq, exq_color *pColor)
{
	exq_float bestv;
	int besti, i;
	exq_color dif;

	bestv = 16;
	besti = 0;
	for(i = 0; i < pExq->numColors; i++)
	{
		dif.r = pColor->r - pExq->node[i].avg.r;
		dif.g = pColor->g - pExq->node[i].avg.g;
		dif.b = pColor->b - pExq->node[i].avg.b;
		dif.a = pColor->a - pExq->node[i].avg.a;
		if(dif.r*dif.r + dif.g*dif.g + dif.b*dif.b + dif.a*dif.a < bestv)
		{
			bestv = dif.r*dif.r + dif.g*dif.g + dif.b*dif.b + dif.a*dif.a;
			besti = i;
		}
	}

	return (unsigned char)besti;
}

void exq_sort(exq_histogram **ppHist, exq_float (*sortfunc)(const exq_histogram *pHist))
{
	exq_histogram *pLow, *pHigh, *pCur, *pNext;
	int n = 0;
	exq_float sum = 0;

	for(pCur = *ppHist; pCur != NULL; pCur = pCur->pNext)
	{
		n++;
		sum += sortfunc(pCur);
	}

	if(n < 2)
		return;

	sum /= n;

	pLow = pHigh = NULL;
	for(pCur = *ppHist; pCur != NULL; pCur = pNext)
	{
		pNext = pCur->pNext;
		if(sortfunc(pCur) < sum)
		{
			pCur->pNext = pLow;
			pLow = pCur;
		}
		else
		{
			pCur->pNext = pHigh;
			pHigh = pCur;
		}
	}

	if(pLow == NULL)
	{
		*ppHist = pHigh;
		return;
	}
	if(pHigh == NULL)
	{
		*ppHist = pLow;
		return;
	}

	exq_sort(&pLow, sortfunc);
	exq_sort(&pHigh, sortfunc);

	*ppHist = pLow;
	while(pLow->pNext != NULL)
		pLow = pLow->pNext;

	pLow->pNext = pHigh;
}

exq_float exq_sort_by_r(const exq_histogram *pHist)
{
	return pHist->color.r;
}

exq_float exq_sort_by_g(const exq_histogram *pHist)
{
	return pHist->color.g;
}

exq_float exq_sort_by_b(const exq_histogram *pHist)
{
	return pHist->color.b;
}

exq_float exq_sort_by_a(const exq_histogram *pHist)
{
	return pHist->color.a;
}

exq_color exq_sort_dir;

exq_float exq_sort_by_dir(const exq_histogram *pHist)
{
	return pHist->color.r * exq_sort_dir.r +
		pHist->color.g * exq_sort_dir.g +
		pHist->color.b * exq_sort_dir.b +
		pHist->color.a * exq_sort_dir.a;
}