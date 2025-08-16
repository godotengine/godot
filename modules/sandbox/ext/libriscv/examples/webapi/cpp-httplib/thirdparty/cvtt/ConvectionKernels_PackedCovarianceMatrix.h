#pragma once
#ifndef __CVTT_COVARIANCEMATRIX_H__
#define __CVTT_COVARIANCEMATRIX_H__

namespace cvtt
{
    namespace Internal
    {

        template<int TMatrixSize>
        class PackedCovarianceMatrix
        {
        public:
            // 0: xx,
            // 1: xy, yy
            // 3: xz, yz, zz 
            // 6: xw, yw, zw, ww
            // ... etc.
            static const int PyramidSize = (TMatrixSize * (TMatrixSize + 1)) / 2;

            typedef ParallelMath::Float MFloat;

            PackedCovarianceMatrix()
            {
                for (int i = 0; i < PyramidSize; i++)
                    m_values[i] = ParallelMath::MakeFloatZero();
            }

            void Add(const ParallelMath::Float *vec, const ParallelMath::Float &weight)
            {
                int index = 0;
                for (int row = 0; row < TMatrixSize; row++)
                {
                    for (int col = 0; col <= row; col++)
                    {
                        m_values[index] = m_values[index] + vec[row] * vec[col] * weight;
                        index++;
                    }
                }
            }

            void Product(MFloat *outVec, const MFloat *inVec)
            {
                for (int row = 0; row < TMatrixSize; row++)
                {
                    MFloat sum = ParallelMath::MakeFloatZero();

                    int index = (row * (row + 1)) >> 1;
                    for (int col = 0; col < TMatrixSize; col++)
                    {
                        sum = sum + inVec[col] * m_values[index];
                        if (col >= row)
                            index += col + 1;
                        else
                            index++;
                    }

                    outVec[row] = sum;
                }
            }

        private:
            ParallelMath::Float m_values[PyramidSize];
        };
    }
}

#endif
