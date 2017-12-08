// This code is in the public domain -- Ignacio Castaño <castanyo@yahoo.es>

#include "Sparse.h"
#include "KahanSum.h"

#include "nvcore/Array.inl"

#define USE_KAHAN_SUM 0


using namespace nv;


FullVector::FullVector(uint dim)
{ 
    m_array.resize(dim); 
}

FullVector::FullVector(const FullVector & v) : m_array(v.m_array)
{
}

const FullVector & FullVector::operator=(const FullVector & v)
{
    nvCheck(dimension() == v.dimension());

    m_array = v.m_array;

    return *this;
}


void FullVector::fill(float f)
{
    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] = f;
    }
}

void FullVector::operator+= (const FullVector & v)
{
    nvDebugCheck(dimension() == v.dimension());

    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] += v.m_array[i];
    }
}

void FullVector::operator-= (const FullVector & v)
{
    nvDebugCheck(dimension() == v.dimension());

    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] -= v.m_array[i];
    }
}

void FullVector::operator*= (const FullVector & v)
{
    nvDebugCheck(dimension() == v.dimension());

    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] *= v.m_array[i];
    }
}

void FullVector::operator+= (float f)
{
    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] += f;
    }
}

void FullVector::operator-= (float f)
{
    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] -= f;
    }
}

void FullVector::operator*= (float f)
{
    const uint dim = dimension();
    for (uint i = 0; i < dim; i++)
    {
        m_array[i] *= f;
    }
}


void nv::saxpy(float a, const FullVector & x, FullVector & y)
{
    nvDebugCheck(x.dimension() == y.dimension());

    const uint dim = x.dimension();
    for (uint i = 0; i < dim; i++)
    {
        y[i] += a * x[i];
    }
}

void nv::copy(const FullVector & x, FullVector & y)
{
    nvDebugCheck(x.dimension() == y.dimension());

    const uint dim = x.dimension();
    for (uint i = 0; i < dim; i++)
    {
        y[i] = x[i];
    }
}

void nv::scal(float a, FullVector & x)
{
    const uint dim = x.dimension();
    for (uint i = 0; i < dim; i++)
    {
        x[i] *= a;
    }
}

float nv::dot(const FullVector & x, const FullVector & y)
{
    nvDebugCheck(x.dimension() == y.dimension());

    const uint dim = x.dimension();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < dim; i++)
    {
        kahan.add(x[i] * y[i]);
    }
    return kahan.sum();
#else
    float sum = 0;
    for (uint i = 0; i < dim; i++)
    {
        sum += x[i] * y[i];
    }
    return sum;
#endif
}


FullMatrix::FullMatrix(uint d) : m_width(d), m_height(d)
{
    m_array.resize(d*d, 0.0f);
}

FullMatrix::FullMatrix(uint w, uint h) : m_width(w), m_height(h)
{
    m_array.resize(w*h, 0.0f);
}

FullMatrix::FullMatrix(const FullMatrix & m) : m_width(m.m_width), m_height(m.m_height)
{
    m_array = m.m_array;
}

const FullMatrix & FullMatrix::operator=(const FullMatrix & m)
{
    nvCheck(width() == m.width());
    nvCheck(height() == m.height());

    m_array = m.m_array;

    return *this;
}


float FullMatrix::getCoefficient(uint x, uint y) const
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    return m_array[y * width() + x];
}

void FullMatrix::setCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    m_array[y * width() + x] = f;
}

void FullMatrix::addCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    m_array[y * width() + x] += f;
}

void FullMatrix::mulCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    m_array[y * width() + x] *= f;
}

float FullMatrix::dotRow(uint y, const FullVector & v) const
{
    nvDebugCheck( v.dimension() == width() );
    nvDebugCheck( y < height() );

    float sum = 0;

    const uint count = v.dimension();
    for (uint i = 0; i < count; i++)
    {
        sum += m_array[y * count + i] * v[i];
    }

    return sum;
}

void FullMatrix::madRow(uint y, float alpha, FullVector & v) const
{
    nvDebugCheck( v.dimension() == width() );
    nvDebugCheck( y < height() );

    const uint count = v.dimension();
    for (uint i = 0; i < count; i++)
    {
        v[i] += m_array[y * count + i];
    }
}


// y = M * x
void nv::mult(const FullMatrix & M, const FullVector & x, FullVector & y)
{
    mult(NoTransposed, M, x, y);
}

void nv::mult(Transpose TM, const FullMatrix & M, const FullVector & x, FullVector & y)
{
    const uint w = M.width();
    const uint h = M.height();

    if (TM == Transposed)
    {
        nvDebugCheck( h == x.dimension() );
        nvDebugCheck( w == y.dimension() );

        y.fill(0.0f);

        for (uint i = 0; i < h; i++)
        {
            M.madRow(i, x[i], y);
        }
    }
    else
    {
        nvDebugCheck( w == x.dimension() );
        nvDebugCheck( h == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            y[i] = M.dotRow(i, x);
        }
    }
}

// y = alpha*A*x + beta*y
void nv::sgemv(float alpha, const FullMatrix & A, const FullVector & x, float beta, FullVector & y)
{
    sgemv(alpha, NoTransposed, A, x, beta, y);
}

void nv::sgemv(float alpha, Transpose TA, const FullMatrix & A, const FullVector & x, float beta, FullVector & y)
{
    const uint w = A.width();
    const uint h = A.height();

    if (TA == Transposed)
    {
        nvDebugCheck( h == x.dimension() );
        nvDebugCheck( w == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            A.madRow(i, alpha * x[i], y);
        }
    }
    else
    {
        nvDebugCheck( w == x.dimension() );
        nvDebugCheck( h == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            y[i] = alpha * A.dotRow(i, x) + beta * y[i];
        }
    }
}


// Multiply a row of A by a column of B.
static float dot(uint j, Transpose TA, const FullMatrix & A, uint i, Transpose TB, const FullMatrix & B)
{
    const uint w = (TA == NoTransposed) ? A.width() : A.height();
    nvDebugCheck(w == ((TB == NoTransposed) ? B.height() : A.width()));

    float sum = 0.0f;

    for (uint k = 0; k < w; k++)
    {
        const float a = (TA == NoTransposed) ? A.getCoefficient(k, j) : A.getCoefficient(j, k); // @@ Move branches out of the loop?
        const float b = (TB == NoTransposed) ? B.getCoefficient(i, k) : A.getCoefficient(k, i);
        sum += a * b;
    }

    return sum;
}


// C = A * B
void nv::mult(const FullMatrix & A, const FullMatrix & B, FullMatrix & C)
{
    mult(NoTransposed, A, NoTransposed, B, C);
}

void nv::mult(Transpose TA, const FullMatrix & A, Transpose TB, const FullMatrix & B, FullMatrix & C)
{
    sgemm(1.0f, TA, A, TB, B, 0.0f, C);
}

// C = alpha*A*B + beta*C
void nv::sgemm(float alpha, const FullMatrix & A, const FullMatrix & B, float beta, FullMatrix & C)
{
    sgemm(alpha, NoTransposed, A, NoTransposed, B, beta, C);
}

void nv::sgemm(float alpha, Transpose TA, const FullMatrix & A, Transpose TB, const FullMatrix & B, float beta, FullMatrix & C)
{
    const uint w = C.width();
    const uint h = C.height();

    uint aw = (TA == NoTransposed) ? A.width() : A.height();
    uint ah = (TA == NoTransposed) ? A.height() : A.width();
    uint bw = (TB == NoTransposed) ? B.width() : B.height();
    uint bh = (TB == NoTransposed) ? B.height() : B.width();

    nvDebugCheck(aw == bh);
    nvDebugCheck(bw == ah);
    nvDebugCheck(w == bw);
    nvDebugCheck(h == ah);

    for (uint y = 0; y < h; y++)
    {
        for (uint x = 0; x < w; x++)
        {
            float c = alpha * ::dot(x, TA, A, y, TB, B) + beta * C.getCoefficient(x, y);
            C.setCoefficient(x, y, c);
        }
    }
}





/// Ctor. Init the size of the sparse matrix.
SparseMatrix::SparseMatrix(uint d) : m_width(d)
{
    m_array.resize(d);
}

/// Ctor. Init the size of the sparse matrix.
SparseMatrix::SparseMatrix(uint w, uint h) : m_width(w)
{
    m_array.resize(h);
}

SparseMatrix::SparseMatrix(const SparseMatrix & m) : m_width(m.m_width)
{
    m_array = m.m_array;
}

const SparseMatrix & SparseMatrix::operator=(const SparseMatrix & m)
{
    nvCheck(width() == m.width());
    nvCheck(height() == m.height());

    m_array = m.m_array;

    return *this;
}


// x is column, y is row
float SparseMatrix::getCoefficient(uint x, uint y) const
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        if (m_array[y][i].x == x) return m_array[y][i].v;
    }

    return 0.0f;
}

void SparseMatrix::setCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        if (m_array[y][i].x == x) 
        {
            m_array[y][i].v = f;
            return;
        }
    }

    if (f != 0.0f)
    {
        Coefficient c = { x, f };
        m_array[y].append( c );
    }
}

void SparseMatrix::addCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    if (f != 0.0f)
    {
        const uint count = m_array[y].count();
        for (uint i = 0; i < count; i++)
        {
            if (m_array[y][i].x == x) 
            {
                m_array[y][i].v += f;
                return;
            }
        }

        Coefficient c = { x, f };
        m_array[y].append( c );
    }
}

void SparseMatrix::mulCoefficient(uint x, uint y, float f)
{
    nvDebugCheck( x < width() );
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        if (m_array[y][i].x == x) 
        {
            m_array[y][i].v *= f;
            return;
        }
    }

    if (f != 0.0f)
    {
        Coefficient c = { x, f };
        m_array[y].append( c );
    }
}


float SparseMatrix::sumRow(uint y) const
{
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < count; i++)
    {
        kahan.add(m_array[y][i].v);
    }
    return kahan.sum();
#else
    float sum = 0;
    for (uint i = 0; i < count; i++)
    {
        sum += m_array[y][i].v;
    }
    return sum;
#endif
}

float SparseMatrix::dotRow(uint y, const FullVector & v) const
{
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < count; i++)
    {
        kahan.add(m_array[y][i].v * v[m_array[y][i].x]);
    }
    return kahan.sum();
#else
    float sum = 0;
    for (uint i = 0; i < count; i++)
    {
        sum += m_array[y][i].v * v[m_array[y][i].x];
    }
    return sum;
#endif
}

void SparseMatrix::madRow(uint y, float alpha, FullVector & v) const
{
    nvDebugCheck(y < height());

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        v[m_array[y][i].x] += alpha * m_array[y][i].v;
    }
}


void SparseMatrix::clearRow(uint y)
{
    nvDebugCheck( y < height() );

    m_array[y].clear();
}

void SparseMatrix::scaleRow(uint y, float f)
{
    nvDebugCheck( y < height() );

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        m_array[y][i].v *= f;
    }
}

void SparseMatrix::normalizeRow(uint y)
{
    nvDebugCheck( y < height() );

    float norm = 0.0f;

    const uint count = m_array[y].count();
    for (uint i = 0; i < count; i++)
    {
        float f = m_array[y][i].v;
        norm += f * f;
    }

    scaleRow(y, 1.0f / sqrtf(norm));
}


void SparseMatrix::clearColumn(uint x)
{
    nvDebugCheck(x < width());

    for (uint y = 0; y < height(); y++)
    {
        const uint count = m_array[y].count();
        for (uint e = 0; e < count; e++)
        {
            if (m_array[y][e].x == x)
            {
                m_array[y][e].v = 0.0f;
                break;
            }
        }
    }
}

void SparseMatrix::scaleColumn(uint x, float f)
{
    nvDebugCheck(x < width());

    for (uint y = 0; y < height(); y++)
    {
        const uint count = m_array[y].count();
        for (uint e = 0; e < count; e++)
        {
            if (m_array[y][e].x == x)
            {
                m_array[y][e].v *= f;
                break;
            }
        }
    }
}

const Array<SparseMatrix::Coefficient> & SparseMatrix::getRow(uint y) const
{
    return m_array[y];
}


bool SparseMatrix::isSymmetric() const
{
    for (uint y = 0; y < height(); y++)
    {
        const uint count = m_array[y].count();
        for (uint e = 0; e < count; e++)
        {
            const uint x = m_array[y][e].x;
            if (x > y) {
                float v = m_array[y][e].v;

                if (!equal(getCoefficient(y, x), v)) {  // @@ epsilon
                    return false;
                }
            }
        }
    }

    return true;
}


// y = M * x
void nv::mult(const SparseMatrix & M, const FullVector & x, FullVector & y)
{
    mult(NoTransposed, M, x, y);
}

void nv::mult(Transpose TM, const SparseMatrix & M, const FullVector & x, FullVector & y)
{
    const uint w = M.width();
    const uint h = M.height();

    if (TM == Transposed)
    {
        nvDebugCheck( h == x.dimension() );
        nvDebugCheck( w == y.dimension() );

        y.fill(0.0f);

        for (uint i = 0; i < h; i++)
        {
            M.madRow(i, x[i], y);
        }
    }
    else
    {
        nvDebugCheck( w == x.dimension() );
        nvDebugCheck( h == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            y[i] = M.dotRow(i, x);
        }
    }
}

// y = alpha*A*x + beta*y
void nv::sgemv(float alpha, const SparseMatrix & A, const FullVector & x, float beta, FullVector & y)
{
    sgemv(alpha, NoTransposed, A, x, beta, y);
}

void nv::sgemv(float alpha, Transpose TA, const SparseMatrix & A, const FullVector & x, float beta, FullVector & y)
{
    const uint w = A.width();
    const uint h = A.height();

    if (TA == Transposed)
    {
        nvDebugCheck( h == x.dimension() );
        nvDebugCheck( w == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            A.madRow(i, alpha * x[i], y);
        }
    }
    else
    {
        nvDebugCheck( w == x.dimension() );
        nvDebugCheck( h == y.dimension() );

        for (uint i = 0; i < h; i++)
        {
            y[i] = alpha * A.dotRow(i, x) + beta * y[i];
        }
    }
}


// dot y-row of A by x-column of B
static float dotRowColumn(int y, const SparseMatrix & A, int x, const SparseMatrix & B)
{
    const Array<SparseMatrix::Coefficient> & row = A.getRow(y);

    const uint count = row.count();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < count; i++)
    {
        const SparseMatrix::Coefficient & c = row[i];
        kahan.add(c.v * B.getCoefficient(x, c.x));
    }
    return kahan.sum();
#else
    float sum = 0.0f;
    for (uint i = 0; i < count; i++)
    {
        const SparseMatrix::Coefficient & c = row[i];
        sum += c.v * B.getCoefficient(x, c.x);
    }
    return sum;
#endif
}

// dot y-row of A by x-row of B
static float dotRowRow(int y, const SparseMatrix & A, int x, const SparseMatrix & B)
{
    const Array<SparseMatrix::Coefficient> & row = A.getRow(y);

    const uint count = row.count();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < count; i++)
    {
        const SparseMatrix::Coefficient & c = row[i];
        kahan.add(c.v * B.getCoefficient(c.x, x));
    }
    return kahan.sum();
#else
    float sum = 0.0f;
    for (uint i = 0; i < count; i++)
    {
        const SparseMatrix::Coefficient & c = row[i];
        sum += c.v * B.getCoefficient(c.x, x);
    }
    return sum;
#endif
}

// dot y-column of A by x-column of B
static float dotColumnColumn(int y, const SparseMatrix & A, int x, const SparseMatrix & B)
{
    nvDebugCheck(A.height() == B.height());

    const uint h = A.height();

#if USE_KAHAN_SUM
    KahanSum kahan;
    for (uint i = 0; i < h; i++)
    {
        kahan.add(A.getCoefficient(y, i) * B.getCoefficient(x, i));
    }
    return kahan.sum();
#else
    float sum = 0.0f;
    for (uint i = 0; i < h; i++)
    {
        sum += A.getCoefficient(y, i) * B.getCoefficient(x, i);
    }
    return sum;
#endif
}


void nv::transpose(const SparseMatrix & A, SparseMatrix & B)
{
    nvDebugCheck(A.width() == B.height());
    nvDebugCheck(B.width() == A.height());

    const uint w = A.width();
    for (uint x = 0; x < w; x++)
    {
        B.clearRow(x);
    }

    const uint h = A.height();
    for (uint y = 0; y < h; y++)
    {
        const Array<SparseMatrix::Coefficient> & row = A.getRow(y);

        const uint count = row.count();
        for (uint i = 0; i < count; i++)
        {
            const SparseMatrix::Coefficient & c = row[i];
            nvDebugCheck(c.x < w);

            B.setCoefficient(y, c.x, c.v);
        }
    }
}

// C = A * B
void nv::mult(const SparseMatrix & A, const SparseMatrix & B, SparseMatrix & C)
{
    mult(NoTransposed, A, NoTransposed, B, C);
}

void nv::mult(Transpose TA, const SparseMatrix & A, Transpose TB, const SparseMatrix & B, SparseMatrix & C)
{
    sgemm(1.0f, TA, A, TB, B, 0.0f, C);
}

// C = alpha*A*B + beta*C
void nv::sgemm(float alpha, const SparseMatrix & A, const SparseMatrix & B, float beta, SparseMatrix & C)
{
    sgemm(alpha, NoTransposed, A, NoTransposed, B, beta, C);
}

void nv::sgemm(float alpha, Transpose TA, const SparseMatrix & A, Transpose TB, const SparseMatrix & B, float beta, SparseMatrix & C)
{
    const uint w = C.width();
    const uint h = C.height();

    uint aw = (TA == NoTransposed) ? A.width() : A.height();
    uint ah = (TA == NoTransposed) ? A.height() : A.width();
    uint bw = (TB == NoTransposed) ? B.width() : B.height();
    uint bh = (TB == NoTransposed) ? B.height() : B.width();

    nvDebugCheck(aw == bh);
    nvDebugCheck(bw == ah);
    nvDebugCheck(w == bw);
    nvDebugCheck(h == ah);


    for (uint y = 0; y < h; y++)
    {
        for (uint x = 0; x < w; x++)
        {
            float c = beta * C.getCoefficient(x, y);

            if (TA == NoTransposed && TB == NoTransposed)
            {
                // dot y-row of A by x-column of B.
                c += alpha * dotRowColumn(y, A, x, B);
            }
            else if (TA == Transposed && TB == Transposed)
            {
                // dot y-column of A by x-row of B.
                c += alpha * dotRowColumn(x, B, y, A);
            }
            else if (TA == Transposed && TB == NoTransposed)
            {
                // dot y-column of A by x-column of B.
                c += alpha * dotColumnColumn(y, A, x, B);
            }
            else if (TA == NoTransposed && TB == Transposed)
            {
                // dot y-row of A by x-row of B.
                c += alpha * dotRowRow(y, A, x, B);
            }

            C.setCoefficient(x, y, c);
        }
    }
}

// C = At * A
void nv::sqm(const SparseMatrix & A, SparseMatrix & C)
{
    // This is quite expensive...
    mult(Transposed, A, NoTransposed, A, C);
}
