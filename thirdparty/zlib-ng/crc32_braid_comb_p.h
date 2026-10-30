#ifndef CRC32_BRAID_COMB_P_H_
#define CRC32_BRAID_COMB_P_H_

/*
  Return a(x) multiplied by b(x) modulo p(x), where p(x) is the CRC polynomial,
  reflected. For speed, this requires that a not be zero.
 */
static uint32_t multmodp(uint32_t a, uint32_t b) {
    uint32_t m, p;

    m = (uint32_t)1 << 31;
    p = 0;
    for (;;) {
        if (a & m) {
            p ^= b;
            if ((a & (m - 1)) == 0)
                break;
        }
        m >>= 1;
        b = b & 1 ? (b >> 1) ^ POLY : b >> 1;
    }
    return p;
}

/*
  Return x^(n * 2^k) modulo p(x). Requires that x2n_table[] has been
  initialized.
 */
static uint32_t x2nmodp(z_off64_t n, unsigned k) {
    uint32_t p;

    p = (uint32_t)1 << 31;           /* x^0 == 1 */
    while (n) {
        if (n & 1)
            p = multmodp(x2n_table[k & 31], p);
        n >>= 1;
        k++;
    }
    return p;
}

#endif /* CRC32_BRAID_COMB_P_H_ */
