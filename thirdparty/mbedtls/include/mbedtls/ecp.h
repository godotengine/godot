/**
 * \file ecp.h
 *
 * \brief This file provides an API for Elliptic Curves over GF(P) (ECP).
 *
 * The use of ECP in cryptography and TLS is defined in
 * <em>Standards for Efficient Cryptography Group (SECG): SEC1
 * Elliptic Curve Cryptography</em> and
 * <em>RFC-4492: Elliptic Curve Cryptography (ECC) Cipher Suites
 * for Transport Layer Security (TLS)</em>.
 *
 * <em>RFC-2409: The Internet Key Exchange (IKE)</em> defines ECP
 * group types.
 *
 */

/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
 *
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

#ifndef MBEDTLS_ECP_H
#define MBEDTLS_ECP_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "bignum.h"

/*
 * ECP error codes
 */
#define MBEDTLS_ERR_ECP_BAD_INPUT_DATA                    -0x4F80  /**< Bad input parameters to function. */
#define MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL                  -0x4F00  /**< The buffer is too small to write to. */
#define MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE               -0x4E80  /**< The requested feature is not available, for example, the requested curve is not supported. */
#define MBEDTLS_ERR_ECP_VERIFY_FAILED                     -0x4E00  /**< The signature is not valid. */
#define MBEDTLS_ERR_ECP_ALLOC_FAILED                      -0x4D80  /**< Memory allocation failed. */
#define MBEDTLS_ERR_ECP_RANDOM_FAILED                     -0x4D00  /**< Generation of random value, such as ephemeral key, failed. */
#define MBEDTLS_ERR_ECP_INVALID_KEY                       -0x4C80  /**< Invalid private or public key. */
#define MBEDTLS_ERR_ECP_SIG_LEN_MISMATCH                  -0x4C00  /**< The buffer contains a valid signature followed by more data. */

/* MBEDTLS_ERR_ECP_HW_ACCEL_FAILED is deprecated and should not be used. */
#define MBEDTLS_ERR_ECP_HW_ACCEL_FAILED                   -0x4B80  /**< The ECP hardware accelerator failed. */

#define MBEDTLS_ERR_ECP_IN_PROGRESS                       -0x4B00  /**< Operation in progress, call again with the same parameters to continue. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Domain-parameter identifiers: curve, subgroup, and generator.
 *
 * \note Only curves over prime fields are supported.
 *
 * \warning This library does not support validation of arbitrary domain
 * parameters. Therefore, only standardized domain parameters from trusted
 * sources should be used. See mbedtls_ecp_group_load().
 */
typedef enum
{
    MBEDTLS_ECP_DP_NONE = 0,       /*!< Curve not defined. */
    MBEDTLS_ECP_DP_SECP192R1,      /*!< Domain parameters for the 192-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP224R1,      /*!< Domain parameters for the 224-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP256R1,      /*!< Domain parameters for the 256-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP384R1,      /*!< Domain parameters for the 384-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP521R1,      /*!< Domain parameters for the 521-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_BP256R1,        /*!< Domain parameters for 256-bit Brainpool curve. */
    MBEDTLS_ECP_DP_BP384R1,        /*!< Domain parameters for 384-bit Brainpool curve. */
    MBEDTLS_ECP_DP_BP512R1,        /*!< Domain parameters for 512-bit Brainpool curve. */
    MBEDTLS_ECP_DP_CURVE25519,     /*!< Domain parameters for Curve25519. */
    MBEDTLS_ECP_DP_SECP192K1,      /*!< Domain parameters for 192-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_SECP224K1,      /*!< Domain parameters for 224-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_SECP256K1,      /*!< Domain parameters for 256-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_CURVE448,       /*!< Domain parameters for Curve448. */
} mbedtls_ecp_group_id;

/**
 * The number of supported curves, plus one for #MBEDTLS_ECP_DP_NONE.
 *
 * \note Montgomery curves are currently excluded.
 */
#define MBEDTLS_ECP_DP_MAX     12

/**
 * Curve information, for use by other modules.
 */
typedef struct mbedtls_ecp_curve_info
{
    mbedtls_ecp_group_id grp_id;    /*!< An internal identifier. */
    uint16_t tls_id;                /*!< The TLS NamedCurve identifier. */
    uint16_t bit_size;              /*!< The curve size in bits. */
    const char *name;               /*!< A human-friendly name. */
} mbedtls_ecp_curve_info;

/**
 * \brief           The ECP point structure, in Jacobian coordinates.
 *
 * \note            All functions expect and return points satisfying
 *                  the following condition: <code>Z == 0</code> or
 *                  <code>Z == 1</code>. Other values of \p Z are
 *                  used only by internal functions.
 *                  The point is zero, or "at infinity", if <code>Z == 0</code>.
 *                  Otherwise, \p X and \p Y are its standard (affine)
 *                  coordinates.
 */
typedef struct mbedtls_ecp_point
{
    mbedtls_mpi X;          /*!< The X coordinate of the ECP point. */
    mbedtls_mpi Y;          /*!< The Y coordinate of the ECP point. */
    mbedtls_mpi Z;          /*!< The Z coordinate of the ECP point. */
}
mbedtls_ecp_point;

#if !defined(MBEDTLS_ECP_ALT)
/*
 * default mbed TLS elliptic curve arithmetic implementation
 *
 * (in case MBEDTLS_ECP_ALT is defined then the developer has to provide an
 * alternative implementation for the whole module and it will replace this
 * one.)
 */

/**
 * \brief           The ECP group structure.
 *
 * We consider two types of curve equations:
 * <ul><li>Short Weierstrass: <code>y^2 = x^3 + A x + B mod P</code>
 * (SEC1 + RFC-4492)</li>
 * <li>Montgomery: <code>y^2 = x^3 + A x^2 + x mod P</code> (Curve25519,
 * Curve448)</li></ul>
 * In both cases, the generator (\p G) for a prime-order subgroup is fixed.
 *
 * For Short Weierstrass, this subgroup is the whole curve, and its
 * cardinality is denoted by \p N. Our code requires that \p N is an
 * odd prime as mbedtls_ecp_mul() requires an odd number, and
 * mbedtls_ecdsa_sign() requires that it is prime for blinding purposes.
 *
 * For Montgomery curves, we do not store \p A, but <code>(A + 2) / 4</code>,
 * which is the quantity used in the formulas. Additionally, \p nbits is
 * not the size of \p N but the required size for private keys.
 *
 * If \p modp is NULL, reduction modulo \p P is done using a generic algorithm.
 * Otherwise, \p modp must point to a function that takes an \p mbedtls_mpi in the
 * range of <code>0..2^(2*pbits)-1</code>, and transforms it in-place to an integer
 * which is congruent mod \p P to the given MPI, and is close enough to \p pbits
 * in size, so that it may be efficiently brought in the 0..P-1 range by a few
 * additions or subtractions. Therefore, it is only an approximative modular
 * reduction. It must return 0 on success and non-zero on failure.
 *
 * \note        Alternative implementations must keep the group IDs distinct. If
 *              two group structures have the same ID, then they must be
 *              identical.
 *
 */
typedef struct mbedtls_ecp_group
{
    mbedtls_ecp_group_id id;    /*!< An internal group identifier. */
    mbedtls_mpi P;              /*!< The prime modulus of the base field. */
    mbedtls_mpi A;              /*!< For Short Weierstrass: \p A in the equation. For
                                     Montgomery curves: <code>(A + 2) / 4</code>. */
    mbedtls_mpi B;              /*!< For Short Weierstrass: \p B in the equation.
                                     For Montgomery curves: unused. */
    mbedtls_ecp_point G;        /*!< The generator of the subgroup used. */
    mbedtls_mpi N;              /*!< The order of \p G. */
    size_t pbits;               /*!< The number of bits in \p P.*/
    size_t nbits;               /*!< For Short Weierstrass: The number of bits in \p P.
                                     For Montgomery curves: the number of bits in the
                                     private keys. */
    unsigned int h;             /*!< \internal 1 if the constants are static. */
    int (*modp)(mbedtls_mpi *); /*!< The function for fast pseudo-reduction
                                     mod \p P (see above).*/
    int (*t_pre)(mbedtls_ecp_point *, void *);  /*!< Unused. */
    int (*t_post)(mbedtls_ecp_point *, void *); /*!< Unused. */
    void *t_data;               /*!< Unused. */
    mbedtls_ecp_point *T;       /*!< Pre-computed points for ecp_mul_comb(). */
    size_t T_size;              /*!< The number of pre-computed points. */
}
mbedtls_ecp_group;

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h, or define them using the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_ECP_MAX_BITS)
/**
 * The maximum size of the groups, that is, of \c N and \c P.
 */
#define MBEDTLS_ECP_MAX_BITS     521   /**< The maximum size of groups, in bits. */
#endif

#define MBEDTLS_ECP_MAX_BYTES    ( ( MBEDTLS_ECP_MAX_BITS + 7 ) / 8 )
#define MBEDTLS_ECP_MAX_PT_LEN   ( 2 * MBEDTLS_ECP_MAX_BYTES + 1 )

#if !defined(MBEDTLS_ECP_WINDOW_SIZE)
/*
 * Maximum "window" size used for point multiplication.
 * Default: 6.
 * Minimum value: 2. Maximum value: 7.
 *
 * Result is an array of at most ( 1 << ( MBEDTLS_ECP_WINDOW_SIZE - 1 ) )
 * points used for point multiplication. This value is directly tied to EC
 * peak memory usage, so decreasing it by one should roughly cut memory usage
 * by two (if large curves are in use).
 *
 * Reduction in size may reduce speed, but larger curves are impacted first.
 * Sample performances (in ECDHE handshakes/s, with FIXED_POINT_OPTIM = 1):
 *      w-size:     6       5       4       3       2
 *      521       145     141     135     120      97
 *      384       214     209     198     177     146
 *      256       320     320     303     262     226
 *      224       475     475     453     398     342
 *      192       640     640     633     587     476
 */
#define MBEDTLS_ECP_WINDOW_SIZE    6   /**< The maximum window size used. */
#endif /* MBEDTLS_ECP_WINDOW_SIZE */

#if !defined(MBEDTLS_ECP_FIXED_POINT_OPTIM)
/*
 * Trade memory for speed on fixed-point multiplication.
 *
 * This speeds up repeated multiplication of the generator (that is, the
 * multiplication in ECDSA signatures, and half of the multiplications in
 * ECDSA verification and ECDHE) by a factor roughly 3 to 4.
 *
 * The cost is increasing EC peak memory usage by a factor roughly 2.
 *
 * Change this value to 0 to reduce peak memory usage.
 */
#define MBEDTLS_ECP_FIXED_POINT_OPTIM  1   /**< Enable fixed-point speed-up. */
#endif /* MBEDTLS_ECP_FIXED_POINT_OPTIM */

/* \} name SECTION: Module settings */

#else  /* MBEDTLS_ECP_ALT */
#include "ecp_alt.h"
#endif /* MBEDTLS_ECP_ALT */

#if defined(MBEDTLS_ECP_RESTARTABLE)

/**
 * \brief           Internal restart context for multiplication
 *
 * \note            Opaque struct
 */
typedef struct mbedtls_ecp_restart_mul mbedtls_ecp_restart_mul_ctx;

/**
 * \brief           Internal restart context for ecp_muladd()
 *
 * \note            Opaque struct
 */
typedef struct mbedtls_ecp_restart_muladd mbedtls_ecp_restart_muladd_ctx;

/**
 * \brief           General context for resuming ECC operations
 */
typedef struct
{
    unsigned ops_done;                  /*!<  current ops count             */
    unsigned depth;                     /*!<  call depth (0 = top-level)    */
    mbedtls_ecp_restart_mul_ctx *rsm;   /*!<  ecp_mul_comb() sub-context    */
    mbedtls_ecp_restart_muladd_ctx *ma; /*!<  ecp_muladd() sub-context      */
} mbedtls_ecp_restart_ctx;

/*
 * Operation counts for restartable functions
 */
#define MBEDTLS_ECP_OPS_CHK   3 /*!< basic ops count for ecp_check_pubkey()  */
#define MBEDTLS_ECP_OPS_DBL   8 /*!< basic ops count for ecp_double_jac()    */
#define MBEDTLS_ECP_OPS_ADD  11 /*!< basic ops count for see ecp_add_mixed() */
#define MBEDTLS_ECP_OPS_INV 120 /*!< empirical equivalent for mpi_mod_inv()  */

/**
 * \brief           Internal; for restartable functions in other modules.
 *                  Check and update basic ops budget.
 *
 * \param grp       Group structure
 * \param rs_ctx    Restart context
 * \param ops       Number of basic ops to do
 *
 * \return          \c 0 if doing \p ops basic ops is still allowed,
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS otherwise.
 */
int mbedtls_ecp_check_budget( const mbedtls_ecp_group *grp,
                              mbedtls_ecp_restart_ctx *rs_ctx,
                              unsigned ops );

/* Utility macro for checking and updating ops budget */
#define MBEDTLS_ECP_BUDGET( ops )   \
    MBEDTLS_MPI_CHK( mbedtls_ecp_check_budget( grp, rs_ctx, \
                                               (unsigned) (ops) ) );

#else /* MBEDTLS_ECP_RESTARTABLE */

#define MBEDTLS_ECP_BUDGET( ops )   /* no-op; for compatibility */

/* We want to declare restartable versions of existing functions anyway */
typedef void mbedtls_ecp_restart_ctx;

#endif /* MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief    The ECP key-pair structure.
 *
 * A generic key-pair that may be used for ECDSA and fixed ECDH, for example.
 *
 * \note    Members are deliberately in the same order as in the
 *          ::mbedtls_ecdsa_context structure.
 */
typedef struct mbedtls_ecp_keypair
{
    mbedtls_ecp_group grp;      /*!<  Elliptic curve and base point     */
    mbedtls_mpi d;              /*!<  our secret value                  */
    mbedtls_ecp_point Q;        /*!<  our public value                  */
}
mbedtls_ecp_keypair;

/*
 * Point formats, from RFC 4492's enum ECPointFormat
 */
#define MBEDTLS_ECP_PF_UNCOMPRESSED    0   /**< Uncompressed point format. */
#define MBEDTLS_ECP_PF_COMPRESSED      1   /**< Compressed point format. */

/*
 * Some other constants from RFC 4492
 */
#define MBEDTLS_ECP_TLS_NAMED_CURVE    3   /**< The named_curve of ECCurveType. */

#if defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Set the maximum number of basic operations done in a row.
 *
 *                  If more operations are needed to complete a computation,
 *                  #MBEDTLS_ERR_ECP_IN_PROGRESS will be returned by the
 *                  function performing the computation. It is then the
 *                  caller's responsibility to either call again with the same
 *                  parameters until it returns 0 or an error code; or to free
 *                  the restart context if the operation is to be aborted.
 *
 *                  It is strictly required that all input parameters and the
 *                  restart context be the same on successive calls for the
 *                  same operation, but output parameters need not be the
 *                  same; they must not be used until the function finally
 *                  returns 0.
 *
 *                  This only applies to functions whose documentation
 *                  mentions they may return #MBEDTLS_ERR_ECP_IN_PROGRESS (or
 *                  #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS for functions in the
 *                  SSL module). For functions that accept a "restart context"
 *                  argument, passing NULL disables restart and makes the
 *                  function equivalent to the function with the same name
 *                  with \c _restartable removed. For functions in the ECDH
 *                  module, restart is disabled unless the function accepts
 *                  an "ECDH context" argument and
 *                  mbedtls_ecdh_enable_restart() was previously called on
 *                  that context. For function in the SSL module, restart is
 *                  only enabled for specific sides and key exchanges
 *                  (currently only for clients and ECDHE-ECDSA).
 *
 * \param max_ops   Maximum number of basic operations done in a row.
 *                  Default: 0 (unlimited).
 *                  Lower (non-zero) values mean ECC functions will block for
 *                  a lesser maximum amount of time.
 *
 * \note            A "basic operation" is defined as a rough equivalent of a
 *                  multiplication in GF(p) for the NIST P-256 curve.
 *                  As an indication, with default settings, a scalar
 *                  multiplication (full run of \c mbedtls_ecp_mul()) is:
 *                  - about 3300 basic operations for P-256
 *                  - about 9400 basic operations for P-384
 *
 * \note            Very low values are not always respected: sometimes
 *                  functions need to block for a minimum number of
 *                  operations, and will do so even if max_ops is set to a
 *                  lower value.  That minimum depends on the curve size, and
 *                  can be made lower by decreasing the value of
 *                  \c MBEDTLS_ECP_WINDOW_SIZE.  As an indication, here is the
 *                  lowest effective value for various curves and values of
 *                  that parameter (w for short):
 *                          w=6     w=5     w=4     w=3     w=2
 *                  P-256   208     208     160     136     124
 *                  P-384   682     416     320     272     248
 *                  P-521  1364     832     640     544     496
 *
 * \note            This setting is currently ignored by Curve25519.
 */
void mbedtls_ecp_set_max_ops( unsigned max_ops );

/**
 * \brief           Check if restart is enabled (max_ops != 0)
 *
 * \return          \c 0 if \c max_ops == 0 (restart disabled)
 * \return          \c 1 otherwise (restart enabled)
 */
int mbedtls_ecp_restart_is_enabled( void );
#endif /* MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief           This function retrieves the information defined in
 *                  mbedtls_ecp_curve_info() for all supported curves in order
 *                  of preference.
 *
 * \return          A statically allocated array. The last entry is 0.
 */
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_list( void );

/**
 * \brief           This function retrieves the list of internal group
 *                  identifiers of all supported curves in the order of
 *                  preference.
 *
 * \return          A statically allocated array,
 *                  terminated with MBEDTLS_ECP_DP_NONE.
 */
const mbedtls_ecp_group_id *mbedtls_ecp_grp_id_list( void );

/**
 * \brief           This function retrieves curve information from an internal
 *                  group identifier.
 *
 * \param grp_id    An \c MBEDTLS_ECP_DP_XXX value.
 *
 * \return          The associated curve information on success.
 * \return          NULL on failure.
 */
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_grp_id( mbedtls_ecp_group_id grp_id );

/**
 * \brief           This function retrieves curve information from a TLS
 *                  NamedCurve value.
 *
 * \param tls_id    An \c MBEDTLS_ECP_DP_XXX value.
 *
 * \return          The associated curve information on success.
 * \return          NULL on failure.
 */
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_tls_id( uint16_t tls_id );

/**
 * \brief           This function retrieves curve information from a
 *                  human-readable name.
 *
 * \param name      The human-readable name.
 *
 * \return          The associated curve information on success.
 * \return          NULL on failure.
 */
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_name( const char *name );

/**
 * \brief           This function initializes a point as zero.
 *
 * \param pt        The point to initialize.
 */
void mbedtls_ecp_point_init( mbedtls_ecp_point *pt );

/**
 * \brief           This function initializes an ECP group context
 *                  without loading any domain parameters.
 *
 * \note            After this function is called, domain parameters
 *                  for various ECP groups can be loaded through the
 *                  mbedtls_ecp_group_load() or mbedtls_ecp_tls_read_group()
 *                  functions.
 */
void mbedtls_ecp_group_init( mbedtls_ecp_group *grp );

/**
 * \brief           This function initializes a key pair as an invalid one.
 *
 * \param key       The key pair to initialize.
 */
void mbedtls_ecp_keypair_init( mbedtls_ecp_keypair *key );

/**
 * \brief           This function frees the components of a point.
 *
 * \param pt        The point to free.
 */
void mbedtls_ecp_point_free( mbedtls_ecp_point *pt );

/**
 * \brief           This function frees the components of an ECP group.
 *
 * \param grp       The group to free. This may be \c NULL, in which
 *                  case this function returns immediately. If it is not
 *                  \c NULL, it must point to an initialized ECP group.
 */
void mbedtls_ecp_group_free( mbedtls_ecp_group *grp );

/**
 * \brief           This function frees the components of a key pair.
 *
 * \param key       The key pair to free. This may be \c NULL, in which
 *                  case this function returns immediately. If it is not
 *                  \c NULL, it must point to an initialized ECP key pair.
 */
void mbedtls_ecp_keypair_free( mbedtls_ecp_keypair *key );

#if defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Initialize a restart context.
 *
 * \param ctx       The restart context to initialize. This must
 *                  not be \c NULL.
 */
void mbedtls_ecp_restart_init( mbedtls_ecp_restart_ctx *ctx );

/**
 * \brief           Free the components of a restart context.
 *
 * \param ctx       The restart context to free. This may be \c NULL, in which
 *                  case this function returns immediately. If it is not
 *                  \c NULL, it must point to an initialized restart context.
 */
void mbedtls_ecp_restart_free( mbedtls_ecp_restart_ctx *ctx );
#endif /* MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief           This function copies the contents of point \p Q into
 *                  point \p P.
 *
 * \param P         The destination point. This must be initialized.
 * \param Q         The source point. This must be initialized.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          Another negative error code for other kinds of failure.
 */
int mbedtls_ecp_copy( mbedtls_ecp_point *P, const mbedtls_ecp_point *Q );

/**
 * \brief           This function copies the contents of group \p src into
 *                  group \p dst.
 *
 * \param dst       The destination group. This must be initialized.
 * \param src       The source group. This must be initialized.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_group_copy( mbedtls_ecp_group *dst,
                            const mbedtls_ecp_group *src );

/**
 * \brief           This function sets a point to the point at infinity.
 *
 * \param pt        The point to set. This must be initialized.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_set_zero( mbedtls_ecp_point *pt );

/**
 * \brief           This function checks if a point is the point at infinity.
 *
 * \param pt        The point to test. This must be initialized.
 *
 * \return          \c 1 if the point is zero.
 * \return          \c 0 if the point is non-zero.
 * \return          A negative error code on failure.
 */
int mbedtls_ecp_is_zero( mbedtls_ecp_point *pt );

/**
 * \brief           This function compares two points.
 *
 * \note            This assumes that the points are normalized. Otherwise,
 *                  they may compare as "not equal" even if they are.
 *
 * \param P         The first point to compare. This must be initialized.
 * \param Q         The second point to compare. This must be initialized.
 *
 * \return          \c 0 if the points are equal.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if the points are not equal.
 */
int mbedtls_ecp_point_cmp( const mbedtls_ecp_point *P,
                           const mbedtls_ecp_point *Q );

/**
 * \brief           This function imports a non-zero point from two ASCII
 *                  strings.
 *
 * \param P         The destination point. This must be initialized.
 * \param radix     The numeric base of the input.
 * \param x         The first affine coordinate, as a null-terminated string.
 * \param y         The second affine coordinate, as a null-terminated string.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_MPI_XXX error code on failure.
 */
int mbedtls_ecp_point_read_string( mbedtls_ecp_point *P, int radix,
                           const char *x, const char *y );

/**
 * \brief           This function exports a point into unsigned binary data.
 *
 * \param grp       The group to which the point should belong.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param P         The point to export. This must be initialized.
 * \param format    The point format. This must be either
 *                  #MBEDTLS_ECP_PF_COMPRESSED or #MBEDTLS_ECP_PF_UNCOMPRESSED.
 * \param olen      The address at which to store the length of
 *                  the output in Bytes. This must not be \c NULL.
 * \param buf       The output buffer. This must be a writable buffer
 *                  of length \p buflen Bytes.
 * \param buflen    The length of the output buffer \p buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL if the output buffer
 *                  is too small to hold the point.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_point_write_binary( const mbedtls_ecp_group *grp, const mbedtls_ecp_point *P,
                            int format, size_t *olen,
                            unsigned char *buf, size_t buflen );

/**
 * \brief           This function imports a point from unsigned binary data.
 *
 * \note            This function does not check that the point actually
 *                  belongs to the given group, see mbedtls_ecp_check_pubkey()
 *                  for that.
 *
 * \param grp       The group to which the point should belong.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param P         The destination context to import the point to.
 *                  This must be initialized.
 * \param buf       The input buffer. This must be a readable buffer
 *                  of length \p ilen Bytes.
 * \param ilen      The length of the input buffer \p buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if the input is invalid.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          #MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE if the point format
 *                  is not implemented.
 */
int mbedtls_ecp_point_read_binary( const mbedtls_ecp_group *grp,
                                   mbedtls_ecp_point *P,
                                   const unsigned char *buf, size_t ilen );

/**
 * \brief           This function imports a point from a TLS ECPoint record.
 *
 * \note            On function return, \p *buf is updated to point immediately
 *                  after the ECPoint record.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param pt        The destination point.
 * \param buf       The address of the pointer to the start of the input buffer.
 * \param len       The length of the buffer.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_MPI_XXX error code on initialization
 *                  failure.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if input is invalid.
 */
int mbedtls_ecp_tls_read_point( const mbedtls_ecp_group *grp,
                                mbedtls_ecp_point *pt,
                                const unsigned char **buf, size_t len );

/**
 * \brief           This function exports a point as a TLS ECPoint record
 *                  defined in RFC 4492, Section 5.4.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param pt        The point to be exported. This must be initialized.
 * \param format    The point format to use. This must be either
 *                  #MBEDTLS_ECP_PF_COMPRESSED or #MBEDTLS_ECP_PF_UNCOMPRESSED.
 * \param olen      The address at which to store the length in Bytes
 *                  of the data written.
 * \param buf       The target buffer. This must be a writable buffer of
 *                  length \p blen Bytes.
 * \param blen      The length of the target buffer \p buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if the input is invalid.
 * \return          #MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL if the target buffer
 *                  is too small to hold the exported point.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_tls_write_point( const mbedtls_ecp_group *grp,
                                 const mbedtls_ecp_point *pt,
                                 int format, size_t *olen,
                                 unsigned char *buf, size_t blen );

/**
 * \brief           This function sets up an ECP group context
 *                  from a standardized set of domain parameters.
 *
 * \note            The index should be a value of the NamedCurve enum,
 *                  as defined in <em>RFC-4492: Elliptic Curve Cryptography
 *                  (ECC) Cipher Suites for Transport Layer Security (TLS)</em>,
 *                  usually in the form of an \c MBEDTLS_ECP_DP_XXX macro.
 *
 * \param grp       The group context to setup. This must be initialized.
 * \param id        The identifier of the domain parameter set to load.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE if \p id doesn't
 *                  correspond to a known group.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_group_load( mbedtls_ecp_group *grp, mbedtls_ecp_group_id id );

/**
 * \brief           This function sets up an ECP group context from a TLS
 *                  ECParameters record as defined in RFC 4492, Section 5.4.
 *
 * \note            The read pointer \p buf is updated to point right after
 *                  the ECParameters record on exit.
 *
 * \param grp       The group context to setup. This must be initialized.
 * \param buf       The address of the pointer to the start of the input buffer.
 * \param len       The length of the input buffer \c *buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if input is invalid.
 * \return          #MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE if the group is not
 *                  recognized.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_tls_read_group( mbedtls_ecp_group *grp,
                                const unsigned char **buf, size_t len );

/**
 * \brief           This function extracts an elliptic curve group ID from a
 *                  TLS ECParameters record as defined in RFC 4492, Section 5.4.
 *
 * \note            The read pointer \p buf is updated to point right after
 *                  the ECParameters record on exit.
 *
 * \param grp       The address at which to store the group id.
 *                  This must not be \c NULL.
 * \param buf       The address of the pointer to the start of the input buffer.
 * \param len       The length of the input buffer \c *buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if input is invalid.
 * \return          #MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE if the group is not
 *                  recognized.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_tls_read_group_id( mbedtls_ecp_group_id *grp,
                                   const unsigned char **buf,
                                   size_t len );
/**
 * \brief           This function exports an elliptic curve as a TLS
 *                  ECParameters record as defined in RFC 4492, Section 5.4.
 *
 * \param grp       The ECP group to be exported.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param olen      The address at which to store the number of Bytes written.
 *                  This must not be \c NULL.
 * \param buf       The buffer to write to. This must be a writable buffer
 *                  of length \p blen Bytes.
 * \param blen      The length of the output buffer \p buf in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL if the output
 *                  buffer is too small to hold the exported group.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_tls_write_group( const mbedtls_ecp_group *grp,
                                 size_t *olen,
                                 unsigned char *buf, size_t blen );

/**
 * \brief           This function performs a scalar multiplication of a point
 *                  by an integer: \p R = \p m * \p P.
 *
 *                  It is not thread-safe to use same group in multiple threads.
 *
 * \note            To prevent timing attacks, this function
 *                  executes the exact same sequence of base-field
 *                  operations for any valid \p m. It avoids any if-branch or
 *                  array index depending on the value of \p m.
 *
 * \note            If \p f_rng is not NULL, it is used to randomize
 *                  intermediate results to prevent potential timing attacks
 *                  targeting these results. We recommend always providing
 *                  a non-NULL \p f_rng. The overhead is negligible.
 *                  Note: unless #MBEDTLS_ECP_NO_INTERNAL_RNG is defined, when
 *                  \p f_rng is NULL, an internal RNG (seeded from the value
 *                  of \p m) will be used instead.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param R         The point in which to store the result of the calculation.
 *                  This must be initialized.
 * \param m         The integer by which to multiply. This must be initialized.
 * \param P         The point to multiply. This must be initialized.
 * \param f_rng     The RNG function. This may be \c NULL if randomization
 *                  of intermediate results isn't desired (discouraged).
 * \param p_rng     The RNG context to be passed to \p p_rng.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if \p m is not a valid private
 *                  key, or \p P is not a valid public key.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_mul( mbedtls_ecp_group *grp, mbedtls_ecp_point *R,
             const mbedtls_mpi *m, const mbedtls_ecp_point *P,
             int (*f_rng)(void *, unsigned char *, size_t), void *p_rng );

/**
 * \brief           This function performs multiplication of a point by
 *                  an integer: \p R = \p m * \p P in a restartable way.
 *
 * \see             mbedtls_ecp_mul()
 *
 * \note            This function does the same as \c mbedtls_ecp_mul(), but
 *                  it can return early and restart according to the limit set
 *                  with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param R         The point in which to store the result of the calculation.
 *                  This must be initialized.
 * \param m         The integer by which to multiply. This must be initialized.
 * \param P         The point to multiply. This must be initialized.
 * \param f_rng     The RNG function. This may be \c NULL if randomization
 *                  of intermediate results isn't desired (discouraged).
 * \param p_rng     The RNG context to be passed to \p p_rng.
 * \param rs_ctx    The restart context (NULL disables restart).
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if \p m is not a valid private
 *                  key, or \p P is not a valid public key.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_mul_restartable( mbedtls_ecp_group *grp, mbedtls_ecp_point *R,
             const mbedtls_mpi *m, const mbedtls_ecp_point *P,
             int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
             mbedtls_ecp_restart_ctx *rs_ctx );

/**
 * \brief           This function performs multiplication and addition of two
 *                  points by integers: \p R = \p m * \p P + \p n * \p Q
 *
 *                  It is not thread-safe to use same group in multiple threads.
 *
 * \note            In contrast to mbedtls_ecp_mul(), this function does not
 *                  guarantee a constant execution flow and timing.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param R         The point in which to store the result of the calculation.
 *                  This must be initialized.
 * \param m         The integer by which to multiply \p P.
 *                  This must be initialized.
 * \param P         The point to multiply by \p m. This must be initialized.
 * \param n         The integer by which to multiply \p Q.
 *                  This must be initialized.
 * \param Q         The point to be multiplied by \p n.
 *                  This must be initialized.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if \p m or \p n are not
 *                  valid private keys, or \p P or \p Q are not valid public
 *                  keys.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_muladd( mbedtls_ecp_group *grp, mbedtls_ecp_point *R,
             const mbedtls_mpi *m, const mbedtls_ecp_point *P,
             const mbedtls_mpi *n, const mbedtls_ecp_point *Q );

/**
 * \brief           This function performs multiplication and addition of two
 *                  points by integers: \p R = \p m * \p P + \p n * \p Q in a
 *                  restartable way.
 *
 * \see             \c mbedtls_ecp_muladd()
 *
 * \note            This function works the same as \c mbedtls_ecp_muladd(),
 *                  but it can return early and restart according to the limit
 *                  set with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param R         The point in which to store the result of the calculation.
 *                  This must be initialized.
 * \param m         The integer by which to multiply \p P.
 *                  This must be initialized.
 * \param P         The point to multiply by \p m. This must be initialized.
 * \param n         The integer by which to multiply \p Q.
 *                  This must be initialized.
 * \param Q         The point to be multiplied by \p n.
 *                  This must be initialized.
 * \param rs_ctx    The restart context (NULL disables restart).
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if \p m or \p n are not
 *                  valid private keys, or \p P or \p Q are not valid public
 *                  keys.
 * \return          #MBEDTLS_ERR_MPI_ALLOC_FAILED on memory-allocation failure.
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_muladd_restartable(
             mbedtls_ecp_group *grp, mbedtls_ecp_point *R,
             const mbedtls_mpi *m, const mbedtls_ecp_point *P,
             const mbedtls_mpi *n, const mbedtls_ecp_point *Q,
             mbedtls_ecp_restart_ctx *rs_ctx );

/**
 * \brief           This function checks that a point is a valid public key
 *                  on this curve.
 *
 *                  It only checks that the point is non-zero, has
 *                  valid coordinates and lies on the curve. It does not verify
 *                  that it is indeed a multiple of \p G. This additional
 *                  check is computationally more expensive, is not required
 *                  by standards, and should not be necessary if the group
 *                  used has a small cofactor. In particular, it is useless for
 *                  the NIST groups which all have a cofactor of 1.
 *
 * \note            This function uses bare components rather than an
 *                  ::mbedtls_ecp_keypair structure, to ease use with other
 *                  structures, such as ::mbedtls_ecdh_context or
 *                  ::mbedtls_ecdsa_context.
 *
 * \param grp       The ECP group the point should belong to.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param pt        The point to check. This must be initialized.
 *
 * \return          \c 0 if the point is a valid public key.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if the point is not
 *                  a valid public key for the given curve.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_check_pubkey( const mbedtls_ecp_group *grp,
                              const mbedtls_ecp_point *pt );

/**
 * \brief           This function checks that an \p mbedtls_mpi is a
 *                  valid private key for this curve.
 *
 * \note            This function uses bare components rather than an
 *                  ::mbedtls_ecp_keypair structure to ease use with other
 *                  structures, such as ::mbedtls_ecdh_context or
 *                  ::mbedtls_ecdsa_context.
 *
 * \param grp       The ECP group the private key should belong to.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param d         The integer to check. This must be initialized.
 *
 * \return          \c 0 if the point is a valid private key.
 * \return          #MBEDTLS_ERR_ECP_INVALID_KEY if the point is not a valid
 *                  private key for the given curve.
 * \return          Another negative error code on other kinds of failure.
 */
int mbedtls_ecp_check_privkey( const mbedtls_ecp_group *grp,
                               const mbedtls_mpi *d );

/**
 * \brief           This function generates a private key.
 *
 * \param grp       The ECP group to generate a private key for.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param d         The destination MPI (secret part). This must be initialized.
 * \param f_rng     The RNG function. This must not be \c NULL.
 * \param p_rng     The RNG parameter to be passed to \p f_rng. This may be
 *                  \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX error code
 *                  on failure.
 */
int mbedtls_ecp_gen_privkey( const mbedtls_ecp_group *grp,
                     mbedtls_mpi *d,
                     int (*f_rng)(void *, unsigned char *, size_t),
                     void *p_rng );

/**
 * \brief           This function generates a keypair with a configurable base
 *                  point.
 *
 * \note            This function uses bare components rather than an
 *                  ::mbedtls_ecp_keypair structure to ease use with other
 *                  structures, such as ::mbedtls_ecdh_context or
 *                  ::mbedtls_ecdsa_context.
 *
 * \param grp       The ECP group to generate a key pair for.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param G         The base point to use. This must be initialized
 *                  and belong to \p grp. It replaces the default base
 *                  point \c grp->G used by mbedtls_ecp_gen_keypair().
 * \param d         The destination MPI (secret part).
 *                  This must be initialized.
 * \param Q         The destination point (public part).
 *                  This must be initialized.
 * \param f_rng     The RNG function. This must not be \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may
 *                  be \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX error code
 *                  on failure.
 */
int mbedtls_ecp_gen_keypair_base( mbedtls_ecp_group *grp,
                                  const mbedtls_ecp_point *G,
                                  mbedtls_mpi *d, mbedtls_ecp_point *Q,
                                  int (*f_rng)(void *, unsigned char *, size_t),
                                  void *p_rng );

/**
 * \brief           This function generates an ECP keypair.
 *
 * \note            This function uses bare components rather than an
 *                  ::mbedtls_ecp_keypair structure to ease use with other
 *                  structures, such as ::mbedtls_ecdh_context or
 *                  ::mbedtls_ecdsa_context.
 *
 * \param grp       The ECP group to generate a key pair for.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param d         The destination MPI (secret part).
 *                  This must be initialized.
 * \param Q         The destination point (public part).
 *                  This must be initialized.
 * \param f_rng     The RNG function. This must not be \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may
 *                  be \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX error code
 *                  on failure.
 */
int mbedtls_ecp_gen_keypair( mbedtls_ecp_group *grp, mbedtls_mpi *d,
                             mbedtls_ecp_point *Q,
                             int (*f_rng)(void *, unsigned char *, size_t),
                             void *p_rng );

/**
 * \brief           This function generates an ECP key.
 *
 * \param grp_id    The ECP group identifier.
 * \param key       The destination key. This must be initialized.
 * \param f_rng     The RNG function to use. This must not be \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may
 *                  be \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX error code
 *                  on failure.
 */
int mbedtls_ecp_gen_key( mbedtls_ecp_group_id grp_id, mbedtls_ecp_keypair *key,
                         int (*f_rng)(void *, unsigned char *, size_t),
                         void *p_rng );

/**
 * \brief           This function checks that the keypair objects
 *                  \p pub and \p prv have the same group and the
 *                  same public point, and that the private key in
 *                  \p prv is consistent with the public key.
 *
 * \param pub       The keypair structure holding the public key. This
 *                  must be initialized. If it contains a private key, that
 *                  part is ignored.
 * \param prv       The keypair structure holding the full keypair.
 *                  This must be initialized.
 *
 * \return          \c 0 on success, meaning that the keys are valid and match.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if the keys are invalid or do not match.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or an \c MBEDTLS_ERR_MPI_XXX
 *                  error code on calculation failure.
 */
int mbedtls_ecp_check_pub_priv( const mbedtls_ecp_keypair *pub,
                                const mbedtls_ecp_keypair *prv );

#if defined(MBEDTLS_SELF_TEST)

/**
 * \brief          The ECP checkup routine.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 */
int mbedtls_ecp_self_test( int verbose );

#endif /* MBEDTLS_SELF_TEST */

#ifdef __cplusplus
}
#endif

#endif /* ecp.h */
