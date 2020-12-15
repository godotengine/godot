# the include directory
_mbedtls_incdirs = include_directories('include')

_mbedtls_deps = []
if host_machine.system() == 'windows'
    _mbedtls_deps += meson.get_compiler('cpp').find_library('bcrypt')
endif


# builtin
if get_option('builtin_mbedtls') and 'mbedtls' in MODULES_DISABLED
    _mbedtls_srcs = files([
        'library/aes.c',
        'library/base64.c',
        'library/godot_core_mbedtls_platform.c',
        'library/md5.c',
        'library/sha1.c',
        'library/sha256.c',
        'library/platform_util.c'
    ])

    _lib_mbedtls = static_library('mbedtls', _mbedtls_srcs, include_directories: _mbedtls_incdirs, dependencies: _mbedtls_deps, build_by_default: false)
    DEP_MBEDTLS = declare_dependency(link_with: _lib_mbedtls, include_directories: _mbedtls_incdirs)
elif get_option('builtin_mbedtls')
    # The full src for the full dependency
    _mbedtls_srcs = files([
        'library/aes.c',
        'library/aesni.c',
        'library/arc4.c',
        'library/aria.c',
        'library/asn1parse.c',
        'library/asn1write.c',
        'library/base64.c',
        'library/bignum.c',
        'library/blowfish.c',
        'library/camellia.c',
        'library/ccm.c',
        'library/certs.c',
        'library/chacha20.c',
        'library/chachapoly.c',
        'library/cipher.c',
        'library/cipher_wrap.c',
        'library/cmac.c',
        'library/ctr_drbg.c',
        'library/debug.c',
        'library/des.c',
        'library/dhm.c',
        'library/ecdh.c',
        'library/ecdsa.c',
        'library/ecjpake.c',
        'library/ecp.c',
        'library/ecp_curves.c',
        'library/entropy.c',
        'library/entropy_poll.c',
        'library/error.c',
        'library/gcm.c',
        'library/godot_core_mbedtls_platform.c',
        'library/havege.c',
        'library/hkdf.c',
        'library/hmac_drbg.c',
        'library/md.c',
        'library/md2.c',
        'library/md4.c',
        'library/md5.c',
        'library/md_wrap.c',
        'library/memory_buffer_alloc.c',
        'library/net_sockets.c',
        'library/nist_kw.c',
        'library/oid.c',
        'library/padlock.c',
        'library/pem.c',
        'library/pk.c',
        'library/pk_wrap.c',
        'library/pkcs11.c',
        'library/pkcs12.c',
        'library/pkcs5.c',
        'library/pkparse.c',
        'library/pkwrite.c',
        'library/platform.c',
        'library/platform_util.c',
        'library/poly1305.c',
        'library/ripemd160.c',
        'library/rsa.c',
        'library/rsa_internal.c',
        'library/sha1.c',
        'library/sha256.c',
        'library/sha512.c',
        'library/ssl_cache.c',
        'library/ssl_ciphersuites.c',
        'library/ssl_cli.c',
        'library/ssl_cookie.c',
        'library/ssl_srv.c',
        'library/ssl_ticket.c',
        'library/ssl_tls.c',
        'library/threading.c',
        'library/timing.c',
        'library/version.c',
        'library/version_features.c',
        'library/x509.c',
        'library/x509_create.c',
        'library/x509_crl.c',
        'library/x509_crt.c',
        'library/x509_csr.c',
        'library/x509write_crt.c',
        'library/x509write_csr.c',
        'library/xtea.c'
    ])

    _lib_mbedtls = static_library('mbedtls', _mbedtls_srcs, include_directories: _mbedtls_incdirs, dependencies: _mbedtls_deps, build_by_default: false)
    DEP_MBEDTLS = declare_dependency(link_with: _lib_mbedtls, include_directories: _mbedtls_incdirs)
else
    DEP_MBEDTLS = [
        dependency('mbedtls'),
        dependency('mbedcrypto'),
        dependency('mbedx509')
    ]
endif