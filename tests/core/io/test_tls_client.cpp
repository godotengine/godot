/**************************************************************************/
/*  test_tls_client.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_tls_client)

#include "core/io/http_client.h"
#include "core/os/os.h"

#include "modules/modules_enabled.gen.h" // For mbedtls.

namespace TestHTTPClient {

#if defined(TEST_NETWORK_ENABLED) && (defined(MODULE_MBEDTLS_ENABLED) || defined(WEB_ENABLED))

String untrusted_root_badssl_com_crt = R"(
-----BEGIN CERTIFICATE-----
MIIGfjCCBGagAwIBAgIJAJeg/PrX5Sj9MA0GCSqGSIb3DQEBCwUAMIGBMQswCQYD
VQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTEWMBQGA1UEBwwNU2FuIEZyYW5j
aXNjbzEPMA0GA1UECgwGQmFkU1NMMTQwMgYDVQQDDCtCYWRTU0wgVW50cnVzdGVk
IFJvb3QgQ2VydGlmaWNhdGUgQXV0aG9yaXR5MB4XDTE2MDcwNzA2MzEzNVoXDTM2
MDcwMjA2MzEzNVowgYExCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlh
MRYwFAYDVQQHDA1TYW4gRnJhbmNpc2NvMQ8wDQYDVQQKDAZCYWRTU0wxNDAyBgNV
BAMMK0JhZFNTTCBVbnRydXN0ZWQgUm9vdCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkw
ggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDKQtPMhEH073gis/HISWAi
bOEpCtOsatA3JmeVbaWal8O/5ZO5GAn9dFVsGn0CXAHR6eUKYDAFJLa/3AhjBvWa
tnQLoXaYlCvBjodjLEaFi8ckcJHrAYG9qZqioRQ16Yr8wUTkbgZf+er/Z55zi1yn
CnhWth7kekvrwVDGP1rApeLqbhYCSLeZf5W/zsjLlvJni9OrU7U3a9msvz8mcCOX
fJX9e3VbkD/uonIbK2SvmAGMaOj/1k0dASkZtMws0Bk7m1pTQL+qXDM/h3BQZJa5
DwTcATaa/Qnk6YHbj/MaS5nzCSmR0Xmvs/3CulQYiZJ3kypns1KdqlGuwkfiCCgD
yWJy7NE9qdj6xxLdqzne2DCyuPrjFPS0mmYimpykgbPnirEPBF1LW3GJc9yfhVXE
Cc8OY8lWzxazDNNbeSRDpAGbBeGSQXGjAbliFJxwLyGzZ+cG+G8lc+zSvWjQu4Xp
GJ+dOREhQhl+9U8oyPX34gfKo63muSgo539hGylqgQyzj+SX8OgK1FXXb2LS1gxt
VIR5Qc4MmiEG2LKwPwfU8Yi+t5TYjGh8gaFv6NnksoX4hU42gP5KvjYggDpR+NSN
CGQSWHfZASAYDpxjrOo+rk4xnO+sbuuMk7gORsrl+jgRT8F2VqoR9Z3CEdQxcCjR
5FsfTymZCk3GfIbWKkaeLQIDAQABo4H2MIHzMB0GA1UdDgQWBBRvx4NzSbWnY/91
3m1u/u37l6MsADCBtgYDVR0jBIGuMIGrgBRvx4NzSbWnY/913m1u/u37l6MsAKGB
h6SBhDCBgTELMAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExFjAUBgNV
BAcMDVNhbiBGcmFuY2lzY28xDzANBgNVBAoMBkJhZFNTTDE0MDIGA1UEAwwrQmFk
U1NMIFVudHJ1c3RlZCBSb290IENlcnRpZmljYXRlIEF1dGhvcml0eYIJAJeg/PrX
5Sj9MAwGA1UdEwQFMAMBAf8wCwYDVR0PBAQDAgEGMA0GCSqGSIb3DQEBCwUAA4IC
AQBQU9U8+jTRT6H9AIFm6y50tXTg/ySxRNmeP1Ey9Zf4jUE6yr3Q8xBv9gTFLiY1
qW2qfkDSmXVdBkl/OU3+xb5QOG5hW7wVolWQyKREV5EvUZXZxoH7LVEMdkCsRJDK
wYEKnEErFls5WPXY3bOglBOQqAIiuLQ0f77a2HXULDdQTn5SueW/vrA4RJEKuWxU
iD9XPnVZ9tPtky2Du7wcL9qhgTddpS/NgAuLO4PXh2TQ0EMCll5reZ5AEr0NSLDF
c/koDv/EZqB7VYhcPzr1bhQgbv1dl9NZU0dWKIMkRE/T7vZ97I3aPZqIapC2ulrf
KrlqjXidwrGFg8xbiGYQHPx3tHPZxoM5WG2voI6G3s1/iD+B4V6lUEvivd3f6tq7
d1V/3q1sL5DNv7TvaKGsq8g5un0TAkqaewJQ5fXLigF/yYu5a24/GUD783MdAPFv
gWz8F81evOyRfpf9CAqIswMF+T6Dwv3aw5L9hSniMrblkg+ai0K22JfoBcGOzMtB
Ke/Ps2Za56dTRoY/a4r62hrcGxufXd0mTdPaJLw3sJeHYjLxVAYWQq4QKJQWDgTS
dAEWyN2WXaBFPx5c8KIW95Eu8ShWE00VVC3oA4emoZ2nrzBXLrUScifY6VaYYkkR
2O2tSqU8Ri3XRdgpNPDWp8ZL49KhYGYo3R/k98gnMHiY5g==
-----END CERTIFICATE-----
)";

String cloudflare_root_ca_crt = R"(
-----BEGIN CERTIFICATE-----
MIID5jCCA42gAwIBAgIQAe7mqrtSHV4U/DFf2ZhWkDAKBggqhkjOPQQDAjBRMQsw
CQYDVQQGEwJVUzEYMBYGA1UECgwPU1NMIENvcnBvcmF0aW9uMSgwJgYDVQQDDB9D
bG91ZGZsYXJlIFRMUyBJc3N1aW5nIEVDQyBDQSAzMB4XDTI2MDkyNjIyNDkxMVoX
DTI2MTIyNTIyNTYzNVowFjEUMBIGA1UEAwwLZXhhbXBsZS5jb20wWTATBgcqhkjO
PQIBBggqhkjOPQMBBwNCAARqadExWzxTJ4S3LSzKIzAAeyF09q0Qt0F0XmLiZiaJ
SD0SQPhVK36pUKlM55TUstJnrU7f+7elE0kZESKzzcbto4ICgDCCAnwwDAYDVR0T
AQH/BAIwADAfBgNVHSMEGDAWgBSDA/3n9vVKTRVB9O0iFtMyCj7KZjBsBggrBgEF
BQcBAQRgMF4wOQYIKwYBBQUHMAKGLWh0dHA6Ly9pLmNmLWkuc3NsLmNvbS9DbG91
ZGZsYXJlLVRMUy1JLUUzLmNlcjAhBggrBgEFBQcwAYYVaHR0cDovL28uY2YtaS5z
c2wuY29tMCUGA1UdEQQeMByCC2V4YW1wbGUuY29tgg0qLmV4YW1wbGUuY29tMCMG
A1UdIAQcMBowCAYGZ4EMAQIBMA4GDCsGAQQBgqkwAQMBATATBgNVHSUEDDAKBggr
BgEFBQcDATBTBgNVHR8ETDBKMEigRqBEhkJodHRwOi8vYy5jZi1pLnNzbC5jb20v
YWU4MDFlZDFjNTViYjU3OWQ3OTIwOGIwZDc3MmFjZmI4Y2MzYTIwOC5jcmwwDgYD
VR0PAQH/BAQDAgeAMA8GCSsGAQQBgtpLLAQCBQAwggEEBgorBgEEAdZ5AgQCBIH1
BIHyAPAAdgCUTkOH+uzB74HzGSQmqBhlAcfTXzgCAT9yZ31VNy4Z2AAAAaDf8a1i
AAAEAwBHMEUCIQCuCUahRazwOcqj3A77y8eXdQBpLlGCjv9D/xvXd3X7hAIgbc0K
MS+MsANY5WLGpHgTk3Uxk0g1VWTQB78vYRMga1EAdgDLOPcViXyEoURfW8Hd+8lu
8ppZzUcKaQWFsMsUwxRY5wAAAaDf8a1+AAAEAwBHMEUCIB2YN/lZP6eSBMtV0BTr
GD7KSyKVu/i68Zza97li96FcAiEA1mtvTnM8zbNKj6ArUpWmgTRURDDgOC4R1C2W
5cSK6fIwCgYIKoZIzj0EAwIDRwAwRAIgNGlXxWLTTqn5+XQOkP06fTgaEbjDMAHU
ILmIapx8i5sCIAeTI/xpafGZNB9HwNfEeU05yh+aSv95pD8SNtIAWNr0
-----END CERTIFICATE-----
)";

String ssl_com_root_ca_crt = R"(
-----BEGIN CERTIFICATE-----
MIICOjCCAcCgAwIBAgIQFAP1q/s3ixdAW+JDsqXRxDAKBggqhkjOPQQDAzBOMQsw
CQYDVQQGEwJVUzEYMBYGA1UECgwPU1NMIENvcnBvcmF0aW9uMSUwIwYDVQQDDBxT
U0wuY29tIFRMUyBFQ0MgUm9vdCBDQSAyMDIyMB4XDTIyMDgyNTE2MzM0OFoXDTQ2
MDgxOTE2MzM0N1owTjELMAkGA1UEBhMCVVMxGDAWBgNVBAoMD1NTTCBDb3Jwb3Jh
dGlvbjElMCMGA1UEAwwcU1NMLmNvbSBUTFMgRUNDIFJvb3QgQ0EgMjAyMjB2MBAG
ByqGSM49AgEGBSuBBAAiA2IABEUpNXP6wrgjzhR9qLFNoFs27iosU8NgCTWyJGYm
acCzldZdkkAZDsalE3D07xJRKF3nzL35PIXBz5SQySvOkkJYWWf9lCcQZIxPBLFN
SeR7T5v15wj4A4j3p8OSSxlUgaNjMGEwDwYDVR0TAQH/BAUwAwEB/zAfBgNVHSME
GDAWgBSJjy+j6CugFFR781a4Jl9nOAuc0DAdBgNVHQ4EFgQUiY8vo+groBRUe/NW
uCZfZzgLnNAwDgYDVR0PAQH/BAQDAgGGMAoGCCqGSM49BAMDA2gAMGUCMFXjIlbp
15IkWE8elDIPDAI2wv2sdDJO4fscgIijzPvX6yv/N33w7deedWo1dlJF4AIxAMeN
b0Igj762TVntd00pxCAgRWSGOlDGxK0tk/UYfXLtqc/ErFc2KAhl3zx5Zn6g6g==
-----END CERTIFICATE-----
)";

bool poll_client(Ref<HTTPClient> &client) {
	switch (client->get_status()) {
		case HTTPClient::Status::STATUS_CANT_RESOLVE:
		case HTTPClient::Status::STATUS_CANT_CONNECT:
		case HTTPClient::Status::STATUS_CONNECTION_ERROR:
		case HTTPClient::Status::STATUS_TLS_HANDSHAKE_ERROR:
		case HTTPClient::Status::STATUS_CONNECTED:
			return false;
		default:
			break;
	}

	client->poll();
	return true;
}

// Validate that a normal connection works.
TEST_CASE("[TLSClient] https://example.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://www.example.com";
	int port = 443;
	Ref<TLSOptions> tls_options;

	// Connect to host.
	Error err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_CONNECTED, "Expected STATUS_CONNECTED for a SSL connection");
}

// Validate that a revoked intermediate CA is rejected.
TEST_CASE("[TLSClient] https://dv-g3b-emudhra.we-have-a-problem.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://dv-g3b-emudhra.we-have-a-problem.com";
	int port = 443;
	Ref<TLSOptions> tls_options;

	// Connect to host.
	Error err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_TLS_HANDSHAKE_ERROR, "Expected STATUS_TLS_HANDSHAKE_ERROR for a SSL connection");
}

// Validate that an untrusted CA is rejected.
TEST_CASE("[TLSClient] https://untrusted-root.badssl.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://untrusted-root.badssl.com";
	int port = 443;
	Ref<TLSOptions> tls_options;

	// Connect to host.
	Error err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_TLS_HANDSHAKE_ERROR, "Expected STATUS_TLS_HANDSHAKE_ERROR for a SSL connection");
}

// Validate that the unsafe client option allows a connection.
TEST_CASE("[TLSClient] unsafe https://untrusted-root.badssl.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://untrusted-root.badssl.com";
	int port = 443;
	Ref<TLSOptions> tls_options = TLSOptions::client_unsafe(nullptr);

	// Connect to host.
	Error err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_CONNECTED, "Expected STATUS_CONNECTED for a unsafe SSL connection");
}

// Validate that an unrelated CA does not allow a successful connection.
TEST_CASE("[TLSClient] untrusted CA https://example.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://example.com";
	int port = 443;
	Error err;

	// Use an unrelated root CA for the target site.
	Ref<X509Certificate> ca_x509 = X509Certificate::create();
	err = ca_x509->load_from_memory((const uint8_t *)untrusted_root_badssl_com_crt.utf8().get_data(), untrusted_root_badssl_com_crt.size());
	CHECK_MESSAGE(err == OK, "Expected OK for certificate load");

	Ref<TLSOptions> tls_options = TLSOptions::client(ca_x509);

	// Connect to host.
	err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_TLS_HANDSHAKE_ERROR, "Expected STATUS_TLS_HANDSHAKE_ERROR for a SSL connection");
}

// Validate an explicitly specified trusted CA allows connections.
TEST_CASE("[TLSClient] trusted root https://example.com") {
	Ref<HTTPClient> client = HTTPClient::create();
	String host = "https://example.com";
	int port = 443;
	Error err;

	Ref<X509Certificate> ca_x509 = X509Certificate::create();
	err = ca_x509->load_from_memory((const uint8_t *)ssl_com_root_ca_crt.utf8().get_data(), ssl_com_root_ca_crt.size());
	CHECK_MESSAGE(err == OK, "Expected OK for certificate load");

	Ref<TLSOptions> tls_options = TLSOptions::client(ca_x509);

	// Connect to host.
	err = client->connect_to_host(host, port, tls_options);
	CHECK_MESSAGE(err == OK, "Expected OK for successful connection");

	while (poll_client(client)) {
		::OS::get_singleton()->delay_usec(100);
	}

	CHECK_MESSAGE(client->get_status() == HTTPClient::Status::STATUS_CONNECTED, "Expected STATUS_CONNECTED for a SSL connection");
}
#endif // defined(TEST_NETWORK_ENABLED) && (defined(MODULE_MBEDTLS_ENABLED) || defined(WEB_ENABLED))

} // namespace TestHTTPClient
