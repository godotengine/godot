/**************************************************************************/
/*  test_crypto_mbedtls.h                                                 */
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

#pragma once

#include "core/crypto/hashing_context.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestCryptoMbedTLS {

// HMACContext
void hmac_digest_test(HashingContext::HashType ht, String expected_hex);

TEST_CASE("[CryptoMbedTLS] HMAC digest") {
	// SHA-256
	hmac_digest_test(HashingContext::HashType::HASH_SHA256, "fe442023f8a7d36a810e1e7cd8a8e2816457f350a008fbf638296afa12085e59");

	// SHA-1
	hmac_digest_test(HashingContext::HashType::HASH_SHA1, "a0ac4cd68a2f4812c355983d94e8d025afe7dddf");
}

void hmac_context_digest_test(HashingContext::HashType ht, String expected_hex);

TEST_CASE("[HMACContext] HMAC digest") {
	// SHA-256
	hmac_context_digest_test(HashingContext::HashType::HASH_SHA256, "fe442023f8a7d36a810e1e7cd8a8e2816457f350a008fbf638296afa12085e59");

	// SHA-1
	hmac_context_digest_test(HashingContext::HashType::HASH_SHA1, "a0ac4cd68a2f4812c355983d94e8d025afe7dddf");
}

// CryptoKey
void crypto_key_public_only_test(const String &p_key_path, bool public_only);

TEST_CASE("[Crypto] CryptoKey is_public_only") {
	crypto_key_public_only_test(TestUtils::get_data_path("crypto/in.key"), false);
	crypto_key_public_only_test(TestUtils::get_data_path("crypto/in.pub"), true);
	crypto_key_public_only_test(TestUtils::get_data_path("crypto/in.prime256v1.key"), false);
	crypto_key_public_only_test(TestUtils::get_data_path("crypto/in.prime256v1.pub"), true);
}

void crypto_key_save_test(const String &p_in_path, const String &p_out_path, bool public_only);

TEST_CASE("[Crypto] CryptoKey save RSA") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.key");
	const String out_priv_path = TestUtils::get_data_path("crypto/out.key");
	crypto_key_save_test(in_priv_path, out_priv_path, false);

	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");
	const String out_pub_path = TestUtils::get_data_path("crypto/out.pub");
	crypto_key_save_test(in_pub_path, out_pub_path, true);
}

TEST_CASE("[Crypto] CryptoKey save EC") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.prime256v1.key");
	const String out_priv_path = TestUtils::get_data_path("crypto/out.prime256v1.key");
	crypto_key_save_test(in_priv_path, out_priv_path, false);

	const String in_pub_path = TestUtils::get_data_path("crypto/in.prime256v1.pub");
	const String out_pub_path = TestUtils::get_data_path("crypto/out.prime256v1.pub");
	crypto_key_save_test(in_pub_path, out_pub_path, true);
}

void crypto_key_save_public_only_test(const String &p_in_priv_path, const String &p_in_pub_path, const String &p_out_path);

TEST_CASE("[Crypto] CryptoKey save public_only RSA") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");
	const String out_path = TestUtils::get_data_path("crypto/out_public_only.pub");
	crypto_key_save_public_only_test(in_priv_path, in_pub_path, out_path);
}

TEST_CASE("[Crypto] CryptoKey save public_only EC") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.prime256v1.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.prime256v1.pub");
	const String out_path = TestUtils::get_data_path("crypto/out_public_only.prime256v1.pub");
	crypto_key_save_public_only_test(in_priv_path, in_pub_path, out_path);
}

void crypto_key_generate_save_load(const String &p_out_priv_path, const String &p_out_pub_path);

TEST_CASE("[Crypto] CryptoKey RSA generate/save/load") {
	const String out_priv_path = TestUtils::get_data_path("crypto/out_generated.key");
	const String out_pub_path = TestUtils::get_data_path("crypto/out_generated.pub");
	crypto_key_generate_save_load(out_priv_path, out_pub_path);
}

// Crypto
void crypto_sign_verify_test(const String &p_in_priv_path, const String &p_in_pub_path, HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash);

void crypto_generate_sign_verify_test(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash);

TEST_CASE("[Crypto] RSA generate/sign/verify") {
	const String msg("Some message to sign.");
	crypto_generate_sign_verify_test(HashingContext::HASH_MD5, msg.md5_buffer());
	crypto_generate_sign_verify_test(HashingContext::HASH_SHA1, msg.sha1_buffer());
	crypto_generate_sign_verify_test(HashingContext::HASH_SHA256, msg.sha256_buffer());
}

TEST_CASE("[Crypto] RSA load/sign/verify") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");
	const String msg("Some message to sign.");
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_MD5, msg.md5_buffer());
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_SHA1, msg.sha1_buffer());
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer());
}

TEST_CASE("[Crypto] EC load/sign/verify") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.prime256v1.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.prime256v1.pub");
	const String msg("Some message to sign.");
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_MD5, msg.md5_buffer());
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_SHA1, msg.sha1_buffer());
	crypto_sign_verify_test(in_priv_path, in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer());
}

void crypto_verify_test(const String &p_in_pub_path, HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, const Vector<uint8_t> &p_signature, bool p_expected);

TEST_CASE("[Crypto] RSA verify well-known signatures") {
	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");
	const String msg("This is a message to verify");

	const char *wk_sig_md5 = "7524b3be41d7c40e4e4f4e3c3a072e5cfe4c66f6f87fa8f11e77e72941a3c601a04281ec2200fadc5d7362807679099dc374d5d8c02f1b6f47a1a3dd2f8f5808b2b7265c04c0da01a144ddc06379994209d5e5e731719e73a90741b3435706b43f0ff28bf707a4d7e17e610ef44dbfe303b52e74118c3a6696d09bf94b92673f8a7b49a391a7c1bdfbd5a8b7e3fad747517d9fe25720072217a29c1ab06c74b97a7464574dc509dd896df129d051823c6d25a1b31a16bdad8be84e8557b31193efe55c5f4d28ef50153c5129ac7d42ea560eb2ac10874c565e0a8fc61067cfc178bad824afbbbb8a659f7ad50a35728d08c30b2ec52eb9627704f6bc4620d0ad1d3a6f28017350f2e470c1a7dd2723ef92732aa9b226d4df241231b0a2c0ac7867bee2f352f853a2812a63b7ee34eec9abb76213d306bf9726dfce483a6f9ade18dee017a1373f26b45ceb8c5754f3e0f565db1d2d2c49b286dd2aef254e05c0041b78347586020492c3199b0c865b5341c95e465fe7bcd83d0ea44bd750f863597fa2ce970ab165a97d3e2978bbbba2eef9135dbe86c53d223d4303a19353c8f9a4f1bed0ac69f53e09dad836f38cdca5d6fe46407de95413bf5e9e551ea84f8ca097b36932432eefe49be707a3999e0f4064933a2a39552b16b2466ee64d72bec3c52337f3326c2acdeee5f3ada69cb11060ab5c97eace333d29d6fcbdf06b";
	const char *wk_sig_sha1 = "9b81157dc6dae7bdd6dfbc3e77bc3714119a9ddef5f73a1d765c9478629eea937d23c450c496e1220fcff32eaf8aadc070739ced0f22691c2b88aafc694f129773d1fd9fa5fc62d7d1c5b9feb7701ea6c451a6b51f5d980a8f13e73eea4d3ebf04da6b5d3c047413d8985fcc6dfccc77714e33e8fefe7a04e98b21c41b1bfe2f78f902ab6acf9fb50db4ff25e79374ae212537099482088b46a4769ca511c9880370698a2e69a3b6427d67e79afb4d830f2e4f85ba380663c7fe188f878f57420ab5584b01cede502dd015e642fe533d62cefb267aea77d2f55e4fe6d93b1dab8dfe2b5c70135f993b41ea1b7a3f627f0293a035bb3ad08b3118f8398a55accad7616a89645f929fafbfad7c22a4e449112b57792dd3480252865c12d1348abcdfe8a8096e254f516d6ffd5d6d40c9ed66c8a82ad2cca2dc2526aab06b1436db9984aea624051e72f949dd163a4a364902930de44deab84e3b5048c5fe67b97dcc858f3334895d68c35dd7821446676ae240a846362878d490b9385437ed0a75ad4f1cc30e4553337629e4b31cede5dfd93aa9ee5fd98d2a2875084212c9264c751350e28b32630befebc1bfb04d46a82d50f9465e76d535c9159b9a902ac3cc437c46928d6839504e66154ee711c9000d99b928753216622cb6ee52ff93e972a8069676953e8d0d3b7306c603ee8e483a94a8733ee9debe020832b14f433307";
	const char *wk_sig_sha256 = "8bd353af89d0c1f3ce39541f2b0e80f40e1e361fbeb754db4b4a532df4821c87f93e13abe397a3e661fb69f0423c2a03bdf7d20903116122315d53baa8885fe26fd4e3f85a83dd3d5d3d526fec9039487e67938b141c7eec23cfa09fad03eeb3e4772b78e768ada24ae79126595b816b748201a6549bf4d967f6c88c830eb77b4afc23990a2b0ab0c82afd9a93d2ee4131d17382cd16dfa228e9a0fbcb22383da347bcbd1baff1efa7ac93590a910977784b37dd2824e600481fc986012d5151a502508e621568283c084c071154d4048ece461993c0831ed5979a100d76659482c0714ceb02d1e2d76d0bc8540d92409ab73bb0688d633c2ff1881f50302ac134b87f0031ed2a67412a9e09617a2b4f6c7b6dc5d82d1cba1b2634d8e70caf786fe91de11b8a57d2d506b7b84e9317687b2bf2c8418c9c7e555a7254cec2b6a1aaf7e614d94b0c170eddf7c981de92a87799c949d4091f99f1c3faee6df86fb701328c990892cec290e810ace2ea64b0f93d01888a6defb09212137ed36746b7149ff81bddaf58012e62ddc49b7f4e7d87b450df2be005f7b5f4c6d06c5e640b4f017f9091a84a91e8ddac6f09ef7349c3ec9959e1b7648d74944b4dea1773f52fbca7436f3704cb5906d9e03a59ff3e40c66eac3dcd7990bf4103dfe10c896dd6897ca3916cd0fa61ece22b54b7c752f03b267b975e72ac187d5ed007c3b26b";

	// Valid
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, msg.md5_buffer(), String(wk_sig_md5).hex_decode(), true);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA1, msg.sha1_buffer(), String(wk_sig_sha1).hex_decode(), true);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer(), String(wk_sig_sha256).hex_decode(), true);

	// Incorrect hash
	ERR_PRINT_OFF;
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, msg.sha256_buffer(), String(wk_sig_sha256).hex_decode(), false);
	ERR_PRINT_ON;

	// Incorrect signature
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer(), String(wk_sig_sha256).replace("a", "b").hex_decode(), false);

	// Tampered message
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, (msg + "!").md5_buffer(), String(wk_sig_md5).hex_decode(), false);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA1, (msg + "!").sha1_buffer(), String(wk_sig_sha1).hex_decode(), false);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, (msg + "!").sha256_buffer(), String(wk_sig_sha256).hex_decode(), false);
}

TEST_CASE("[Crypto] EC verify well-known signatures") {
	const String in_pub_path = TestUtils::get_data_path("crypto/in.prime256v1.pub");
	const String msg("This is a message to verify");

	const char *wk_sig_md5 = "3044022047f4782bc7319d7562eb9f20e9b8a924fe9a58f0fac0710d609bbe448f368188022035d5c332388f577c5ef7c829d1c58cca817a6ade4406c931a7cfafc367895c8a";
	const char *wk_sig_sha1 = "3044022023f1e922dc51e88960dd133bd192a405fcb7f3b71fd59d49b9943d2037be1c140220053e6ad1230b422556db3bd8dbaa05358399db24f5ad4a044be4c53dc6b2a491";
	const char *wk_sig_sha256 = "3046022100a1b768db040dea7a79c81f10fd0b3d34d53fe941cdd46ee1a382ce27310a9826022100c038ce88b59ff6f9fbe7837f0ff5e5079adf7995ac2d86e2baf03bfa1cda5cd4";

	// Valid
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, msg.md5_buffer(), String(wk_sig_md5).hex_decode(), true);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA1, msg.sha1_buffer(), String(wk_sig_sha1).hex_decode(), true);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer(), String(wk_sig_sha256).hex_decode(), true);

	// Incorrect hash
	ERR_PRINT_OFF;
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, msg.sha256_buffer(), String(wk_sig_sha256).hex_decode(), false);
	ERR_PRINT_ON;

	// Incorrect signature
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, msg.sha256_buffer(), String(wk_sig_sha256).replace("a", "b").hex_decode(), false);

	// Tampered message
	crypto_verify_test(in_pub_path, HashingContext::HASH_MD5, (msg + "!").md5_buffer(), String(wk_sig_md5).hex_decode(), false);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA1, (msg + "!").sha1_buffer(), String(wk_sig_sha1).hex_decode(), false);
	crypto_verify_test(in_pub_path, HashingContext::HASH_SHA256, (msg + "!").sha256_buffer(), String(wk_sig_sha256).hex_decode(), false);
}

void crypto_encrypt_decrypt_test(const String &p_in_priv_path, const String &p_in_pub_path, const Vector<uint8_t> &p_message);

TEST_CASE("[Crypto] Crypto encrypt/decrypt") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");
	crypto_encrypt_decrypt_test(in_priv_path, in_pub_path, String("Some message").to_utf8_buffer());
	crypto_encrypt_decrypt_test(in_priv_path, in_pub_path, String("Some very long message").repeat(16).to_utf8_buffer());
}

void crypto_decrypt_test(const String &p_in_priv_path, const Vector<uint8_t> &p_ciphertext, const Vector<uint8_t> &p_message, bool p_valid, bool p_try_pub_key = false);

TEST_CASE("[Crypto] Crypto decrypt well-known") {
	const String in_priv_path = TestUtils::get_data_path("crypto/in.key");
	const String in_pub_path = TestUtils::get_data_path("crypto/in.pub");

	const char *wk_ciphertext = "85ca5bf60eac80831b9927c5dba1a4dcf253a5f019e56b71262a6b5476c8294418c59ad30d2815fc26f1ce1e1860637ade746c262c44443cef84467babbba456c0dec7093481fc56795ebe170372fccb39744590e32201752981e39b81d24915af90240632878b96617103eff46c999ee5b4f9aa009eed6a1f9a8be4fde142c754e8f71cddac8405329d8ae6d9f98eb23547b60c416f01cd9199bbfec5e6fe076b032d8900ac7c6b14112ceebfef67962814b9090c49d0ca2fe14adf40242759e374c7944a413743fcaf655ede33d57cfb93cf2c33a292894b6450f34faf1b2772813c8007dd19d2fceaa1c9a8632dba021ecd80305a26b8b0ecfb3c26edca2d81bf48381db0b3f7eb17e6513b4e691d1965ab2b1b7d0dff1c6f717cd498233367081bb989ed8e1836d7229ec8323d7cd8d8fd68ee434c8e703b1215acd10dbd5ca7d52599503d1775f53d726396ac7bed171810de498ff5364bcb2f09c6b75775d0974a20563c51530ca41addabe99f46076ff11d037c23be13bb195e5416c73a3471d7ffec3f0bb65ce4b0674be8e3f4c7da2fe89a5a7d2994cefd936a0a4443d38d8490353640598725da7e8ab85b038ec81d558d56df0d4fa733ddc8972104c0ded969908d3d9833cbb4fd34e528368fe7b2b988a222707671af11972e98e6ec7f38d9847843c6adfb46ff66c888a037c4affd6d7c7a6e6437b53d2fdfdb";

	const char *wk_empty_ciphertext = "a63f7948a1c5ac185c893b1e783571b31b34678462ea413fcbc3cb531b575b3c49cc7547dcfcaaf37015785ad917be1b08cd25adb1ff1a448bf6b76eadf57b26236644e6aadb60a04e41da7add9238c5c211af8a3dacaf64fba47125314189defd9e305e7264d156f3a25972acc2195d33257fa427d519d349f21319fdd6d8a6d33a64ee174a61af92a999ae4fc9034286b307c6872c6e8bd809595bb586958e3362b52460b6bb83edcc94c38a90c799ce4905f0f2efade25b17ac1d2795b15eedc99ea1212ef9922b629f1ca3e0288bf493be6bed4066cb29f75cd37af9eca7c012514907970ba59ed803759bb2d4235ed9fdce12183b703246625ee221c51c9a0bc341a7bb3d45720f6072fffa2d7233bd94ff79fbdd90138137cb00d069f82be63c5fe20df7e23bbcd7d77f1a5a71b5936557a76703c3aa4fdbd4950d3aca68db28d117ca26a5ff0cda4b1d4841a8814ce93f097ac2c94381f77a29de4dadbca7daefaebbd03097f74b36644453bfce5c7a3aee2d99ca552389d9c8895d2933b874201b15fa63af37ee722e8a0d848665f85a00b234b2927b06e12861a539d9e20549af07bfa9db68775bf821007363015b09e2d8ed3b9c2fbf431c44619152159106368a9c04c6e8c0eb5295f11e1dcebaf5691a70dbb9c520bd186e13b6cbaf0381377a7f367eebd2c45c636e9cc1116729e23614001b92ebc26a899226";

	const Vector<uint8_t> message = String("This is a secret message").to_utf8_buffer();
	crypto_decrypt_test(in_priv_path, String(wk_ciphertext).hex_decode(), message, true);
	crypto_decrypt_test(in_priv_path, String(wk_empty_ciphertext).hex_decode(), Vector<uint8_t>(), true);

	// Invalid ciphertext
	ERR_PRINT_OFF;
	crypto_decrypt_test(in_priv_path, String(wk_ciphertext).replace("a", "b").hex_decode(), message, false);
	ERR_PRINT_ON;

	// Can't decrypt with public key
	ERR_PRINT_OFF;
	crypto_decrypt_test(in_pub_path, String(wk_ciphertext).hex_decode(), message, false, true);
	ERR_PRINT_ON;
}

} // namespace TestCryptoMbedTLS
