/**************************************************************************/
/*  ApkSignerUtil.kt                                                      */
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

@file:JvmName("ApkSignerUtil")

package org.godotengine.editor.utils

import android.util.Log
import com.android.apksig.ApkSigner
import com.android.apksig.ApkVerifier
import org.bouncycastle.jce.provider.BouncyCastleProvider
import org.godotengine.godot.error.Error
import org.godotengine.godot.io.file.FileAccessHandler
import java.io.File
import java.security.KeyStore
import java.security.PrivateKey
import java.security.Security
import java.security.cert.X509Certificate
import java.util.ArrayList


/**
 * Contains utilities methods to sign and verify Android apks using apksigner
 */
private const val TAG = "ApkSignerUtil"

private const val DEFAULT_KEYSTORE_TYPE = "PKCS12"

/**
 * Validates that the correct version of the BouncyCastleProvider is added.
 */
private fun validateBouncyCastleProvider() {
	val bcProvider = Security.getProvider(BouncyCastleProvider.PROVIDER_NAME)
	if (bcProvider !is BouncyCastleProvider) {
		Log.v(TAG, "Removing BouncyCastleProvider $bcProvider (${bcProvider::class.java.name})")
		Security.removeProvider(BouncyCastleProvider.PROVIDER_NAME)

		val updatedBcProvider = BouncyCastleProvider()
		val addResult = Security.addProvider(updatedBcProvider)
		if (addResult == -1) {
			Log.e(TAG, "Unable to add BouncyCastleProvider ${updatedBcProvider::class.java.name}")
		} else {
			Log.v(TAG, "Updated BouncyCastleProvider to $updatedBcProvider (${updatedBcProvider::class.java.name})")
		}
	}
}

/**
 * Verifies the given Android apk
 *
 * @return true if verification was successful, false otherwise.
 */
internal fun verifyApk(fileAccessHandler: FileAccessHandler, apkPath: String): Error {
	if (!fileAccessHandler.fileExists(apkPath)) {
		Log.e(TAG, "Unable to access apk $apkPath")
		return Error.ERR_FILE_NOT_FOUND
	}

	try {
		val apkVerifier = ApkVerifier.Builder(File(apkPath)).build()

		Log.v(TAG, "Verifying apk $apkPath")
		val result = apkVerifier.verify()

		Log.v(TAG, "Verification result: ${result.isVerified}")
		return if (result.isVerified) {
			Error.OK
		} else {
			Error.FAILED
		}
	} catch (e: Exception) {
		Log.e(TAG, "Error occurred during verification for $apkPath", e)
		return Error.ERR_INVALID_DATA
	}
}

/**
 * Signs the given Android apk
 *
 * @return true if signing is successful, false otherwise.
 */
internal fun signApk(fileAccessHandler: FileAccessHandler,
					 inputPath: String,
					 outputPath: String,
					 keystorePath: String,
					 keystoreUser: String,
					 keystorePassword: String,
					 keystoreType: String = DEFAULT_KEYSTORE_TYPE): Error {
	if (!fileAccessHandler.fileExists(inputPath)) {
		Log.e(TAG, "Unable to access input path $inputPath")
		return Error.ERR_FILE_NOT_FOUND
	}

	val tmpOutputPath = if (outputPath != inputPath) { outputPath } else { "$outputPath.signed" }
	if (!fileAccessHandler.canAccess(tmpOutputPath)) {
		Log.e(TAG, "Unable to access output path $tmpOutputPath")
		return Error.ERR_FILE_NO_PERMISSION
	}

	if (!fileAccessHandler.fileExists(keystorePath) ||
		keystoreUser.isBlank() ||
		keystorePassword.isBlank()) {
		Log.e(TAG, "Invalid keystore credentials")
		return Error.ERR_INVALID_PARAMETER
	}

	validateBouncyCastleProvider()

	// 1. Obtain a KeyStore implementation
	val keyStore = KeyStore.getInstance(keystoreType)

	// 2. Load the keystore
	val inputStream = fileAccessHandler.getInputStream(keystorePath)
	if (inputStream == null) {
		Log.e(TAG, "Unable to retrieve input stream from $keystorePath")
		return Error.ERR_FILE_CANT_READ
	}
	try {
		inputStream.use {
			Log.v(TAG, "Loading keystore $keystorePath with type $keystoreType")
			keyStore.load(it, keystorePassword.toCharArray())
		}
	} catch (e: Exception) {
		Log.e(TAG, "Unable to load the keystore from $keystorePath", e)
		return Error.ERR_FILE_CANT_READ
	}

	// 3. Load the private key and cert chain from the keystore
	if (!keyStore.isKeyEntry(keystoreUser)) {
		Log.e(TAG, "Key alias $keystoreUser is invalid")
		return Error.ERR_INVALID_PARAMETER
	}

	val keyStoreKey = try {
		keyStore.getKey(keystoreUser, keystorePassword.toCharArray())
	} catch (e: Exception) {
		Log.e(TAG, "Unable to recover keystore alias $keystoreUser")
		return Error.ERR_CANT_ACQUIRE_RESOURCE
	}

	if (keyStoreKey !is PrivateKey) {
		Log.e(TAG, "Unable to recover keystore alias $keystoreUser")
		return Error.ERR_CANT_ACQUIRE_RESOURCE
	}

	val certChain = keyStore.getCertificateChain(keystoreUser)
	if (certChain.isNullOrEmpty()) {
		Log.e(TAG, "Keystore alias $keystoreUser does not contain certificates")
		return Error.ERR_INVALID_DATA
	}
	val certs = ArrayList<X509Certificate>(certChain.size)
	for (cert in certChain) {
		certs.add(cert as X509Certificate)
	}

	val signerConfig = ApkSigner.SignerConfig.Builder(keystoreUser, keyStoreKey, certs).build()

	val apkSigner = ApkSigner.Builder(listOf(signerConfig))
		.setInputApk(File(inputPath))
		.setOutputApk(File(tmpOutputPath))
		.build()

	try {
		apkSigner.sign()
	} catch (e: Exception) {
		Log.e(TAG, "Unable to sign $inputPath", e)
		return Error.FAILED
	}

	if (outputPath != tmpOutputPath && !fileAccessHandler.renameFile(tmpOutputPath, outputPath)) {
		Log.e(TAG, "Unable to rename temp output file $tmpOutputPath to $outputPath")
		return Error.ERR_FILE_CANT_WRITE
	}

	Log.v(TAG, "Signed $inputPath")
	return Error.OK
}
