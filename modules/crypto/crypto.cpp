#include "crypto.h"
#include <stdlib.h>

void Crypto::set_HMAC(bool hmac){
	use_hmac = hmac;
}

Vector<uint8_t> Crypto::HMAC(Vector<uint8_t> data){

	static unsigned char hash[32];
	
	data.resize(data.size()+32);
	
	for(size_t i =0; i < data.size()-32; i++){
		data[i+32] = data[i];
	}
	for(size_t i =0; i < 32; i++){
		data[i] = iKey[i];
	}
	
	
	sha256_context shaCtx;
	sha256_init(&shaCtx);
	sha256_hash(&shaCtx,(unsigned char*)data.ptr(),data.size());
	sha256_done(&shaCtx, hash);
	
	Vector<uint8_t> hashVec;
	
	hashVec.resize(64);
	
	for(size_t i=0;i<32;i++){
		hashVec[i] = oKey[i];
		hashVec[i+32] = hash[i];
	}
	
	sha256_init(&shaCtx);
	sha256_hash(&shaCtx,(unsigned char*)hashVec.ptr(),hashVec.size());
	sha256_done(&shaCtx, hash);
	
	hashVec.resize(32);
	
	for(size_t i=0;i<32;i++){
		hashVec[i] = hash[i];
	}
	
	
	return hashVec;
}

void Crypto::set_password(String password){
	CharString cs=password.utf8();
	unsigned char key[32];
	sha256_context ctx;
	sha256_init(&ctx);
	sha256_hash(&ctx,(unsigned char*)cs.ptr(),cs.length());
	sha256_done(&ctx, key);
	
	Vector<uint8_t> keyVect;
	keyVect.resize(32);
	
	for(int i=0;i<32;i++){
		keyVect[i] = key[i];
	}

	Crypto::set_key(keyVect);	
}

void Crypto::set_key(Vector<uint8_t> key){
	
	if(key.size() < 32){
	
		size_t keySize = key.size();
	
		key.resize(32);
		
		for(size_t i =keySize; i<32; i++){
			key[i] = 0;
		}
		
	}else if(key.size() > 32){
		static unsigned char hash[32];
		
		Vector<uint8_t> data;
		
		sha256_context shaCtx;
		sha256_init(&shaCtx);
		sha256_hash(&shaCtx,(unsigned char*)data.ptr(),data.size());
		sha256_done(&shaCtx, hash);
				
		key.resize(32);
		
		for(size_t i=0;i<32;i++){
			key[i] = hash[i];
		}
	
	}
	
	keyToUse.resize(32);
	iKey.resize(32);		
	oKey.resize(32);
	
	
	for(size_t j=0;j<32;j++){	
		keyToUse[j] = key[j]; //Key
		oKey[j] = key[j] ^ 92; //Key
		iKey[j] = key[j] ^ 54; //Key
	}
	
	
	

}

Vector<uint8_t> Crypto::encrypt_string(String plainText){
	

	CharString cs =  plainText.utf8();
	
	Vector<uint8_t> data;
	data.resize(cs.size());
	
	for(int i=0;i<cs.size();i++){
		data[i] = cs[i];
	}
	

	return encrypt_raw(data);
}

Vector<uint8_t> Crypto::encrypt_raw(Vector<uint8_t> plainRaw){
	Vector<uint8_t> data;
	
	aes256_context ctx;
	
	aes256_init(&ctx,keyToUse.ptr());
	
	
	int len = plainRaw.size();
	int paddedLen = len+ 1; //We need at least one byte to say how much padding their is.
	
	if (paddedLen % 16) {
		paddedLen+=16-(paddedLen % 16); //We need to get a 16byte block
	}
	
	data.resize(paddedLen+16); //Padding Plus Iv
		
	char padding = (char)paddedLen-len; //Which char is the padding chracter
	
	
	for(size_t j=0;j<16;j++){	
		data[j] = rand()%256; //random byte for the first 16.
	}
	
	aes256_encrypt_ecb(&ctx,&data[0]); //Now we encrypt the first 16 bytes to get our iv.
		
	for(size_t j=0;j<len;j++){	
		
		data[j+16] = plainRaw[j]; //Actual data;
		
		
	}

	for(size_t j=len;j<paddedLen;j++){	
		
		data[j+16] = padding; //padding at the end 
	}
			
	for(size_t i=16;i<data.size();i+=16) {			
		for(size_t j=0;j<16;j++) {		
			data[i+j] = data[i+j] ^ data[i+j-16]; //Xor the previous block with the current one
		}
		
		aes256_encrypt_ecb(&ctx,&data[i]);
	}
	
	
	Vector<uint8_t> hash;
	
	aes256_done(&ctx);	
	
	if (use_hmac ){
	
		hash = HMAC(data);
	
		data.resize(data.size()+32);
		
		for(size_t i =0; i< data.size()-32;i++){
			data[i+32] = data[i]; //Move the data 32 chracters along
		}
		for(size_t i =0; i< 32;i++){
			data[i] = hash[i]; //Add the hash			
		}
	
	}
	
	
		
	return data;
}

String Crypto::decrypt_string (Vector<uint8_t> encrypted){
	Vector<uint8_t>  dc = decrypt_raw(encrypted);
	
	char* data = new char[dc.size()];
	
	for (size_t i =0;i<dc.size();i++){
		data[i] = (char)dc[i];
	}
	
	return String::utf8(data,dc.size());
}

Vector<uint8_t> Crypto::decrypt_raw (Vector<uint8_t> encrypted){
	Vector<uint8_t> data;
	
	if (use_hmac){
	
		if(encrypted.size() < 32){
			return data;
		}
	
		Vector<uint8_t> hash;
		Vector<uint8_t> hashCalculated;
		
		hash.resize(32);
			
		for(size_t i =0;i<32;i++){
			hash[i] = encrypted[i]; //Get the hash
		}
				
		for(size_t i =0;i<encrypted.size()-32;i++){
			encrypted[i] = encrypted[i+32]; //Removing the hash from the front
		}
		
		encrypted.resize(encrypted.size()-32); //Removing the left over from the back
		
		hashCalculated = HMAC(encrypted);
			
		bool correct = true;
		
		for(size_t i=0;i<32;i++){
		
			if(hashCalculated[i] != hash[i]){
				correct = false;
				//You do not want to exit early as that introduces a timing attack
			}
		}
		
		if(!correct){
			return data; //Empty Array
		}
		
		
	}
	
	data.resize(encrypted.size()-16); //We do not want to return the IV so we need 16 bytes less
	
	aes256_context ctx;
	aes256_init(&ctx,keyToUse.ptr());	
	
	for(size_t i=0;i<data.size();i+=16){
		
		for(size_t j=0;j<16;j++) {		
			data[i+j] = encrypted[i+j+16]; //Get the data 
		}	
			
		aes256_decrypt_ecb(&ctx,&data[i]); //Decrypt the data
		
		for(size_t j=0;j<16;j++) {		
			data[i+j] = data[i+j] ^ encrypted[i+j]; //And xor with the previous block
		}
	}
	
	uint8_t paddingNumber = (uint8_t) data[data.size()-1];
		
	if (paddingNumber < 16){		
		data.resize(data.size() - paddingNumber ); //Remove the padding
	}
	
	return data;
}

void Crypto::_bind_methods(){

    ObjectTypeDB::bind_method("set_password",&Crypto::set_password);
    ObjectTypeDB::bind_method("set_key",&Crypto::set_key);
    ObjectTypeDB::bind_method("set_HMAC",&Crypto::set_HMAC);
	
	
    ObjectTypeDB::bind_method("encrypt_string",&Crypto::encrypt_string);
    ObjectTypeDB::bind_method("encrypt_raw",&Crypto::encrypt_raw);
	
    ObjectTypeDB::bind_method("decrypt_string",&Crypto::decrypt_string);
    ObjectTypeDB::bind_method("decrypt_raw",&Crypto::decrypt_raw);
	
    ObjectTypeDB::bind_method("HMAC",&Crypto::HMAC);
	
}

