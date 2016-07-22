/* sumator.h */
#ifndef CRYPTO_H
#define CRYPTO_H


#include "reference.h"
#include "io/sha256.h"
#include "io/aes256.h"

class Crypto : public Reference {
    OBJ_TYPE(Crypto,Reference);

	private:
		Vector<uint8_t> keyToUse;
		
		Vector<uint8_t> oKey;		
		Vector<uint8_t> iKey;
		
		bool use_hmac ;
		
		
	protected:
		 static void _bind_methods();
		
	public:
		Crypto(){
			use_hmac = true;
		}
		
		void set_password(String password);
		void set_key(Vector<uint8_t> key);
		
		void set_HMAC(bool hmac);
		
		Vector<uint8_t> encrypt_string( String plainText);
		Vector<uint8_t> encrypt_raw( Vector<uint8_t> plainRaw);
		
		String decrypt_string (Vector<uint8_t> encrypted);	
		Vector<uint8_t> decrypt_raw (Vector<uint8_t> encrypted);
				
		Vector<uint8_t>  HMAC(Vector<uint8_t> data);
};

#endif

