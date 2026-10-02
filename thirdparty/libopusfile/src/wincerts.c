/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE libopusfile SOFTWARE CODEC SOURCE CODE. *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE libopusfile SOURCE CODE IS (C) COPYRIGHT 2013-2016           *
 * by the Xiph.Org Foundation and contributors https://xiph.org/    *
 *                                                                  *
 ********************************************************************/

/*This should really be part of OpenSSL, but there's been a patch [1] sitting
   in their bugtracker for over two years that implements this, without any
   action, so I'm giving up and re-implementing it locally.

  [1] <https://rt.openssl.org/Ticket/Display.html?id=2158>*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "internal.h"
#if defined(OP_ENABLE_HTTP)&&defined(_WIN32)
/*You must include windows.h before wincrypt.h and x509.h.*/
# define WIN32_LEAN_AND_MEAN
# define WIN32_EXTRA_LEAN
# include <windows.h>
/*You must include wincrypt.h before x509.h, too, or X509_NAME doesn't get
   defined properly.*/
# include <wincrypt.h>
# include <openssl/ssl.h>
# include <openssl/err.h>
# include <openssl/x509.h>

static int op_capi_new(X509_LOOKUP *_lu){
  HCERTSTORE h_store;
  h_store=CertOpenStore(CERT_STORE_PROV_SYSTEM_A,0,0,
   CERT_STORE_OPEN_EXISTING_FLAG|CERT_STORE_READONLY_FLAG|
   CERT_SYSTEM_STORE_CURRENT_USER|CERT_STORE_SHARE_CONTEXT_FLAG,"ROOT");
  if(h_store!=NULL){
    _lu->method_data=(char *)h_store;
    return 1;
  }
  return 0;
}

static void op_capi_free(X509_LOOKUP *_lu){
  HCERTSTORE h_store;
  h_store=(HCERTSTORE)_lu->method_data;
# if defined(OP_ENABLE_ASSERTIONS)
  OP_ALWAYS_TRUE(CertCloseStore(h_store,CERT_CLOSE_STORE_CHECK_FLAG));
# else
  CertCloseStore(h_store,0);
# endif
}

static int op_capi_retrieve_by_subject(X509_LOOKUP *_lu,int _type,
 X509_NAME *_name,X509_OBJECT *_ret){
  X509_OBJECT *obj;
  CRYPTO_w_lock(CRYPTO_LOCK_X509_STORE);
  obj=X509_OBJECT_retrieve_by_subject(_lu->store_ctx->objs,_type,_name);
  CRYPTO_w_unlock(CRYPTO_LOCK_X509_STORE);
  if(obj!=NULL){
    _ret->type=obj->type;
    memcpy(&_ret->data,&obj->data,sizeof(_ret->data));
    return 1;
  }
  return 0;
}

static int op_capi_get_by_subject(X509_LOOKUP *_lu,int _type,X509_NAME *_name,
 X509_OBJECT *_ret){
  HCERTSTORE h_store;
  if(_name==NULL)return 0;
  if(_name->bytes==NULL||_name->bytes->length<=0||_name->modified){
    if(i2d_X509_NAME(_name,NULL)<0)return 0;
    OP_ASSERT(_name->bytes->length>0);
  }
  h_store=(HCERTSTORE)_lu->method_data;
  switch(_type){
    case X509_LU_X509:{
      CERT_NAME_BLOB  find_para;
      PCCERT_CONTEXT  cert;
      X509           *x;
      int             ret;
      /*Although X509_NAME contains a canon_enc field, that "canonical" [1]
         encoding was just made up by OpenSSL.
        It doesn't correspond to any actual standard, and since it drops the
         initial sequence header, won't be recognized by the Crypto API.
        The assumption here is that CertFindCertificateInStore() will allow any
         appropriate variations in the encoding when it does its comparison.
        This is, however, emphatically not true under Wine, which just compares
         the encodings with memcmp().
        Most of the time things work anyway, though, and there isn't really
         anything we can do to make the situation better.

        [1] A "canonical form" is defined as the one where, if you locked 10
         mathematicians in a room and asked them to come up with a
         representation for something, it's the answer that 9 of them would
         give you back.
        I don't think OpenSSL's encoding qualifies.*/
      if(OP_UNLIKELY(_name->bytes->length>MAXDWORD))return 0;
      find_para.cbData=(DWORD)_name->bytes->length;
      find_para.pbData=(unsigned char *)_name->bytes->data;
      cert=CertFindCertificateInStore(h_store,X509_ASN_ENCODING,0,
       CERT_FIND_SUBJECT_NAME,&find_para,NULL);
      if(cert==NULL)return 0;
      x=d2i_X509(NULL,(const unsigned char **)&cert->pbCertEncoded,
       cert->cbCertEncoded);
      CertFreeCertificateContext(cert);
      if(x==NULL)return 0;
      ret=X509_STORE_add_cert(_lu->store_ctx,x);
      X509_free(x);
      if(ret)return op_capi_retrieve_by_subject(_lu,_type,_name,_ret);
    }break;
    case X509_LU_CRL:{
      CERT_INFO      cert_info;
      CERT_CONTEXT   find_para;
      PCCRL_CONTEXT  crl;
      X509_CRL      *x;
      int            ret;
      ret=op_capi_retrieve_by_subject(_lu,_type,_name,_ret);
      if(ret>0)return ret;
      memset(&cert_info,0,sizeof(cert_info));
      if(OP_UNLIKELY(_name->bytes->length>MAXDWORD))return 0;
      cert_info.Issuer.cbData=(DWORD)_name->bytes->length;
      cert_info.Issuer.pbData=(unsigned char *)_name->bytes->data;
      memset(&find_para,0,sizeof(find_para));
      find_para.pCertInfo=&cert_info;
      crl=CertFindCRLInStore(h_store,0,0,CRL_FIND_ISSUED_BY,&find_para,NULL);
      if(crl==NULL)return 0;
      x=d2i_X509_CRL(NULL,(const unsigned char **)&crl->pbCrlEncoded,
       crl->cbCrlEncoded);
      CertFreeCRLContext(crl);
      if(x==NULL)return 0;
      ret=X509_STORE_add_crl(_lu->store_ctx,x);
      X509_CRL_free(x);
      if(ret)return op_capi_retrieve_by_subject(_lu,_type,_name,_ret);
    }break;
  }
  return 0;
}

/*This is not const because OpenSSL doesn't allow it, even though it won't
   write to it.*/
static X509_LOOKUP_METHOD X509_LOOKUP_CAPI={
  "Load Crypto API store into cache",
  op_capi_new,
  op_capi_free,
  NULL,
  NULL,
  NULL,
  op_capi_get_by_subject,
  NULL,
  NULL,
  NULL
};

int SSL_CTX_set_default_verify_paths_win32(SSL_CTX *_ssl_ctx){
  X509_STORE  *store;
  X509_LOOKUP *lu;
  /*We intentionally do not add the normal default paths, as they are usually
     wrong, and are just asking to be used as an exploit vector.*/
  store=SSL_CTX_get_cert_store(_ssl_ctx);
  OP_ASSERT(store!=NULL);
  lu=X509_STORE_add_lookup(store,&X509_LOOKUP_CAPI);
  if(lu==NULL)return 0;
  ERR_clear_error();
  return 1;
}

#endif
