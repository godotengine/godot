/* set of parsable strings -- ALL LOWER CASE */

#if !defined(STORE_IN_ROM)
#define STORE_IN_ROM
#endif

STORE_IN_ROM static const char * const set[] = {
	"get ",
	"post ",
	"options ",
	"host:",
	"connection:",
	"upgrade:",
	"origin:",
	"sec-websocket-draft:",
	"\x0d\x0a",

	"sec-websocket-extensions:",
	"sec-websocket-key1:",
	"sec-websocket-key2:",
	"sec-websocket-protocol:",

	"sec-websocket-accept:",
	"sec-websocket-nonce:",
	"http/1.1 ",
	"http2-settings:",

	"accept:",
	"access-control-request-headers:",
	"if-modified-since:",
	"if-none-match:",
	"accept-encoding:",
	"accept-language:",
	"pragma:",
	"cache-control:",
	"authorization:",
	"cookie:",
	"content-length:",
	"content-type:",
	"date:",
	"range:",
	"referer:",
	"sec-websocket-key:",
	"sec-websocket-version:",
	"sec-websocket-origin:",

	":authority",
	":method",
	":path",
	":scheme",
	":status",

	"accept-charset:",
	"accept-ranges:",
	"access-control-allow-origin:",
	"age:",
	"allow:",
	"content-disposition:",
	"content-encoding:",
	"content-language:",
	"content-location:",
	"content-range:",
	"etag:",
	"expect:",
	"expires:",
	"from:",
	"if-match:",
	"if-range:",
	"if-unmodified-since:",
	"last-modified:",
	"link:",
	"location:",
	"max-forwards:",
	"proxy-authenticate:",
	"proxy-authorization:",
	"refresh:",
	"retry-after:",
	"server:",
	"set-cookie:",
	"strict-transport-security:",
	"transfer-encoding:",
	"user-agent:",
	"vary:",
	"via:",
	"www-authenticate:",

	"patch",
	"put",
	"delete",

	"uri-args", /* fake header used for uri-only storage */

	"proxy ",
	"x-real-ip:",
	"http/1.0 ",

	"x-forwarded-for",
	"connect ",
	"head ",
	"te:",		/* http/2 wants it to reject it */

	"", /* not matchable */

};
