/* $Id: upnpreplyparse.h,v 1.22 2025/03/29 17:58:12 nanard Exp $ */
/* MiniUPnP project
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * (c) 2006-2025 Thomas Bernard
 * This software is subject to the conditions detailed
 * in the LICENCE file provided within the distribution */

#ifndef UPNPREPLYPARSE_H_INCLUDED
#define UPNPREPLYPARSE_H_INCLUDED

/*! \file upnpreplyparse.h
 * \brief Parsing of UPnP SOAP responses
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Name/Value linked list
 * not exposed in the public API
 */
struct NameValue;

/*! \brief data structure for parsing */
struct NameValueParserData {
	/*! \brief name/value linked list */
	struct NameValue * l_head;
	/*! \brief current element name */
	char curelt[64];
	/*! \brief port listing array */
	char * portListing;
	/*! \brief port listing array length */
	int portListingLength;
	/*! \brief flag indicating the current element is  */
	int topelt;
	/*! \brief top element character data */
	const char * cdata;
	/*! \brief top element character data length */
	int cdatalen;
};

/*!
 * \brief Parse XML and fill the structure
 *
 * \param[in] buffer XML data
 * \param[in] bufsize buffer length
 * \param[out] data structure to fill
 */
void
ParseNameValue(const char * buffer, int bufsize,
               struct NameValueParserData * data);

/*!
 * \brief free memory
 *
 * \param[in,out] pdata data structure
 */
void
ClearNameValueList(struct NameValueParserData * pdata);

/*!
 * \brief get a value from the parsed data
 *
 * \param[in] pdata data structure
 * \param[in] name name
 * \return the value or NULL if not found
 */
char *
GetValueFromNameValueList(struct NameValueParserData * pdata,
                          const char * name);

/* DisplayNameValueList() */
#ifdef DEBUG
void
DisplayNameValueList(char * buffer, int bufsize);
#endif

#ifdef __cplusplus
}
#endif

#endif
