/* $Id: upnpreplyparse.c,v 1.22 2025/02/08 23:12:26 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * MiniUPnP project
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * (c) 2006-2025 Thomas Bernard
 * This software is subject to the conditions detailed
 * in the LICENCE file provided within the distribution */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "upnpreplyparse.h"
#include "minixml.h"

struct NameValue {
	/*! \brief pointer to the next element */
	struct NameValue * l_next;
	/*! \brief name */
	char name[64];
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
	/* C99 flexible array member */
	/*! \brief character value */
	char value[];
#elif defined(__GNUC__)
	char value[0];
#else
	/* Fallback to a hack */
	char value[1];
#endif
};

static void
NameValueParserStartElt(void * d, const char * name, int l)
{
	struct NameValueParserData * data = (struct NameValueParserData *)d;
	data->topelt = 1;
    if(l>63)
        l = 63;
    memcpy(data->curelt, name, l);
    data->curelt[l] = '\0';
	data->cdata = NULL;
	data->cdatalen = 0;
}

static void
NameValueParserEndElt(void * d, const char * name, int namelen)
{
    struct NameValueParserData * data = (struct NameValueParserData *)d;
    struct NameValue * nv;
	(void)name;
	(void)namelen;
	if(!data->topelt)
		return;
	if(strcmp(data->curelt, "NewPortListing") != 0)
	{
		int l;
		/* standard case. Limited to n chars strings */
		l = data->cdatalen;
	    nv = malloc(sizeof(struct NameValue) + l + 1);
		if(nv == NULL)
		{
			/* malloc error */
#ifdef DEBUG
			fprintf(stderr, "%s: error allocating memory",
			        "NameValueParserEndElt");
#endif /* DEBUG */
			return;
		}
	    strncpy(nv->name, data->curelt, 64);
		nv->name[63] = '\0';
		if(data->cdata != NULL)
		{
			memcpy(nv->value, data->cdata, l);
			nv->value[l] = '\0';
		}
		else
		{
			nv->value[0] = '\0';
		}
		nv->l_next = data->l_head;	/* insert in list */
		data->l_head = nv;
	}
	data->cdata = NULL;
	data->cdatalen = 0;
	data->topelt = 0;
}

static void
NameValueParserGetData(void * d, const char * datas, int l)
{
    struct NameValueParserData * data = (struct NameValueParserData *)d;
	if(strcmp(data->curelt, "NewPortListing") == 0)
	{
		/* specific case for NewPortListing which is a XML Document */
		free(data->portListing);
		data->portListing = malloc(l + 1);
		if(!data->portListing)
		{
			/* malloc error */
#ifdef DEBUG
			fprintf(stderr, "%s: error allocating memory",
			        "NameValueParserGetData");
#endif /* DEBUG */
			return;
		}
		memcpy(data->portListing, datas, l);
		data->portListing[l] = '\0';
		data->portListingLength = l;
	}
	else
	{
		/* standard case. */
		data->cdata = datas;
		data->cdatalen = l;
	}
}

void
ParseNameValue(const char * buffer, int bufsize,
               struct NameValueParserData * data)
{
	struct xmlparser parser;
	memset(data, 0, sizeof(struct NameValueParserData));
	/* init xmlparser object */
	parser.xmlstart = buffer;
	parser.xmlsize = bufsize;
	parser.data = data;
	parser.starteltfunc = NameValueParserStartElt;
	parser.endeltfunc = NameValueParserEndElt;
	parser.datafunc = NameValueParserGetData;
	parser.attfunc = 0;
	parsexml(&parser);
}

void
ClearNameValueList(struct NameValueParserData * pdata)
{
    struct NameValue * nv;
	if(pdata->portListing)
	{
		free(pdata->portListing);
		pdata->portListing = NULL;
		pdata->portListingLength = 0;
	}
    while((nv = pdata->l_head) != NULL)
    {
		pdata->l_head = nv->l_next;
        free(nv);
    }
}

char *
GetValueFromNameValueList(struct NameValueParserData * pdata,
                          const char * name)
{
    struct NameValue * nv;
    char * p = NULL;
    for(nv = pdata->l_head;
        (nv != NULL) && (p == NULL);
        nv = nv->l_next)
    {
        if(strcmp(nv->name, name) == 0)
            p = nv->value;
    }
    return p;
}

/* debug all-in-one function
 * do parsing then display to stdout */
#ifdef DEBUG
void
DisplayNameValueList(char * buffer, int bufsize)
{
    struct NameValueParserData pdata;
    struct NameValue * nv;
    ParseNameValue(buffer, bufsize, &pdata);
    for(nv = pdata.l_head;
        nv != NULL;
        nv = nv->l_next)
    {
        printf("%s = %s\n", nv->name, nv->value);
    }
    ClearNameValueList(&pdata);
}
#endif /* DEBUG */

