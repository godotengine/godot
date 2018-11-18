/* $Id: minixml.c,v 1.10 2012/03/05 19:42:47 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * minixml.c : the minimum size a xml parser can be ! */
/* Project : miniupnp
 * webpage: http://miniupnp.free.fr/ or http://miniupnp.tuxfamily.org/
 * Author : Thomas Bernard

Copyright (c) 2005-2017, Thomas BERNARD
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * The name of the author may not be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#include <string.h>
#include "minixml.h"

/* parseatt : used to parse the argument list
 * return 0 (false) in case of success and -1 (true) if the end
 * of the xmlbuffer is reached. */
static int parseatt(struct xmlparser * p)
{
	const char * attname;
	int attnamelen;
	const char * attvalue;
	int attvaluelen;
	while(p->xml < p->xmlend)
	{
		if(*p->xml=='/' || *p->xml=='>')
			return 0;
		if( !IS_WHITE_SPACE(*p->xml) )
		{
			char sep;
			attname = p->xml;
			attnamelen = 0;
			while(*p->xml!='=' && !IS_WHITE_SPACE(*p->xml) )
			{
				attnamelen++; p->xml++;
				if(p->xml >= p->xmlend)
					return -1;
			}
			while(*(p->xml++) != '=')
			{
				if(p->xml >= p->xmlend)
					return -1;
			}
			while(IS_WHITE_SPACE(*p->xml))
			{
				p->xml++;
				if(p->xml >= p->xmlend)
					return -1;
			}
			sep = *p->xml;
			if(sep=='\'' || sep=='\"')
			{
				p->xml++;
				if(p->xml >= p->xmlend)
					return -1;
				attvalue = p->xml;
				attvaluelen = 0;
				while(*p->xml != sep)
				{
					attvaluelen++; p->xml++;
					if(p->xml >= p->xmlend)
						return -1;
				}
			}
			else
			{
				attvalue = p->xml;
				attvaluelen = 0;
				while(   !IS_WHITE_SPACE(*p->xml)
					  && *p->xml != '>' && *p->xml != '/')
				{
					attvaluelen++; p->xml++;
					if(p->xml >= p->xmlend)
						return -1;
				}
			}
			/*printf("%.*s='%.*s'\n",
			       attnamelen, attname, attvaluelen, attvalue);*/
			if(p->attfunc)
				p->attfunc(p->data, attname, attnamelen, attvalue, attvaluelen);
		}
		p->xml++;
	}
	return -1;
}

/* parseelt parse the xml stream and
 * call the callback functions when needed... */
static void parseelt(struct xmlparser * p)
{
	int i;
	const char * elementname;
	while(p->xml < (p->xmlend - 1))
	{
		if((p->xml + 4) <= p->xmlend && (0 == memcmp(p->xml, "<!--", 4)))
		{
			p->xml += 3;
			/* ignore comments */
			do
			{
				p->xml++;
				if ((p->xml + 3) >= p->xmlend)
					return;
			}
			while(memcmp(p->xml, "-->", 3) != 0);
			p->xml += 3;
		}
		else if((p->xml)[0]=='<' && (p->xml)[1]!='?')
		{
			i = 0; elementname = ++p->xml;
			while( !IS_WHITE_SPACE(*p->xml)
				  && (*p->xml!='>') && (*p->xml!='/')
				 )
			{
				i++; p->xml++;
				if (p->xml >= p->xmlend)
					return;
				/* to ignore namespace : */
				if(*p->xml==':')
				{
					i = 0;
					elementname = ++p->xml;
				}
			}
			if(i>0)
			{
				if(p->starteltfunc)
					p->starteltfunc(p->data, elementname, i);
				if(parseatt(p))
					return;
				if(*p->xml!='/')
				{
					const char * data;
					i = 0; data = ++p->xml;
					if (p->xml >= p->xmlend)
						return;
					while( IS_WHITE_SPACE(*p->xml) )
					{
						i++; p->xml++;
						if (p->xml >= p->xmlend)
							return;
					}
					/* CDATA are at least 9 + 3 characters long : <![CDATA[ ]]> */
					if((p->xmlend >= (p->xml + (9 + 3))) && (memcmp(p->xml, "<![CDATA[", 9) == 0))
					{
						/* CDATA handling */
						p->xml += 9;
						data = p->xml;
						i = 0;
						while(memcmp(p->xml, "]]>", 3) != 0)
						{
							i++; p->xml++;
							if ((p->xml + 3) >= p->xmlend)
								return;
						}
						if(i>0 && p->datafunc)
							p->datafunc(p->data, data, i);
						while(*p->xml!='<')
						{
							p->xml++;
							if (p->xml >= p->xmlend)
								return;
						}
					}
					else
					{
						while(*p->xml!='<')
						{
							i++; p->xml++;
							if ((p->xml + 1) >= p->xmlend)
								return;
						}
						if(i>0 && p->datafunc && *(p->xml + 1) == '/')
							p->datafunc(p->data, data, i);
					}
				}
			}
			else if(*p->xml == '/')
			{
				i = 0; elementname = ++p->xml;
				if (p->xml >= p->xmlend)
					return;
				while((*p->xml != '>'))
				{
					i++; p->xml++;
					if (p->xml >= p->xmlend)
						return;
				}
				if(p->endeltfunc)
					p->endeltfunc(p->data, elementname, i);
				p->xml++;
			}
		}
		else
		{
			p->xml++;
		}
	}
}

/* the parser must be initialized before calling this function */
void parsexml(struct xmlparser * parser)
{
	parser->xml = parser->xmlstart;
	parser->xmlend = parser->xmlstart + parser->xmlsize;
	parseelt(parser);
}


