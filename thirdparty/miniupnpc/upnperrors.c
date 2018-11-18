/* $Id: upnperrors.c,v 1.5 2011/04/10 11:19:36 nanard Exp $ */
/* Project : miniupnp
 * Author : Thomas BERNARD
 * copyright (c) 2007 Thomas Bernard
 * All Right reserved.
 * http://miniupnp.free.fr/ or http://miniupnp.tuxfamily.org/
 * This software is subjet to the conditions detailed in the
 * provided LICENCE file. */
#include <string.h>
#include "upnperrors.h"
#include "upnpcommands.h"
#include "miniupnpc.h"

const char * strupnperror(int err)
{
	const char * s = NULL;
	switch(err) {
	case UPNPCOMMAND_SUCCESS:
		s = "Success";
		break;
	case UPNPCOMMAND_UNKNOWN_ERROR:
		s = "Miniupnpc Unknown Error";
		break;
	case UPNPCOMMAND_INVALID_ARGS:
		s = "Miniupnpc Invalid Arguments";
		break;
	case UPNPCOMMAND_INVALID_RESPONSE:
		s = "Miniupnpc Invalid response";
		break;
	case UPNPDISCOVER_SOCKET_ERROR:
		s = "Miniupnpc Socket error";
		break;
	case UPNPDISCOVER_MEMORY_ERROR:
		s = "Miniupnpc Memory allocation error";
		break;
	case 401:
		s = "Invalid Action";
		break;
	case 402:
		s = "Invalid Args";
		break;
	case 501:
		s = "Action Failed";
		break;
	case 606:
		s = "Action not authorized";
		break;
	case 701:
		s = "PinholeSpaceExhausted";
		break;
	case 702:
		s = "FirewallDisabled";
		break;
	case 703:
		s = "InboundPinholeNotAllowed";
		break;
	case 704:
		s = "NoSuchEntry";
		break;
	case 705:
		s = "ProtocolNotSupported";
		break;
	case 706:
		s = "InternalPortWildcardingNotAllowed";
		break;
	case 707:
		s = "ProtocolWildcardingNotAllowed";
		break;
	case 708:
		s = "WildcardNotPermittedInSrcIP";
		break;
	case 709:
		s = "NoPacketSent";
		break;
	case 713:
		s = "SpecifiedArrayIndexInvalid";
		break;
	case 714:
		s = "NoSuchEntryInArray";
		break;
	case 715:
		s = "WildCardNotPermittedInSrcIP";
		break;
	case 716:
		s = "WildCardNotPermittedInExtPort";
		break;
	case 718:
		s = "ConflictInMappingEntry";
		break;
	case 724:
		s = "SamePortValuesRequired";
		break;
	case 725:
		s = "OnlyPermanentLeasesSupported";
		break;
	case 726:
		s = "RemoteHostOnlySupportsWildcard";
		break;
	case 727:
		s = "ExternalPortOnlySupportsWildcard";
		break;
	default:
		s = "UnknownError";
		break;
	}
	return s;
}
