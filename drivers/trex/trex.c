	/* see copyright notice in trex.h */
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <setjmp.h>
#include "trex.h"

#ifdef _UINCODE
#define scisprint iswprint
#define scstrlen wcslen
#define scprintf wprintf
#define _SC(x) L##c
#else
#define scisprint isprint
#define scstrlen strlen
#define scprintf printf
#define _SC(x) (x)
#endif

#ifdef _DEBUG
#include <stdio.h>

static const TRexChar *g_nnames[] =
{
	_SC("NONE"),_SC("OP_GREEDY"),	_SC("OP_OR"),
	_SC("OP_EXPR"),_SC("OP_NOCAPEXPR"),_SC("OP_DOT"),	_SC("OP_CLASS"),
	_SC("OP_CCLASS"),_SC("OP_NCLASS"),_SC("OP_RANGE"),_SC("OP_CHAR"),
	_SC("OP_EOL"),_SC("OP_BOL"),_SC("OP_WB")
};

#endif
#define OP_GREEDY		(MAX_CHAR+1) // * + ? {n}
#define OP_OR			(MAX_CHAR+2)
#define OP_EXPR			(MAX_CHAR+3) //parentesis ()
#define OP_NOCAPEXPR	(MAX_CHAR+4) //parentesis (?:)
#define OP_DOT			(MAX_CHAR+5)
#define OP_CLASS		(MAX_CHAR+6)
#define OP_CCLASS		(MAX_CHAR+7)
#define OP_NCLASS		(MAX_CHAR+8) //negates class the [^
#define OP_RANGE		(MAX_CHAR+9)
#define OP_CHAR			(MAX_CHAR+10)
#define OP_EOL			(MAX_CHAR+11)
#define OP_BOL			(MAX_CHAR+12)
#define OP_WB			(MAX_CHAR+13)

#define TREX_SYMBOL_ANY_CHAR ('.')
#define TREX_SYMBOL_GREEDY_ONE_OR_MORE ('+')
#define TREX_SYMBOL_GREEDY_ZERO_OR_MORE ('*')
#define TREX_SYMBOL_GREEDY_ZERO_OR_ONE ('?')
#define TREX_SYMBOL_BRANCH ('|')
#define TREX_SYMBOL_END_OF_STRING ('$')
#define TREX_SYMBOL_BEGINNING_OF_STRING ('^')
#define TREX_SYMBOL_ESCAPE_CHAR ('\\')


typedef int TRexNodeType;

typedef struct tagTRexNode{
	TRexNodeType type;
	int left;
	int right;
	int next;
}TRexNode;

struct TRex{
	const TRexChar *_eol;
	const TRexChar *_bol;
	const TRexChar *_p;
	int _first;
	int _op;
	TRexNode *_nodes;
	int _nallocated;
	int _nsize;
	int _nsubexpr;
	TRexMatch *_matches;
	int _currsubexp;
	void *_jmpbuf;
	const TRexChar **_error;
};

static int trex_list(TRex *exp);

static int trex_newnode(TRex *exp, TRexNodeType type)
{
	TRexNode n;
	int newid;
	n.type = type;
	n.next = n.right = n.left = -1;
	if(type == OP_EXPR)
		n.right = exp->_nsubexpr++;
	if(exp->_nallocated < (exp->_nsize + 1)) {
		//int oldsize = exp->_nallocated;
		exp->_nallocated *= 2;
		exp->_nodes = (TRexNode *)realloc(exp->_nodes, exp->_nallocated * sizeof(TRexNode));
	}
	exp->_nodes[exp->_nsize++] = n;
	newid = exp->_nsize - 1;
	return (int)newid;
}

static void trex_error(TRex *exp,const TRexChar *error)
{
	if(exp->_error) *exp->_error = error;
	longjmp(*((jmp_buf*)exp->_jmpbuf),-1);
}

static void trex_expect(TRex *exp, int n){
	if((*exp->_p) != n) 
		trex_error(exp, _SC("expected paren"));
	exp->_p++;
}

static TRexChar trex_escapechar(TRex *exp)
{
	if(*exp->_p == TREX_SYMBOL_ESCAPE_CHAR){
		exp->_p++;
		switch(*exp->_p) {
		case 'v': exp->_p++; return '\v';
		case 'n': exp->_p++; return '\n';
		case 't': exp->_p++; return '\t';
		case 'r': exp->_p++; return '\r';
		case 'f': exp->_p++; return '\f';
		default: return (*exp->_p++);
		}
	} else if(!scisprint(*exp->_p)) trex_error(exp,_SC("letter expected"));
	return (*exp->_p++);
}

static int trex_charclass(TRex *exp,int classid)
{
	int n = trex_newnode(exp,OP_CCLASS);
	exp->_nodes[n].left = classid;
	return n;
}

static int trex_charnode(TRex *exp,TRexBool isclass)
{
	TRexChar t;
	if(*exp->_p == TREX_SYMBOL_ESCAPE_CHAR) {
		exp->_p++;
		switch(*exp->_p) {
			case 'n': exp->_p++; return trex_newnode(exp,'\n');
			case 't': exp->_p++; return trex_newnode(exp,'\t');
			case 'r': exp->_p++; return trex_newnode(exp,'\r');
			case 'f': exp->_p++; return trex_newnode(exp,'\f');
			case 'v': exp->_p++; return trex_newnode(exp,'\v');
			case 'a': case 'A': case 'w': case 'W': case 's': case 'S': 
			case 'd': case 'D': case 'x': case 'X': case 'c': case 'C': 
			case 'p': case 'P': case 'l': case 'u': 
				{
				t = *exp->_p; exp->_p++; 
				return trex_charclass(exp,t);
				}
			case 'b': 
			case 'B':
				if(!isclass) {
					int node = trex_newnode(exp,OP_WB);
					exp->_nodes[node].left = *exp->_p;
					exp->_p++; 
					return node;
				} //else default
			default: 
				t = *exp->_p; exp->_p++; 
				return trex_newnode(exp,t);
		}
	}
	else if(!scisprint(*exp->_p)) {
		
		trex_error(exp,_SC("letter expected"));
	}
	t = *exp->_p; exp->_p++; 
	return trex_newnode(exp,t);
}
static int trex_class(TRex *exp)
{
	int ret = -1;
	int first = -1,chain;
	if(*exp->_p == TREX_SYMBOL_BEGINNING_OF_STRING){
		ret = trex_newnode(exp,OP_NCLASS);
		exp->_p++;
	}else ret = trex_newnode(exp,OP_CLASS);
	
	if(*exp->_p == ']') trex_error(exp,_SC("empty class"));
	chain = ret;
	while(*exp->_p != ']' && exp->_p != exp->_eol) {
		if(*exp->_p == '-' && first != -1){ 
			int r,t;
			if(*exp->_p++ == ']') trex_error(exp,_SC("unfinished range"));
			r = trex_newnode(exp,OP_RANGE);
			if(first>*exp->_p) trex_error(exp,_SC("invalid range"));
			if(exp->_nodes[first].type == OP_CCLASS) trex_error(exp,_SC("cannot use character classes in ranges"));
			exp->_nodes[r].left = exp->_nodes[first].type;
			t = trex_escapechar(exp);
			exp->_nodes[r].right = t;
            exp->_nodes[chain].next = r;
			chain = r;
			first = -1;
		}
		else{
			if(first!=-1){
				int c = first;
				exp->_nodes[chain].next = c;
				chain = c;
				first = trex_charnode(exp,TRex_True);
			}
			else{
				first = trex_charnode(exp,TRex_True);
			}
		}
	}
	if(first!=-1){
		int c = first;
		exp->_nodes[chain].next = c;
		chain = c;
		first = -1;
	}
	/* hack? */
	exp->_nodes[ret].left = exp->_nodes[ret].next;
	exp->_nodes[ret].next = -1;
	return ret;
}

static int trex_parsenumber(TRex *exp)
{
	int ret = *exp->_p-'0';
	int positions = 10;
	exp->_p++;
	while(isdigit(*exp->_p)) {
		ret = ret*10+(*exp->_p++-'0');
		if(positions==1000000000) trex_error(exp,_SC("overflow in numeric constant"));
		positions *= 10;
	};
	return ret;
}

static int trex_element(TRex *exp)
{
	int ret = -1;
	switch(*exp->_p)
	{
	case '(': {
		int expr,newn;
		exp->_p++;


		if(*exp->_p =='?') {
			exp->_p++;
			trex_expect(exp,':');
			expr = trex_newnode(exp,OP_NOCAPEXPR);
		}
		else
			expr = trex_newnode(exp,OP_EXPR);
		newn = trex_list(exp);
		exp->_nodes[expr].left = newn;
		ret = expr;
		trex_expect(exp,')');
			  }
			  break;
	case '[':
		exp->_p++;
		ret = trex_class(exp);
		trex_expect(exp,']');
		break;
	case TREX_SYMBOL_END_OF_STRING: exp->_p++; ret = trex_newnode(exp,OP_EOL);break;
	case TREX_SYMBOL_ANY_CHAR: exp->_p++; ret = trex_newnode(exp,OP_DOT);break;
	default:
		ret = trex_charnode(exp,TRex_False);
		break;
	}

	{
		int op;
		TRexBool isgreedy = TRex_False;
		unsigned short p0 = 0, p1 = 0;
		switch(*exp->_p){
			case TREX_SYMBOL_GREEDY_ZERO_OR_MORE: p0 = 0; p1 = 0xFFFF; exp->_p++; isgreedy = TRex_True; break;
			case TREX_SYMBOL_GREEDY_ONE_OR_MORE: p0 = 1; p1 = 0xFFFF; exp->_p++; isgreedy = TRex_True; break;
			case TREX_SYMBOL_GREEDY_ZERO_OR_ONE: p0 = 0; p1 = 1; exp->_p++; isgreedy = TRex_True; break;
			case '{':
				exp->_p++;
				if(!isdigit(*exp->_p)) trex_error(exp,_SC("number expected"));
				p0 = (unsigned short)trex_parsenumber(exp);
				/*******************************/
				switch(*exp->_p) {
			case '}':
				p1 = p0; exp->_p++;
				break;
			case ',':
				exp->_p++;
				p1 = 0xFFFF;
				if(isdigit(*exp->_p)){
					p1 = (unsigned short)trex_parsenumber(exp);
				}
				trex_expect(exp,'}');
				break;
			default:
				trex_error(exp,_SC(", or } expected"));
		}
		/*******************************/
		isgreedy = TRex_True; 
		break;

		}
		if(isgreedy) {
			int nnode = trex_newnode(exp,OP_GREEDY);
			op = OP_GREEDY;
			exp->_nodes[nnode].left = ret;
			exp->_nodes[nnode].right = ((p0)<<16)|p1;
			ret = nnode;
		}
	}
	if((*exp->_p != TREX_SYMBOL_BRANCH) && (*exp->_p != ')') && (*exp->_p != TREX_SYMBOL_GREEDY_ZERO_OR_MORE) && (*exp->_p != TREX_SYMBOL_GREEDY_ONE_OR_MORE) && (*exp->_p != '\0')) {
		int nnode = trex_element(exp);
		exp->_nodes[ret].next = nnode;
	}

	return ret;
}

static int trex_list(TRex *exp)
{
	int ret=-1,e;
	if(*exp->_p == TREX_SYMBOL_BEGINNING_OF_STRING) {
		exp->_p++;
		ret = trex_newnode(exp,OP_BOL);
	}
	e = trex_element(exp);
	if(ret != -1) {
		exp->_nodes[ret].next = e;
	}
	else ret = e;

	if(*exp->_p == TREX_SYMBOL_BRANCH) {
		int temp,tright;
		exp->_p++;
		temp = trex_newnode(exp,OP_OR);
		exp->_nodes[temp].left = ret;
		tright = trex_list(exp);
		exp->_nodes[temp].right = tright;
		ret = temp;
	}
	return ret;
}

static TRexBool trex_matchcclass(int cclass,TRexChar c)
{
	switch(cclass) {
	case 'a': return isalpha(c)?TRex_True:TRex_False;
	case 'A': return !isalpha(c)?TRex_True:TRex_False;
	case 'w': return (isalnum(c) || c == '_')?TRex_True:TRex_False;
	case 'W': return (!isalnum(c) && c != '_')?TRex_True:TRex_False;
	case 's': return isspace(c)?TRex_True:TRex_False;
	case 'S': return !isspace(c)?TRex_True:TRex_False;
	case 'd': return isdigit(c)?TRex_True:TRex_False;
	case 'D': return !isdigit(c)?TRex_True:TRex_False;
	case 'x': return isxdigit(c)?TRex_True:TRex_False;
	case 'X': return !isxdigit(c)?TRex_True:TRex_False;
	case 'c': return iscntrl(c)?TRex_True:TRex_False;
	case 'C': return !iscntrl(c)?TRex_True:TRex_False;
	case 'p': return ispunct(c)?TRex_True:TRex_False;
	case 'P': return !ispunct(c)?TRex_True:TRex_False;
	case 'l': return islower(c)?TRex_True:TRex_False;
	case 'u': return isupper(c)?TRex_True:TRex_False;
	}
	return TRex_False; /*cannot happen*/
}

static TRexBool trex_matchclass(TRex* exp,TRexNode *node,TRexChar c)
{
	do {
		switch(node->type) {
			case OP_RANGE:
				if(c >= node->left && c <= node->right) return TRex_True;
				break;
			case OP_CCLASS:
				if(trex_matchcclass(node->left,c)) return TRex_True;
				break;
			default:
				if(c == node->type)return TRex_True;
		}
	} while((node->next != -1) && (node = &exp->_nodes[node->next]));
	return TRex_False;
}

static const TRexChar *trex_matchnode(TRex* exp,TRexNode *node,const TRexChar *str,TRexNode *next)
{
	
	TRexNodeType type = node->type;
	switch(type) {
	case OP_GREEDY: {
		//TRexNode *greedystop = (node->next != -1) ? &exp->_nodes[node->next] : NULL;
		TRexNode *greedystop = NULL;
		int p0 = (node->right >> 16)&0x0000FFFF, p1 = node->right&0x0000FFFF, nmaches = 0;
		const TRexChar *s=str, *good = str;

		if(node->next != -1) {
			greedystop = &exp->_nodes[node->next];
		}
		else {
			greedystop = next;
		}

		while((nmaches == 0xFFFF || nmaches < p1)) {

			const TRexChar *stop;
			if(!(s = trex_matchnode(exp,&exp->_nodes[node->left],s,greedystop)))
				break;
			nmaches++;
			good=s;
			if(greedystop) {
				//checks that 0 matches satisfy the expression(if so skips)
				//if not would always stop(for instance if is a '?')
				if(greedystop->type != OP_GREEDY ||
				(greedystop->type == OP_GREEDY && ((greedystop->right >> 16)&0x0000FFFF) != 0))
				{
					TRexNode *gnext = NULL;
					if(greedystop->next != -1) {
						gnext = &exp->_nodes[greedystop->next];
					}else if(next && next->next != -1){
						gnext = &exp->_nodes[next->next];
					}
					stop = trex_matchnode(exp,greedystop,s,gnext);
					if(stop) {
						//if satisfied stop it
						if(p0 == p1 && p0 == nmaches) break;
						else if(nmaches >= p0 && p1 == 0xFFFF) break;
						else if(nmaches >= p0 && nmaches <= p1) break;
					}
				}
			}
			
			if(s >= exp->_eol)
				break;
		}
		if(p0 == p1 && p0 == nmaches) return good;
		else if(nmaches >= p0 && p1 == 0xFFFF) return good;
		else if(nmaches >= p0 && nmaches <= p1) return good;
		return NULL;
	}
	case OP_OR: {
			const TRexChar *asd = str;
			TRexNode *temp=&exp->_nodes[node->left];
			while( (asd = trex_matchnode(exp,temp,asd,NULL)) ) {
				if(temp->next != -1)
					temp = &exp->_nodes[temp->next];
				else
					return asd;
			}
			asd = str;
			temp = &exp->_nodes[node->right];
			while( (asd = trex_matchnode(exp,temp,asd,NULL)) ) {
				if(temp->next != -1)
					temp = &exp->_nodes[temp->next];
				else
					return asd;
			}
			return NULL;
			break;
	}
	case OP_EXPR:
	case OP_NOCAPEXPR:{
			TRexNode *n = &exp->_nodes[node->left];
			const TRexChar *cur = str;
			int capture = -1;
			if(node->type != OP_NOCAPEXPR && node->right == exp->_currsubexp) {
				capture = exp->_currsubexp;
				exp->_matches[capture].begin = cur;
				exp->_currsubexp++;
			}
			
			do {
				TRexNode *subnext = NULL;
				if(n->next != -1) {
					subnext = &exp->_nodes[n->next];
				}else {
					subnext = next;
				}
				if(!(cur = trex_matchnode(exp,n,cur,subnext))) {
					if(capture != -1){
						exp->_matches[capture].begin = 0;
						exp->_matches[capture].len = 0;
					}
					return NULL;
				}
			} while((n->next != -1) && (n = &exp->_nodes[n->next]));

			if(capture != -1) 
				exp->_matches[capture].len = cur - exp->_matches[capture].begin;
			return cur;
	}				 
	case OP_WB:
		if((str == exp->_bol && !isspace(*str))
		 || (str == exp->_eol && !isspace(*(str-1)))
		 || (!isspace(*str) && isspace(*(str+1)))
		 || (isspace(*str) && !isspace(*(str+1))) ) {
			return (node->left == 'b')?str:NULL;
		}
		return (node->left == 'b')?NULL:str;
	case OP_BOL:
		if(str == exp->_bol) return str;
		return NULL;
	case OP_EOL:
		if(str == exp->_eol) return str;
		return NULL;
	case OP_DOT:{
		*str++;
				}
		return str;
	case OP_NCLASS:
	case OP_CLASS:
		if(trex_matchclass(exp,&exp->_nodes[node->left],*str)?(type == OP_CLASS?TRex_True:TRex_False):(type == OP_NCLASS?TRex_True:TRex_False)) {
			*str++;
			return str;
		}
		return NULL;
	case OP_CCLASS:
		if(trex_matchcclass(node->left,*str)) {
			*str++;
			return str;
		}
		return NULL;
	default: /* char */
		if(*str != node->type) return NULL;
		*str++;
		return str;
	}
	return NULL;
}

/* public api */
TRex *trex_compile(const TRexChar *pattern,const TRexChar **error)
{
	TRex *exp = (TRex *)malloc(sizeof(TRex));
	exp->_eol = exp->_bol = NULL;
	exp->_p = pattern;
	exp->_nallocated = (int)scstrlen(pattern) * sizeof(TRexChar);
	exp->_nodes = (TRexNode *)malloc(exp->_nallocated * sizeof(TRexNode));
	exp->_nsize = 0;
	exp->_matches = 0;
	exp->_nsubexpr = 0;
	exp->_first = trex_newnode(exp,OP_EXPR);
	exp->_error = error;
	exp->_jmpbuf = malloc(sizeof(jmp_buf));
	if(setjmp(*((jmp_buf*)exp->_jmpbuf)) == 0) {
		int res = trex_list(exp);
		exp->_nodes[exp->_first].left = res;
		if(*exp->_p!='\0')
			trex_error(exp,_SC("unexpected character"));
#ifdef _DEBUG
		{
			int nsize,i;
			TRexNode *t;
			nsize = exp->_nsize;
			t = &exp->_nodes[0];
			scprintf(_SC("\n"));
			for(i = 0;i < nsize; i++) {
				if(exp->_nodes[i].type>MAX_CHAR)
					scprintf(_SC("[%02d] %10s "),i,g_nnames[exp->_nodes[i].type-MAX_CHAR]);
				else
					scprintf(_SC("[%02d] %10c "),i,exp->_nodes[i].type);
				scprintf(_SC("left %02d right %02d next %02d\n"),exp->_nodes[i].left,exp->_nodes[i].right,exp->_nodes[i].next);
			}
			scprintf(_SC("\n"));
		}
#endif
		exp->_matches = (TRexMatch *) malloc(exp->_nsubexpr * sizeof(TRexMatch));
		memset(exp->_matches,0,exp->_nsubexpr * sizeof(TRexMatch));
	}
	else{
		trex_free(exp);
		return NULL;
	}
	return exp;
}

void trex_free(TRex *exp)
{
	if(exp)	{
		if(exp->_nodes) free(exp->_nodes);
		if(exp->_jmpbuf) free(exp->_jmpbuf);
		if(exp->_matches) free(exp->_matches);
		free(exp);
	}
}

TRexBool trex_match(TRex* exp,const TRexChar* text)
{
	const TRexChar* res = NULL;
	exp->_bol = text;
	exp->_eol = text + scstrlen(text);
	exp->_currsubexp = 0;
	res = trex_matchnode(exp,exp->_nodes,text,NULL);
	if(res == NULL || res != exp->_eol)
		return TRex_False;
	return TRex_True;
}

TRexBool trex_searchrange(TRex* exp,const TRexChar* text_begin,const TRexChar* text_end,const TRexChar** out_begin, const TRexChar** out_end)
{
	const TRexChar *cur = NULL;
	int node = exp->_first;
	if(text_begin >= text_end) return TRex_False;
	exp->_bol = text_begin;
	exp->_eol = text_end;
	do {
		cur = text_begin;
		while(node != -1) {
			exp->_currsubexp = 0;
			cur = trex_matchnode(exp,&exp->_nodes[node],cur,NULL);
			if(!cur)
				break;
			node = exp->_nodes[node].next;
		}
		*text_begin++;
	} while(cur == NULL && text_begin != text_end);

	if(cur == NULL)
		return TRex_False;

	--text_begin;

	if(out_begin) *out_begin = text_begin;
	if(out_end) *out_end = cur;
	return TRex_True;
}

TRexBool trex_search(TRex* exp,const TRexChar* text, const TRexChar** out_begin, const TRexChar** out_end)
{
	return trex_searchrange(exp,text,text + scstrlen(text),out_begin,out_end);
}

int trex_getsubexpcount(TRex* exp)
{
	return exp->_nsubexpr;
}

TRexBool trex_getsubexp(TRex* exp, int n, TRexMatch *subexp)
{
	if( n<0 || n >= exp->_nsubexpr) return TRex_False;
	*subexp = exp->_matches[n];
	return TRex_True;
}

