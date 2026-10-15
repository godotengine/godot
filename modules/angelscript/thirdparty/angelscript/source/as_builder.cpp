/*
   AngelCode Scripting Library
   Copyright (c) 2003-2025 Andreas Jonsson

   This software is provided 'as-is', without any express or implied
   warranty. In no event will the authors be held liable for any
   damages arising from the use of this software.

   Permission is granted to anyone to use this software for any
   purpose, including commercial applications, and to alter it and
   redistribute it freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you
      must not claim that you wrote the original software. If you use
      this software in a product, an acknowledgment in the product
      documentation would be appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and
      must not be misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
      distribution.

   The original version of this library can be located at:
   http://www.angelcode.com/angelscript/

   Andreas Jonsson
   andreas@angelcode.com
*/


//
// as_builder.cpp
//
// This is the class that manages the compilation of the scripts
//


#include "as_config.h"
#include "as_builder.h"
#include "as_parser.h"
#include "as_compiler.h"
#include "as_tokendef.h"
#include "as_string_util.h"
#include "as_outputbuffer.h"
#include "as_texts.h"
#include "as_scriptobject.h"
#include "as_debug.h"

BEGIN_AS_NAMESPACE

#ifndef AS_NO_COMPILER

// asCSymbolTable template specializations for sGlobalVariableDescription entries
template<>
void asCSymbolTable<sGlobalVariableDescription>::GetKey(const sGlobalVariableDescription *entry, asSNameSpaceNamePair &key) const
{
	asSNameSpace *ns = entry->ns;
	asCString name = entry->name;
	key = asSNameSpaceNamePair(ns, name);
}

// Comparator for exact variable search
class asCCompGlobVarType : public asIFilter
{
public:
	const asCDataType &m_type;
	asCCompGlobVarType(const asCDataType &type) : m_type(type) {}

	bool operator()(const void *p) const
	{
		const sGlobalVariableDescription* desc = reinterpret_cast<const sGlobalVariableDescription*>(p);
		return desc->datatype == m_type;
	}

private:
	// The assignment operator is required for MSVC9, otherwise it will complain that it is not possible to auto generate the operator
	asCCompGlobVarType &operator=(const asCCompGlobVarType &) {return *this;}
};

#endif

asCBuilder::asCBuilder(asCScriptEngine *_engine, asCModule *_module)
{
	this->engine = _engine;
	this->module = _module;
	silent = false;
}

asCBuilder::~asCBuilder()
{
#ifndef AS_NO_COMPILER
	asUINT n;

	// Free all functions
	for( n = 0; n < functions.GetLength(); n++ )
	{
		if( functions[n] )
		{
			if( functions[n]->node )
				functions[n]->node->Destroy(engine);

			asDELETE(functions[n],sFunctionDescription);
		}

		functions[n] = 0;
	}

	// Free all global variables
	CleanupEnumValues();
	asCSymbolTable<sGlobalVariableDescription>::iterator it = globVariables.List();
	while( it )
	{
		if( (*it)->declaredAtNode )
			(*it)->declaredAtNode->Destroy(engine);
		if( (*it)->initializationNode )
			(*it)->initializationNode->Destroy(engine);
		asDELETE((*it),sGlobalVariableDescription);
		it++;
	}
	globVariables.Clear();

	// Free all the loaded files
	for( n = 0; n < scripts.GetLength(); n++ )
	{
		if( scripts[n] )
			asDELETE(scripts[n],asCScriptCode);

		scripts[n] = 0;
	}

	// Free all class declarations
	for( n = 0; n < classDeclarations.GetLength(); n++ )
	{
		if( classDeclarations[n] )
		{
			if( classDeclarations[n]->node )
				classDeclarations[n]->node->Destroy(engine);

			asDELETE(classDeclarations[n],sClassDeclaration);
			classDeclarations[n] = 0;
		}
	}

	for( n = 0; n < interfaceDeclarations.GetLength(); n++ )
	{
		if( interfaceDeclarations[n] )
		{
			if( interfaceDeclarations[n]->node )
				interfaceDeclarations[n]->node->Destroy(engine);

			asDELETE(interfaceDeclarations[n],sClassDeclaration);
			interfaceDeclarations[n] = 0;
		}
	}

	for( n = 0; n < namedTypeDeclarations.GetLength(); n++ )
	{
		if( namedTypeDeclarations[n] )
		{
			if( namedTypeDeclarations[n]->node )
				namedTypeDeclarations[n]->node->Destroy(engine);

			asDELETE(namedTypeDeclarations[n],sClassDeclaration);
			namedTypeDeclarations[n] = 0;
		}
	}

	for( n = 0; n < funcDefs.GetLength(); n++ )
	{
		if( funcDefs[n] )
		{
			if( funcDefs[n]->node )
				funcDefs[n]->node->Destroy(engine);

			asDELETE(funcDefs[n],sFuncDef);
			funcDefs[n] = 0;
		}
	}

	for( n = 0; n < mixinClasses.GetLength(); n++ )
	{
		if( mixinClasses[n] )
		{
			if( mixinClasses[n]->node )
				mixinClasses[n]->node->Destroy(engine);

			asDELETE(mixinClasses[n],sMixinClass);
			mixinClasses[n] = 0;
		}
	}

#endif // AS_NO_COMPILER
}

void asCBuilder::Reset()
{
	numErrors = 0;
	numWarnings = 0;
	engine->preMessage.isSet = false;

#ifndef AS_NO_COMPILER
	// Clear the cache of known types
	hasCachedKnownTypes = false;
	knownTypes.EraseAll();
#endif
}

#ifndef AS_NO_COMPILER
int asCBuilder::AddCode(const char *name, const char *code, int codeLength, int lineOffset, int sectionIdx, bool makeCopy)
{
	asCScriptCode *script = asNEW(asCScriptCode);
	if( script == 0 )
		return asOUT_OF_MEMORY;

	int r = script->SetCode(name, code, codeLength, makeCopy);
	if( r < 0 )
	{
		asDELETE(script, asCScriptCode);
		return r;
	}

	script->lineOffset = lineOffset;
	script->idx = sectionIdx;
	scripts.PushLast(script);

	return 0;
}

asCScriptCode *asCBuilder::FindOrAddCode(const char *name, const char *code, size_t length)
{
	for (asUINT n = 0; n < scripts.GetLength(); n++)
		if( scripts[n]->name == name && scripts[n]->codeLength == length && memcmp(scripts[n]->code, code, length) == 0 )
			return scripts[n];

	asCScriptCode *script = asNEW(asCScriptCode);
	if (script == 0)
		return 0;

	int r = script->SetCode(name, code, length, true);
	if (r < 0)
	{
		asDELETE(script, asCScriptCode);
		return 0;
	}

	script->idx = engine->GetScriptSectionNameIndex(name);
	scripts.PushLast(script);
	return script;
}

void asCBuilder::EvaluateTemplateInstances(asUINT startIdx, bool keepSilent)
{
	// Backup the original message stream
	bool                       msgCallback     = engine->msgCallback;
	asSSystemFunctionInterface msgCallbackFunc = engine->msgCallbackFunc;
	void                      *msgCallbackObj  = engine->msgCallbackObj;

	// Set the new temporary message stream
	asCOutputBuffer outBuffer;
	if( keepSilent )
		engine->SetMessageCallback(asFUNCTION(asCOutputBuffer::CDeclCallback), &outBuffer, asCALL_CDECL_OBJLAST);

	// Evaluate each of the template instances that have been created since the start of the build
	// TODO: This is not exactly correct, since another thread may have created template instances in parallel
	for( asUINT n = startIdx; n < engine->templateInstanceTypes.GetLength(); n++ )
	{
		bool dontGarbageCollect = false;
		asCObjectType *tmpl = engine->templateInstanceTypes[n];
		asCScriptFunction *callback = engine->scriptFunctions[tmpl->beh.templateCallback];
		if( callback && !engine->CallGlobalFunctionRetBool(tmpl, &dontGarbageCollect, callback->sysFuncIntf, callback) )
		{
			asCString sub = tmpl->templateSubTypes[0].Format(engine->nameSpaces[0]);
			for( asUINT m = 1; m < tmpl->templateSubTypes.GetLength(); m++ )
			{
				sub += ",";
				sub += tmpl->templateSubTypes[m].Format(engine->nameSpaces[0]);
			}
			asCString str;
			str.Format(TXT_INSTANCING_INVLD_TMPL_TYPE_s_s, tmpl->name.AddressOf(), sub.AddressOf());
			WriteError(tmpl->scriptSectionIdx >= 0 ? engine->scriptSectionNames[tmpl->scriptSectionIdx]->AddressOf() : "", str, tmpl->declaredAt&0xFFFFF, (tmpl->declaredAt>>20)&0xFFF);
		}
		else
		{
			// If the callback said this template instance won't be garbage collected then remove the flag
			if( dontGarbageCollect )
				tmpl->flags &= ~asOBJ_GC;
		}
	}

	// Restore message callback
	if( keepSilent )
	{
		engine->msgCallback     = msgCallback;
		engine->msgCallbackFunc = msgCallbackFunc;
		engine->msgCallbackObj  = msgCallbackObj;
	}
}

int asCBuilder::Build()
{
	Reset();

	// The template callbacks must only be called after the subtypes have a known structure,
	// otherwise the callback may think it is not possible to create the template instance,
	// even though it is.
	// TODO: This flag shouldn't be set globally in the engine, as it would mean that another
	//       thread requesting a template instance in parallel to the compilation wouldn't
	//       evaluate the template instance.
	engine->deferValidationOfTemplateTypes = true;
	asUINT numTempl = (asUINT)engine->templateInstanceTypes.GetLength();

	ParseScripts();
	if (numErrors > 0)
		return asERROR;

	// Compile the types first
	CompileInterfaces();
	CompileClasses(numTempl);

	// Evaluate the template instances one last time, this time with error messages, as we know
	// all classes have been fully built and it is known which ones will need garbage collection.
	EvaluateTemplateInstances(numTempl, false);
	engine->deferValidationOfTemplateTypes = false;
	if (numErrors > 0)
		return asERROR;

	// Then the global variables. Here the variables declared with auto
	// will be resolved, so they can be accessed properly in the functions
	CompileGlobalVariables();

	// Finally the global functions and class methods
	CompileFunctions();

	// TODO: Attempt to reorder the initialization of global variables so that
	//       they do not access other uninitialized global variables out-of-order
	//       The builder needs to check for each of the global variable, what functions
	//       that are accessed, and what global variables are access by these functions.

	if( numWarnings > 0 && engine->ep.compilerWarnings == 2 )
		WriteError(TXT_WARNINGS_TREATED_AS_ERROR, 0, 0);

	if( numErrors > 0 )
		return asERROR;

	// Make sure something was compiled, otherwise return an error
	if( module->IsEmpty() )
	{
		WriteError(TXT_NOTHING_WAS_BUILT, 0, 0);
		return asERROR;
	}

	return asSUCCESS;
}

int asCBuilder::CompileGlobalVar(const char *sectionName, const char *code, int lineOffset)
{
	Reset();

	// Add the string to the script code
	asCScriptCode *script = asNEW(asCScriptCode);
	if( script == 0 )
		return asOUT_OF_MEMORY;

	script->SetCode(sectionName, code, true);
	script->lineOffset = lineOffset;
	script->idx = engine->GetScriptSectionNameIndex(sectionName ? sectionName : "");
	scripts.PushLast(script);

	// Parse the string
	asCParser parser(this);
	if( parser.ParseScript(scripts[0]) < 0 )
		return asERROR;

	asCScriptNode *node = parser.GetScriptNode();

	// Make sure there is nothing else than the global variable in the script code
	if( node == 0 ||
		node->firstChild == 0 ||
		node->firstChild != node->lastChild ||
		node->firstChild->nodeType != snDeclaration )
	{
		WriteError(TXT_ONLY_ONE_VARIABLE_ALLOWED, script, 0);
		return asERROR;
	}

	node = node->firstChild;
	node->DisconnectParent();
	RegisterGlobalVar(node, script, module->m_defaultNamespace);

	CompileGlobalVariables();

	// It is possible that the global variable initialization included anonymous functions that must be compiled too
	for( asUINT n = 0; n < functions.GetLength(); n++ )
	{
		asCCompiler compiler(engine);
		asCScriptFunction *func = engine->scriptFunctions[functions[n]->funcId];
		int r = compiler.CompileFunction(this, functions[n]->script, func->parameterNames, functions[n]->node, func, 0);
		if( r < 0 )
			break;
	}

	if( numWarnings > 0 && engine->ep.compilerWarnings == 2 )
		WriteError(TXT_WARNINGS_TREATED_AS_ERROR, 0, 0);

	// None of the functions should be added to the module if any error occurred,
	// or it was requested that the functions wouldn't be added to the scope
	if( numErrors > 0 )
	{
		for( asUINT n = 0; n < functions.GetLength(); n++ )
		{
			asCScriptFunction *func = engine->scriptFunctions[functions[n]->funcId];
			if( module->m_globalFunctions.GetIndex(func) >= 0 )
			{
				module->m_globalFunctions.Erase(module->m_globalFunctions.GetIndex(func));
				module->m_scriptFunctions.RemoveValue(func);
				func->ReleaseInternal();
			}
		}
	}

	if( numErrors > 0 )
	{
		// Remove the variable from the module, if it was registered
		if( globVariables.GetSize() > 0 )
			module->RemoveGlobalVar(module->GetGlobalVarCount()-1);

		return asERROR;
	}

	return 0;
}
#endif

int asCBuilder::ValidateDefaultArgs(asCScriptCode *script, asCScriptNode *node, asCScriptFunction *func)
{
	int firstArgWithDefaultValue = -1;
	for( asUINT n = 0; n < func->defaultArgs.GetLength(); n++ )
	{
		if( func->defaultArgs[n] )
			firstArgWithDefaultValue = n;
		else if( firstArgWithDefaultValue >= 0 )
		{
			asCString str;
			str.Format(TXT_DEF_ARG_MISSING_IN_FUNC_s, func->GetDeclaration());
			WriteError(str, script, node);
			return asINVALID_DECLARATION;
		}
	}

	return 0;
}

#ifndef AS_NO_COMPILER
// This function will verify if the newly created function will conflict another overload due to having
// identical function arguments that are not default args, e.g: foo(int) and foo(int, int=0)
int asCBuilder::CheckForConflictsDueToDefaultArgs(asCScriptCode *script, asCScriptNode *node, asCScriptFunction *func, asCObjectType *objType)
{
	// TODO: Implement for global functions too
	if( func->objectType == 0 || objType == 0 ) return 0;

	asCArray<int> funcs;
	GetObjectMethodDescriptions(func->name.AddressOf(), objType, funcs, false);
	for( asUINT n = 0; n < funcs.GetLength(); n++ )
	{
		asCScriptFunction *func2 = engine->scriptFunctions[funcs[n]];
		if( func == func2 )
			continue;

		if( func->IsReadOnly() != func2->IsReadOnly() )
			continue;

		bool match = true;
		asUINT p = 0;
		for( ; p < func->parameterTypes.GetLength() && p < func2->parameterTypes.GetLength(); p++ )
		{
			// Only verify until the first argument with default args
			if( (func->defaultArgs.GetLength() > p && func->defaultArgs[p]) ||
				(func2->defaultArgs.GetLength() > p && func2->defaultArgs[p]) )
				break;

			if( func->parameterTypes[p] != func2->parameterTypes[p] ||
				func->inOutFlags[p] != func2->inOutFlags[p] )
			{
				match = false;
				break;
			}
		}

		if( match )
		{
			if( !((p >= func->parameterTypes.GetLength() && p < func2->defaultArgs.GetLength() && func2->defaultArgs[p]) ||
				  (p >= func2->parameterTypes.GetLength() && p < func->defaultArgs.GetLength() && func->defaultArgs[p])) )
			{
				// The argument lists match for the full length of the shorter, but the next
				// argument on the longer does not have a default arg so there is no conflict
				match = false;
			}
		}

		if( match )
		{
			WriteWarning(TXT_OVERLOAD_CONFLICTS_DUE_TO_DEFAULT_ARGS, script, node);
			WriteInfo(func->GetDeclaration(), script, node);
			WriteInfo(func2->GetDeclaration(), script, node);
			break;
		}
	}

	return 0;
}

int asCBuilder::CompileFunction(const char *sectionName, const char *code, int lineOffset, asDWORD compileFlags, asCScriptFunction **outFunc)
{
	asASSERT(outFunc != 0);

	Reset();

	// Add the string to the script code
	asCScriptCode *script = asNEW(asCScriptCode);
	if( script == 0 )
		return asOUT_OF_MEMORY;

	script->SetCode(sectionName, code, true);
	script->lineOffset = lineOffset;
	script->idx = engine->GetScriptSectionNameIndex(sectionName ? sectionName : "");
	scripts.PushLast(script);

	// Parse the string
	asCParser parser(this);
	if( parser.ParseScript(scripts[0]) < 0 )
		return asERROR;

	asCScriptNode *node = parser.GetScriptNode();

	// Make sure there is nothing else than the function in the script code
	if( node == 0 ||
		node->firstChild == 0 ||
		node->firstChild != node->lastChild ||
		node->firstChild->nodeType != snFunction )
	{
		WriteError(TXT_ONLY_ONE_FUNCTION_ALLOWED, script, 0);
		return asERROR;
	}

	// Find the function node
	node = node->firstChild;

	// Create the function
	asSFunctionTraits funcTraits;
	asCScriptFunction *func = asNEW(asCScriptFunction)(engine, compileFlags & asCOMP_ADD_TO_MODULE ? module : 0, asFUNC_SCRIPT);
	if( func == 0 )
		return asOUT_OF_MEMORY;

	GetParsedFunctionDetails(node, scripts[0], 0, func->name, func->returnType, func->parameterNames, func->parameterTypes, func->inOutFlags, func->defaultArgs, funcTraits, module->m_defaultNamespace);
	func->id                           = engine->GetNextScriptFunctionId();
	func->scriptData->scriptSectionIdx = engine->GetScriptSectionNameIndex(sectionName ? sectionName : "");
	int row, col;
	scripts[0]->ConvertPosToRowCol(node->tokenPos, &row, &col);
	func->scriptData->declaredAt       = (row & 0xFFFFF)|((col & 0xFFF)<<20);
	func->nameSpace                    = module->m_defaultNamespace;

	// Make sure the default args are declared correctly
	int r = ValidateDefaultArgs(script, node, func);
	if( r < 0 )
	{
		func->ReleaseInternal();
		return asERROR;
	}

	// Tell the engine that the function exists already so the compiler can access it
	if( compileFlags & asCOMP_ADD_TO_MODULE )
	{
		r = CheckNameConflict(func->name.AddressOf(), node, scripts[0], module->m_defaultNamespace, false, false, false);
		if( r < 0 )
		{
			func->ReleaseInternal();
			return asERROR;
		}

		module->m_globalFunctions.Put(func);

		module->AddScriptFunction(func);
	}
	else
		engine->AddScriptFunction(func);

	// Fill in the function info for the builder too
	node->DisconnectParent();
	sFunctionDescription *funcDesc = asNEW(sFunctionDescription);
	if( funcDesc == 0 )
	{
		func->ReleaseInternal();
		return asOUT_OF_MEMORY;
	}

	functions.PushLast(funcDesc);
	funcDesc->script            = scripts[0];
	funcDesc->node              = node;
	funcDesc->name              = func->name;
	funcDesc->funcId            = func->id;
	funcDesc->paramNames        = func->parameterNames;
	funcDesc->isExistingShared  = false;

	// This must be done in a loop, as it is possible that additional functions get declared as lambda's in the code
	for( asUINT n = 0; n < functions.GetLength(); n++ )
	{
		asCCompiler compiler(engine);
		asCScriptFunction *f = engine->scriptFunctions[functions[n]->funcId];
		r = compiler.CompileFunction(this, functions[n]->script, f->parameterNames, functions[n]->node, f, 0);
		if( r < 0 )
			break;
	}

	if( numWarnings > 0 && engine->ep.compilerWarnings == 2 )
		WriteError(TXT_WARNINGS_TREATED_AS_ERROR, 0, 0);

	// None of the functions should be added to the module if any error occurred,
	// or it was requested that the functions wouldn't be added to the scope
	if( !(compileFlags & asCOMP_ADD_TO_MODULE) || numErrors > 0 )
	{
		for( asUINT n = 0; n < functions.GetLength(); n++ )
		{
			asCScriptFunction *f = engine->scriptFunctions[functions[n]->funcId];
			if( module->m_globalFunctions.GetIndex(f) >= 0 )
			{
				module->m_globalFunctions.Erase(module->m_globalFunctions.GetIndex(f));
				module->m_scriptFunctions.RemoveValue(f);
				f->ReleaseInternal();
			}
		}
	}

	if( numErrors > 0 )
	{
		// Release the function pointer that would otherwise be returned if no errors occured
		func->ReleaseInternal();

		return asERROR;
	}

	// Return the function
	*outFunc = func;

	return asSUCCESS;
}

void asCBuilder::ParseScripts()
{
	TimeIt("asCBuilder::ParseScripts");

	asCArray<asCParser*> parsers((int)scripts.GetLength());

	// Parse all the files as if they were one
	asUINT n = 0;
	for( n = 0; n < scripts.GetLength(); n++ )
	{
		asCParser *parser = asNEW(asCParser)(this);
		if( parser != 0 )
		{
			parsers.PushLast(parser);

			// Parse the script file
			parser->ParseScript(scripts[n]);
		}
	}

	if (numErrors == 0)
	{
		// Register namespace visibility
		for (n = 0; n < scripts.GetLength(); n++)
		{
			asCScriptNode* node = parsers[n]->GetScriptNode();
			RegisterNamespaceVisibility(node, scripts[n], engine->nameSpaces[0]);
		}
		
		// Find all type declarations
		for (n = 0; n < scripts.GetLength(); n++)
		{
			asCScriptNode *node = parsers[n]->GetScriptNode();
			RegisterTypesFromScript(node, scripts[n], engine->nameSpaces[0]);
		}

		// Before moving forward the builder must establish the relationship between types
		// so that a derived type can see the child types of the parent type.
		DetermineTypeRelations();

		// Complete function definitions (defining returntype and parameters)
		for( n = 0; n < funcDefs.GetLength(); n++ )
			CompleteFuncDef(funcDefs[n]);

		// Find other global nodes
		for (n = 0; n < scripts.GetLength(); n++)
		{
			// Find other global nodes
			asCScriptNode *node = parsers[n]->GetScriptNode();
			RegisterNonTypesFromScript(node, scripts[n], engine->nameSpaces[0]);
		}

		// Register script methods found in the interfaces
		for( n = 0; n < interfaceDeclarations.GetLength(); n++ )
		{
			sClassDeclaration *decl = interfaceDeclarations[n];
			asCScriptNode *node = decl->node->firstChild->next;

			// Skip list of inherited interfaces
			while( node && node->nodeType == snIdentifier )
				node = node->next;

			while( node )
			{
				asCScriptNode *next = node->next;
				if( node->nodeType == snFunction )
				{
					node->DisconnectParent();
					RegisterScriptFunctionFromNode(node, decl->script, CastToObjectType(decl->typeInfo), true, false, 0, decl->isExistingShared);
				}
				else if( node->nodeType == snVirtualProperty )
				{
					node->DisconnectParent();
					RegisterVirtualProperty(node, decl->script, CastToObjectType(decl->typeInfo), true, false, 0, decl->isExistingShared);
				}

				node = next;
			}
		}

		// Register script methods found in the classes
		for( n = 0; n < classDeclarations.GetLength(); n++ )
		{
			sClassDeclaration *decl = classDeclarations[n];

			asCScriptNode *node = decl->node->firstChild->next;

			// Skip list of classes and interfaces
			while( node && node->nodeType == snIdentifier )
				node = node->next;

			while( node )
			{
				asCScriptNode *next = node->next;
				if( node->nodeType == snFunction )
				{
					node->DisconnectParent();
					RegisterScriptFunctionFromNode(node, decl->script, CastToObjectType(decl->typeInfo), false, false, 0, decl->isExistingShared, false, decl);
				}
				else if( node->nodeType == snVirtualProperty )
				{
					node->DisconnectParent();
					RegisterVirtualProperty(node, decl->script, CastToObjectType(decl->typeInfo), false, false, 0, decl->isExistingShared);
				}

				node = next;
			}

			if (!decl->isExistingShared)
			{
				// Add the default copy operator if needed (only if no other opAssign with single parameter is defined)
				bool copyOperatorExists = false;
				asCObjectType* ot = CastToObjectType(decl->typeInfo);
				for (asUINT i = 0; i < ot->methods.GetLength(); i++)
				{
					asCScriptFunction* f = engine->scriptFunctions[ot->methods[i]];
					if (f->name == "opAssign" && f->parameterTypes.GetLength() == 1)
					{
						copyOperatorExists = true;
						break;
					}
				}
				if (engine->ep.alwaysImplDefaultCopy == 2 || decl->isDefaultCopyDeleted ||
					(copyOperatorExists && ot->beh.copy == engine->scriptTypeBehaviours.beh.copy && engine->ep.alwaysImplDefaultCopy == 0))
				{
					// Script class has a declared constructor, so remove the default opAssign
					// unless the engine is configured to always provide a default opAssign
					engine->scriptFunctions[ot->beh.copy]->ReleaseInternal();
					ot->beh.copy = 0;
				}

				// Add the default constructors if needed (only if no other constructor is explicitly defined)
				if ( !decl->isDefaultConstructorDeleted && 
					 ((engine->ep.alwaysImplDefaultConstruct == 0 && ot->beh.construct == engine->scriptTypeBehaviours.beh.construct && ot->beh.constructors.GetLength() == 1) ||
					 engine->ep.alwaysImplDefaultConstruct == 1) )
				{
					AddDefaultConstructor(ot, decl->script);
				}

				// Add the default copy constructor if needed (only if no other constructor with single parameter is explicitly defined)
				bool copyConstructExists = false;
				for (asUINT i = 0; i < ot->beh.constructors.GetLength(); i++)
				{
					if (engine->scriptFunctions[ot->beh.constructors[i]]->parameterTypes.GetLength() == 1)
					{
						copyConstructExists = true;
						break;
					}
				}
				if ( !decl->isDefaultCopyConstructorDeleted &&
					 ((engine->ep.alwaysImplDefaultCopyConstruct == 0 && !copyConstructExists) || engine->ep.alwaysImplDefaultCopyConstruct == 1) )
					AddDefaultCopyConstructor(ot, decl->script);

				// If the default constructor has not been generated now, then release the dummy 
				if (ot->beh.construct == engine->scriptTypeBehaviours.beh.construct)
				{
					engine->scriptFunctions[ot->beh.construct]->ReleaseInternal();
					ot->beh.construct = 0;
					ot->beh.constructors.RemoveIndex(0);

					if (ot->beh.factory)
					{
						engine->scriptFunctions[ot->beh.factory]->ReleaseInternal();
						ot->beh.factory = 0;
						ot->beh.factories.RemoveIndex(0);
					}
				}
			}
		}
	}

	for( n = 0; n < parsers.GetLength(); n++ )
	{
		asDELETE(parsers[n],asCParser);
	}
}

void asCBuilder::RegisterTypesFromScript(asCScriptNode *node, asCScriptCode *script, asSNameSpace *ns)
{
	asASSERT(node->nodeType == snScript);

	// Find structure definitions first
	node = node->firstChild;
	while( node )
	{
		asCScriptNode *next = node->next;
		if( node->nodeType == snNamespace )
		{
			// Recursively register the entities defined in the namespace
			asCString nsName;
			nsName.Assign(&script->code[node->firstChild->tokenPos], node->firstChild->tokenLength);
			if( ns->name != "" )
				nsName = ns->name + "::" + nsName;

			asSNameSpace *nsChild = engine->AddNameSpace(nsName.AddressOf());
			RegisterTypesFromScript(node->lastChild, script, nsChild);
		}
		else
		{
			if( node->nodeType == snClass )
			{
				node->DisconnectParent();
				RegisterClass(node, script, ns);
			}
			else if( node->nodeType == snInterface )
			{
				node->DisconnectParent();
				RegisterInterface(node, script, ns);
			}
			else if( node->nodeType == snEnum )
			{
				node->DisconnectParent();
				RegisterEnum(node, script, ns);
			}
			else if( node->nodeType == snTypedef )
			{
				node->DisconnectParent();
				RegisterTypedef(node, script, ns);
			}
			else if( node->nodeType == snFuncDef )
			{
				node->DisconnectParent();
				RegisterFuncDef(node, script, ns, 0);
			}
			else if( node->nodeType == snMixin )
			{
				node->DisconnectParent();
				RegisterMixinClass(node, script, ns);
			}
		}

		node = next;
	}
}

void asCBuilder::RegisterNamespaceVisibility(asCScriptNode* node, asCScriptCode* script, asSNameSpace* ns)
{
	asASSERT(node->nodeType == snScript);

	node = node->firstChild;
	// First, register all namespaces
	while ( node )
	{
		asCScriptNode* next = node->next;
		if ( node->nodeType == snNamespace )
		{
			asCString nsName;
			nsName.Assign(&script->code[node->firstChild->tokenPos], node->firstChild->tokenLength);
			if ( ns->name != "" )
				nsName = ns->name + "::" + nsName;

			asSNameSpace* nsChild = engine->AddNameSpace(nsName.AddressOf());
			RegisterNamespaceVisibility(node->lastChild, script, nsChild);
		}
		else if ( node->nodeType == snUsing )
		{
			node->DisconnectParent();
			RegisterUsingNamespace(node, script, ns);
		}
		node = next;
	}
}
#endif

void asCBuilder::AddVisibleNamespaces(asSNameSpace *ns, const asCArray<asSNameSpace*>& visited, asCArray<asSNameSpace*>& pending)
{
	asSMapNode<asSNameSpace*, asCArray<asSNameSpace*>>* cursor = 0;

	if (namespaceVisibility.MoveTo(&cursor, ns))
	{
		asCArray<asSNameSpace*>& visibilityBuffer = namespaceVisibility.GetValue(cursor);
		asUINT count = visibilityBuffer.GetLength();

		for (asUINT i = 0; i < count; ++i)
		{
			asSNameSpace* visibleNameSpace = visibilityBuffer[i];
			if (!visited.Exists(visibleNameSpace) && !pending.Exists(visibleNameSpace))
			{
				pending.PushLast(visibleNameSpace);
			}
		}
	}
}

asSNameSpace *asCBuilder::FindNextVisibleNamespace(const asCArray<asSNameSpace*>& visited, asCArray<asSNameSpace*>& pending, asSNameSpace *parentNs, bool* checkAmbiguous)
{
	// First iterate through all pending namespaces that have been included with 'using namespace'
	asSNameSpace* nextNs = 0;
	while ( pending.GetLength() != 0 )
	{
		asSNameSpace *candidate = pending.PopLast();
		if (!visited.Exists(candidate))
		{
			nextNs = candidate;
			break;
		}
	}

	// Even if the parent namespace has already been visited, it still needs to be traversed
	if (nextNs == 0)
		nextNs = parentNs;

	if (checkAmbiguous && !(*checkAmbiguous))
		*checkAmbiguous = nextNs == engine->nameSpaces[0];

	return nextNs;
}

#ifndef AS_NO_COMPILER
void asCBuilder::RegisterNonTypesFromScript(asCScriptNode *node, asCScriptCode *script, asSNameSpace *ns)
{
	node = node->firstChild;
	while( node )
	{
		asCScriptNode *next = node->next;
		if( node->nodeType == snNamespace )
		{
			// Determine the name of the namespace
			asCString nsName;
			nsName.Assign(&script->code[node->firstChild->tokenPos], node->firstChild->tokenLength);
			if( ns->name != "" )
				nsName = ns->name + "::" + nsName;

			// Declare the namespace, then add the entities
			asSNameSpace *nsChild = engine->AddNameSpace(nsName.AddressOf());
			RegisterNonTypesFromScript(node->lastChild, script, nsChild);
		}
		else
		{
			node->DisconnectParent();
			if( node->nodeType == snFunction )
				RegisterScriptFunctionFromNode(node, script, 0, false, true, ns);
			else if( node->nodeType == snDeclaration )
				RegisterGlobalVar(node, script, ns);
			else if( node->nodeType == snVirtualProperty )
				RegisterVirtualProperty(node, script, 0, false, true, ns);
			else if( node->nodeType == snImport )
				RegisterImportedFunction(module->GetNextImportedFunctionId(), node, script, ns);
			else
			{
				// Unused script node
				int r, c;
				script->ConvertPosToRowCol(node->tokenPos, &r, &c);

				WriteWarning(script->name, TXT_UNUSED_SCRIPT_NODE, r, c);

				node->Destroy(engine);
			}
		}

		node = next;
	}
}

void asCBuilder::CompileFunctions()
{
	// Compile each function
	for( asUINT n = 0; n < functions.GetLength(); n++ )
	{
		sFunctionDescription *current = functions[n];
		if( current == 0 ) continue;

		// Don't compile the function again if it was an existing shared function
		if( current->isExistingShared ) continue;

		// Don't compile if there is no statement block
		if (current->node && !(current->node->nodeType == snStatementBlock || current->node->lastChild->nodeType == snStatementBlock))
			continue;

		asCCompiler compiler(engine);
		asCScriptFunction *func = engine->scriptFunctions[current->funcId];

		// Find the class declaration for constructors
		sClassDeclaration *classDecl = 0;
		if( current->objType && current->name == current->objType->name )
		{
			for( asUINT c = 0; c < classDeclarations.GetLength(); c++ )
			{
				if( classDeclarations[c]->typeInfo == current->objType )
				{
					classDecl = classDeclarations[c];
					break;
				}
			}

			asASSERT( classDecl );
		}

		if( current->node )
		{
			int r, c;
			current->script->ConvertPosToRowCol(current->node->tokenPos, &r, &c);

			asCString str = func->GetDeclarationStr();
			str.Format(TXT_COMPILING_s, str.AddressOf());
			WriteInfo(current->script->name, str, r, c, true);

			// When compiling a constructor need to pass the class declaration for member initializations
			compiler.CompileFunction(this, current->script, current->paramNames, current->node, func, classDecl);

			engine->preMessage.isSet = false;
		}
		else if( current->objType && current->name == current->objType->name )
		{
			asCScriptNode *node = classDecl ? classDecl->node : 0;

			if (func->parameterTypes.GetLength() == 0)
			{
				int r = 0, c = 0;
				if (node)
					current->script->ConvertPosToRowCol(node->tokenPos, &r, &c);

				asCString str = func->GetDeclarationStr();
				str.Format(TXT_COMPILING_AUTO_s, str.AddressOf());
				WriteInfo(current->script->name, str, r, c, true);

				// This is the default constructor that is generated
				// automatically if not implemented by the user.
				r = compiler.CompileDefaultConstructor(this, current->script, node, func, classDecl);
			}
			else
			{
				asASSERT(func->parameterTypes.GetLength() == 1 && func->parameterTypes[0].GetTypeInfo() == current->objType);

				int r = 0, c = 0;
				if (node)
					current->script->ConvertPosToRowCol(node->tokenPos, &r, &c);

				asCString str = func->GetDeclarationStr();
				str.Format(TXT_COMPILING_AUTO_s, str.AddressOf());
				WriteInfo(current->script->name, str, r, c, true);

				// This is the default copy constructor that is generated
				// automatically if not implemented by the user.
				r = compiler.CompileDefaultCopyConstructor(this, current->script, node, func, classDecl);
			}

			engine->preMessage.isSet = false;
		}
		else
		{
			asASSERT( false );
		}
	}
}
#endif

// Called from module and engine
int asCBuilder::ParseDataType(const char *datatype, asCDataType *result, asSNameSpace *implicitNamespace, bool isReturnType)
{
	Reset();

	asCScriptCode source;
	source.SetCode("", datatype, true);

	asCParser parser(this);
	int r = parser.ParseDataType(&source, isReturnType);
	if( r < 0 )
		return asINVALID_TYPE;

	// Get data type and property name
	asCScriptNode *dataType = parser.GetScriptNode()->firstChild;

	*result = CreateDataTypeFromNode(dataType, &source, implicitNamespace, true);
	if( isReturnType )
		*result = ModifyDataTypeFromNode(*result, dataType->next, &source, 0, 0);

	if( numErrors > 0 )
		return asINVALID_TYPE;

	return asSUCCESS;
}

int asCBuilder::ParseTemplateDecl(const char *decl, asCString *name, asCArray<asCString> &subtypeNames)
{
	Reset();

	asCScriptCode source;
	source.SetCode("", decl, true);

	asCParser parser(this);
	int r = parser.ParseTemplateDecl(&source);
	if( r < 0 )
		return asINVALID_TYPE;

	// Get the template name and subtype names
	asCScriptNode *node = parser.GetScriptNode()->firstChild;

	name->Assign(&decl[node->tokenPos], node->tokenLength);
	while( (node = node->next) != 0 )
	{
		asCString subtypeName;
		subtypeName.Assign(&decl[node->tokenPos], node->tokenLength);
		subtypeNames.PushLast(subtypeName);
	}

	// TODO: template: check for name conflicts

	if( numErrors > 0 )
		return asINVALID_DECLARATION;

	return asSUCCESS;
}

int asCBuilder::VerifyProperty(asCDataType *dt, const char *decl, asCString &name, asCDataType &type, asSNameSpace *ns)
{
	// Either datatype or namespace must be informed
	asASSERT( dt || ns );

	Reset();

	if( dt )
	{
		// Verify that the object type exist
		if( CastToObjectType(dt->GetTypeInfo()) == 0 )
			return asINVALID_OBJECT;
	}

	// Check property declaration and type
	asCScriptCode source;
	source.SetCode(TXT_PROPERTY, decl, true);

	asCParser parser(this);
	int r = parser.ParsePropertyDeclaration(&source);
	if( r < 0 )
		return asINVALID_DECLARATION;

	// Get data type
	asCScriptNode *dataType = parser.GetScriptNode()->firstChild;

	// Check if the property is declared 'by reference'
	bool isReference = (dataType->next->tokenType == ttAmp);

	// Get the name of the property
	asCScriptNode *nameNode = isReference ? dataType->next->next : dataType->next;

	// If an object property is registered, then use the
	// object's namespace, otherwise use the specified namespace
	type = CreateDataTypeFromNode(dataType, &source, dt ? dt->GetTypeInfo()->nameSpace : ns);
	name.Assign(&decl[nameNode->tokenPos], nameNode->tokenLength);
	type.MakeReference(isReference);

	// Validate that the type really can be a registered property
	// We cannot use CanBeInstantiated, as it is allowed to register
	// properties of type that cannot otherwise be instantiated
	if( type.IsFuncdef() && !type.IsObjectHandle() )
	{
		// Function definitions must always be handles
		return asINVALID_DECLARATION;
	}

	// Verify property name
	if( dt )
	{
		if( CheckNameConflictMember(dt->GetTypeInfo(), name.AddressOf(), nameNode, &source, true, false) < 0 )
			return asNAME_TAKEN;
	}
	else
	{
		if( CheckNameConflict(name.AddressOf(), nameNode, &source, ns, true, false, false) < 0 )
			return asNAME_TAKEN;
	}

	if( numErrors > 0 )
		return asINVALID_DECLARATION;

	return asSUCCESS;
}

#ifndef AS_NO_COMPILER
asCObjectProperty *asCBuilder::GetObjectProperty(asCDataType &obj, const char *prop)
{
	asASSERT(CastToObjectType(obj.GetTypeInfo()) != 0);

	// TODO: optimize: Improve linear search
	asCArray<asCObjectProperty *> &props = CastToObjectType(obj.GetTypeInfo())->properties;
	for( asUINT n = 0; n < props.GetLength(); n++ )
	{
		if( props[n]->name == prop )
		{
			if( module->m_accessMask & props[n]->accessMask )
				return props[n];
			else
				return 0;
		}
	}

	return 0;
}
#endif

bool asCBuilder::DoesGlobalPropertyExist(const char *prop, asSNameSpace *ns, asCGlobalProperty **outProp, sGlobalVariableDescription **outDesc, bool *isAppProp)
{
	if( outProp )   *outProp = 0;
	if( outDesc )   *outDesc = 0;
	if( isAppProp ) *isAppProp = false;

	// Check application registered properties
	asCString name(prop);
	asCGlobalProperty *globProp = engine->registeredGlobalProps.GetFirst(ns, name);
	if( globProp )
	{
		if( isAppProp ) *isAppProp = true;
		if( outProp )   *outProp   = globProp;
		return true;
	}

#ifndef AS_NO_COMPILER
	// Check properties being compiled now
	sGlobalVariableDescription* desc = globVariables.GetFirst(ns, prop);
	if( desc && !desc->isEnumValue )
	{
		if( outProp ) *outProp = desc->property;
		if( outDesc ) *outDesc = desc;
		return true;
	}
#endif

	// Check previously compiled global variables
	if( module )
	{
		globProp = module->m_scriptGlobals.GetFirst(ns, prop);
		if( globProp )
		{
			if( outProp ) *outProp = globProp;
			return true;
		}
	}

	return false;
}

asCGlobalProperty *asCBuilder::GetGlobalProperty(const char *prop, asSNameSpace *ns, bool *isCompiled, bool *isPureConstant, asQWORD *constantValue, bool *isAppProp)
{
	if( isCompiled )     *isCompiled     = true;
	if( isPureConstant ) *isPureConstant = false;
	if( isAppProp )      *isAppProp      = false;
	if( constantValue )  *constantValue  = 0;

	asCGlobalProperty          *globProp = 0;
	sGlobalVariableDescription *globDesc = 0;
	if( DoesGlobalPropertyExist(prop, ns, &globProp, &globDesc, isAppProp) )
	{
#ifndef AS_NO_COMPILER
		if( globDesc )
		{
			// The property was declared in this build call, check if it has been compiled successfully already
			if( isCompiled )     *isCompiled     = globDesc->isCompiled;
			if( isPureConstant ) *isPureConstant = globDesc->isPureConstant;
			if( constantValue  ) *constantValue  = globDesc->constantValue;
		}
		else
#endif
		if( isAppProp )
		{
			// Don't return the property if the module doesn't have access to it
			if( !(module->m_accessMask & globProp->accessMask) )
				globProp = 0;
		}
		return globProp;
	}

	return 0;
}

int asCBuilder::ParseFunctionDeclaration(asCObjectType *objType, const char *decl, asCScriptFunction *func, bool isSystemFunction, asCArray<bool> *paramAutoHandles, bool *returnAutoHandle, asSNameSpace *ns, asCScriptNode **listPattern, asCObjectType **outParentClass)
{
	asASSERT( objType || ns );

	if (listPattern)
		*listPattern = 0;
	if (outParentClass)
		*outParentClass = 0;

	// TODO: Can't we use GetParsedFunctionDetails to do most of what is done in this function?

	Reset();

	asCScriptCode source;
	source.SetCode(TXT_SYSTEM_FUNCTION, decl, true);

	asCParser parser(this);
	int r = parser.ParseFunctionDefinition(&source, listPattern != 0);
	if( r < 0 )
		return asINVALID_DECLARATION;

	asCScriptNode *node = parser.GetScriptNode();

	// Determine scope
	asCScriptNode *n = node->firstChild->next->next;
	asCObjectType *parentClass = 0;
	func->nameSpace = GetNameSpaceFromNode(n, &source, ns, &n, &parentClass);
	if( func->nameSpace == 0 && parentClass == 0 )
		return asINVALID_DECLARATION;
	if (parentClass && func->funcType != asFUNC_FUNCDEF)
		return asINVALID_DECLARATION;

	if (outParentClass)
		*outParentClass = parentClass;

	// Find name
	func->name.Assign(&source.code[n->tokenPos], n->tokenLength);

	// Handle templates
	asCScriptNode* tmp = n;
	bool isTemplate = false;
	while(tmp->next->nodeType != snParameterList)
	{
		asCString name(decl + tmp->next->tokenPos, tmp->next->tokenLength);
		asCTypeInfo* templSubType = engine->GetTemplateSubTypeByName(name);
		if (templSubType == 0)
			return asOUT_OF_MEMORY;

		func->templateSubTypes.PushLast(asCDataType::CreateType(templSubType,false));
		templSubType->AddRefInternal();
		isTemplate = true;
		tmp = tmp->next;
	}
	
	// Initialize a script function object for registration
	bool autoHandle;

	// Scoped reference types are allowed to use handle when returned from application functions
	func->returnType = CreateDataTypeFromNode(node->firstChild, &source, objType ? objType->nameSpace : ns, true, parentClass ? parentClass : objType, true, 0, isTemplate ? &func->templateSubTypes : 0);
	func->returnType = ModifyDataTypeFromNode(func->returnType, node->firstChild->next, &source, 0, &autoHandle);
		
	if( autoHandle && (!func->returnType.IsObjectHandle() || func->returnType.IsReference()) )
		return asINVALID_DECLARATION;
	if( returnAutoHandle ) *returnAutoHandle = autoHandle;

	// Reference types cannot be returned by value from system functions
	if( isSystemFunction &&
		(func->returnType.GetTypeInfo() &&
		 (func->returnType.GetTypeInfo()->flags & asOBJ_REF)) &&
		!(func->returnType.IsReference() ||
		  func->returnType.IsObjectHandle()) )
		return asINVALID_DECLARATION;

	// Count number of parameters
	int paramCount = 0;
	asCScriptNode *paramList = tmp->next;
	n = paramList->firstChild;
	while( n )
	{
		if (n && n->nodeType == snVariadic)
			break; // Should be the last argument
		
		paramCount++;
		n = n->next->next;
		if( n && n->nodeType == snIdentifier )
			n = n->next;

		if( n && n->nodeType == snExpression )
			n = n->next;
	}

	// Preallocate memory
	func->parameterTypes.Allocate(paramCount, false);
	func->parameterNames.SetLength(paramCount);
	func->inOutFlags.Allocate(paramCount, false);
	func->defaultArgs.Allocate(paramCount, false);
	if( paramAutoHandles ) paramAutoHandles->Allocate(paramCount, false);

	n = paramList->firstChild;
	asUINT index = 0;
	while( n )
	{
		asETypeModifiers inOutFlags;
		asCDataType type;
		type = CreateDataTypeFromNode(n, &source, objType ? objType->nameSpace : ns, false, parentClass ? parentClass : objType, true, 0, isTemplate ? &func->templateSubTypes : 0);
		type = ModifyDataTypeFromNode(type, n->next, &source, &inOutFlags, &autoHandle);

		// Reference types cannot be passed by value to system functions
		if( isSystemFunction &&
			(type.GetTypeInfo() &&
			 (type.GetTypeInfo()->flags & asOBJ_REF)) &&
			!(type.IsReference() ||
			  type.IsObjectHandle()) )
			return asINVALID_DECLARATION;

		// Store the parameter type
		func->parameterTypes.PushLast(type);
		func->inOutFlags.PushLast(inOutFlags);

		// Don't permit void parameters
		if( type.GetTokenType() == ttVoid )
			return asINVALID_DECLARATION;

		if( autoHandle && (!type.IsObjectHandle() || type.IsReference()) )
			return asINVALID_DECLARATION;

		if( paramAutoHandles ) paramAutoHandles->PushLast(autoHandle);

		// Make sure that var type parameters are references
		if( type.GetTokenType() == ttQuestion &&
			!type.IsReference() )
			return asINVALID_DECLARATION;

		// Move to next parameter
		n = n->next->next;
		if (n && n->nodeType == snVariadic)
		{
			func->SetVariadic(true);
			break;
		}
		if( n && n->nodeType == snIdentifier )
		{
			func->parameterNames[index] = asCString(&source.code[n->tokenPos], n->tokenLength);
			n = n->next;
		}
		++index;

		if( n && n->nodeType == snExpression )
		{
			// Strip out white space and comments to better share the string
			asCString *defaultArgStr = asNEW(asCString);
			if( defaultArgStr )
			{
				*defaultArgStr = GetCleanExpressionString(n, &source);
				func->defaultArgs.PushLast(defaultArgStr);
			}

			n = n->next;
		}
		else
			func->defaultArgs.PushLast(0);
	}

	// Set the read-only flag if const is declared after parameter list
	n = paramList->next;
	if( n && n->nodeType == snUndefined && n->tokenType == ttConst )
	{
		if( objType == 0 )
			return asINVALID_DECLARATION;
		func->SetReadOnly(true);

		n = n->next;
	}
	else
		func->SetReadOnly(false);

	// Check for additional function traits
	while (n && n->nodeType == snIdentifier)
	{
		if (source.TokenEquals(n->tokenPos, n->tokenLength, EXPLICIT_TOKEN))
			func->SetExplicit(true);
		else if( source.TokenEquals(n->tokenPos, n->tokenLength, PROPERTY_TOKEN))
			func->SetProperty(true);
		else
			return asINVALID_DECLARATION;

		n = n->next;
	}

	// If the caller expects a list pattern, check for the existence, else report an error if not
	if( listPattern )
	{
		if( n == 0 || n->nodeType != snListPattern )
			return asINVALID_DECLARATION;
		else
		{
			*listPattern = n;
			n->DisconnectParent();
		}
	}
	else
	{
		if( n )
			return asINVALID_DECLARATION;
	}

	// Make sure the default args are declared correctly
	ValidateDefaultArgs(&source, node, func);

	if( numErrors > 0 || numWarnings > 0 )
		return asINVALID_DECLARATION;

	return 0;
}

int asCBuilder::ParseVariableDeclaration(const char *decl, asSNameSpace *implicitNamespace, asCString &outName, asSNameSpace *&outNamespace, asCDataType &outDt)
{
	Reset();

	asCScriptCode source;
	source.SetCode(TXT_VARIABLE_DECL, decl, true);

	asCParser parser(this);

	int r = parser.ParsePropertyDeclaration(&source);
	if( r < 0 )
		return asINVALID_DECLARATION;

	asCScriptNode *node = parser.GetScriptNode();

	// Determine the scope from declaration
	asCScriptNode *n = node->firstChild->next;
	// TODO: child funcdef: The parentType will be set if the scope is actually a type rather than a namespace
	outNamespace = GetNameSpaceFromNode(n, &source, implicitNamespace, &n);
	if( outNamespace == 0 )
		return asINVALID_DECLARATION;

	// Find name
	outName.Assign(&source.code[n->tokenPos], n->tokenLength);

	// Initialize a script variable object for registration
	outDt = CreateDataTypeFromNode(node->firstChild, &source, implicitNamespace);

	if( numErrors > 0 || numWarnings > 0 )
		return asINVALID_DECLARATION;

	return 0;
}

// TODO: This should use SymbolLookupMember, which should be available in the TypeInfo class
int asCBuilder::CheckNameConflictMember(asCTypeInfo *t, const char *name, asCScriptNode *node, asCScriptCode *code, bool isProperty, bool isVirtualProperty)
{
	// It's not necessary to check against object types

	asCObjectType *ot = CastToObjectType(t);
	if (!ot)
		return 0;

	// Check against properties
	// TODO: optimize: Improve linear search
	// Properties are allowed to have the same name as virtual properties
	if( !isVirtualProperty )
	{
		asCArray<asCObjectProperty *> &props = ot->properties;
		for( asUINT n = 0; n < props.GetLength(); n++ )
		{
			if( props[n]->name == name )
			{
				if( code )
				{
					asCString str;
					str.Format(TXT_NAME_CONFLICT_s_OBJ_PROPERTY, name);
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

	// Check against virtual properties
	// Don't do this when the check is for a virtual property, as it is allowed to have multiple overloads for virtual properties
	// Properties are allowed to have the same name as virtual properties
	if( !isProperty && !isVirtualProperty )
	{
		asCArray<int> methods = ot->methods;
		for( asUINT n = 0; n < methods.GetLength(); n++ )
		{
			asCScriptFunction *func = engine->scriptFunctions[methods[n]];
			if( func->IsProperty() && func->name.SubString(4) == name )
			{
				if( code )
				{
					asCString str;
					str.Format(TXT_NAME_CONFLICT_s_OBJ_PROPERTY, name);
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

	// Check against child types
	asCArray<asCFuncdefType*> &funcdefs = ot->childFuncDefs;
	for (asUINT n = 0; n < funcdefs.GetLength(); n++)
	{
		if (funcdefs[n]->name == name)
		{
			if (code)
			{
				asCString str;
				str.Format(TXT_NAME_CONFLICT_s_IS_FUNCDEF, name);
				WriteError(str, code, node);
			}

			return -1;
		}
	}

	// Property names must be checked against method names
	if( isProperty )
	{
		asCArray<int> methods = ot->methods;
		for( asUINT n = 0; n < methods.GetLength(); n++ )
		{
			if( engine->scriptFunctions[methods[n]]->name == name )
			{
				if( code )
				{
					asCString str;
					str.Format(TXT_NAME_CONFLICT_s_METHOD, name);
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

	// If there is a namespace at the same level with the same name as the class, then need to check for conflicts with symbols in that namespace too
	// TODO: When classes can have static members, the code should change so that class name cannot be the same as a namespace
	asCString scope;
	if (ot->nameSpace->name != "")
		scope = ot->nameSpace->name + "::" + ot->name;
	else
		scope = ot->name;
	asSNameSpace *ns = engine->FindNameSpace(scope.AddressOf());
	if (ns)
	{
		// Check as if not a function as it doesn't matter the function signature
		return CheckNameConflict(name, node, code, ns, true, isVirtualProperty, false);
	}
	
	return 0;
}

// TODO: This should use SymbolLookup
int asCBuilder::CheckNameConflict(const char *name, asCScriptNode *node, asCScriptCode *code, asSNameSpace *ns, bool isProperty, bool isVirtualProperty, bool isSharedIntf)
{
	// Check against registered object types
	if( engine->GetRegisteredType(name, ns) != 0 )
	{
		if( code )
		{
			asCString str;
			if (ns->name != "")
				str = ns->name + "::" + name;
			else
				str = name;
			str.Format(TXT_NAME_CONFLICT_s_EXTENDED_TYPE, str.AddressOf());
			WriteError(str, code, node);
		}

		return -1;
	}

	// Check against global properties
	// Virtual properties are allowed to have the same name as a real property
	if( !isVirtualProperty && DoesGlobalPropertyExist(name, ns) )
	{
		if( code )
		{
			asCString str;
			if (ns->name != "")
				str = ns->name + "::" + name;
			else
				str = name;
			str.Format(TXT_NAME_CONFLICT_s_GLOBAL_PROPERTY, str.AddressOf());
			WriteError(str, code, node);
		}

		return -1;
	}
	
	// Check against registered global virtual properties
	// Don't do this when the check is for a virtual property, as it is allowed to have multiple overloads for virtual properties
	if( !isProperty || !isVirtualProperty )
	{
		for (asUINT n = 0; n < engine->registeredGlobalFuncs.GetSize(); n++)
		{
			asCScriptFunction *func = engine->registeredGlobalFuncs.Get(n);
			if (func->IsProperty() &&
				func->nameSpace == ns &&
				func->name.SubString(4) == name)
			{
				if (code)
				{
					asCString str;
					if (ns->name != "")
						str = ns->name + "::" + name;
					else
						str = name;
					str.Format(TXT_NAME_CONFLICT_s_IS_VIRTPROP, str.AddressOf());
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

	// Property names must be checked against function names
	if (isProperty)
	{
		for (asUINT n = 0; n < engine->registeredGlobalFuncs.GetSize(); n++)
		{
			if (engine->registeredGlobalFuncs.Get(n)->name == name &&
				engine->registeredGlobalFuncs.Get(n)->nameSpace == ns)
			{
				if (code)
				{
					asCString str;
					if (ns->name != "")
						str = ns->name + "::" + name;
					else
						str = name;
					str.Format(TXT_NAME_CONFLICT_s_IS_FUNCTION, str.AddressOf());
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

#ifndef AS_NO_COMPILER
	// Check against interface types
	asUINT n;
	for (n = 0; n < interfaceDeclarations.GetLength(); n++)
	{
		if (interfaceDeclarations[n]->name == name &&
			interfaceDeclarations[n]->typeInfo->nameSpace == ns)
		{
			// Don't report an error if the application wants to ignore duplicate declarations of shared interfaces
			if (!(isSharedIntf && engine->ep.ignoreDuplicateSharedIntf))
			{
				if (code)
				{
					asCString str;
					if (ns->name != "")
						str = ns->name + "::" + name;
					else
						str = name;
					str.Format(TXT_NAME_CONFLICT_s_INTF, str.AddressOf());
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}

	// Check against class types
	for( n = 0; n < classDeclarations.GetLength(); n++ )
	{
		if( classDeclarations[n]->name == name &&
			classDeclarations[n]->typeInfo->nameSpace == ns )
		{
			if( code )
			{
				asCString str;
				if (ns->name != "")
					str = ns->name + "::" + name;
				else
					str = name;
				str.Format(TXT_NAME_CONFLICT_s_STRUCT, str.AddressOf());
				WriteError(str, code, node);
			}

			return -1;
		}
	}

	// Check against named types
	for( n = 0; n < namedTypeDeclarations.GetLength(); n++ )
	{
		if( namedTypeDeclarations[n]->name == name &&
			namedTypeDeclarations[n]->typeInfo->nameSpace == ns )
		{
			if( code )
			{
				asCString str;
				if (ns->name != "")
					str = ns->name + "::" + name;
				else
					str = name;
				str.Format(TXT_NAME_CONFLICT_s_IS_NAMED_TYPE, str.AddressOf());
				WriteError(str, code, node);
			}

			return -1;
		}
	}

	// Must check for name conflicts with funcdefs
	for( n = 0; n < funcDefs.GetLength(); n++ )
	{
		if( funcDefs[n]->name == name &&
			module->m_funcDefs[funcDefs[n]->idx]->nameSpace == ns )
		{
			if( code )
			{
				asCString str;
				if (ns->name != "")
					str = ns->name + "::" + name;
				else
					str = name;
				str.Format(TXT_NAME_CONFLICT_s_IS_FUNCDEF, str.AddressOf());
				WriteError(str, code, node);
			}

			return -1;
		}
	}

	// Check against mixin classes
	if( GetMixinClass(name, ns) )
	{
		if( code )
		{
			asCString str;
			if (ns->name != "")
				str = ns->name + "::" + name;
			else
				str = name;
			str.Format(TXT_NAME_CONFLICT_s_IS_MIXIN, str.AddressOf());
			WriteError(str, code, node);
		}

		return -1;
	}

	// Check against virtual properties
	// Don't do this when the check is for a virtual property, as it is allowed to have multiple overloads for virtual properties
	if( !isProperty && !isVirtualProperty )
	{
		for (n = 0; n < functions.GetLength(); n++)
		{
			asCScriptFunction *func = engine->scriptFunctions[functions[n] ? functions[n]->funcId : 0];
			if (func && 
				func->IsProperty() &&
				func->objectType == 0 && 
				func->nameSpace == ns &&
				func->name.SubString(4) == name)
			{
				if (code)
				{
					asCString str;
					if (ns->name != "")
						str = ns->name + "::" + name;
					else
						str = name;
					str.Format(TXT_NAME_CONFLICT_s_IS_VIRTPROP, str.AddressOf());
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}
	
	// Property names must be checked against function names
	if (isProperty)
	{
		for (n = 0; n < functions.GetLength(); n++)
		{
			if (functions[n] && 
				functions[n]->objType == 0 && 
				functions[n]->name == name &&
				engine->scriptFunctions[functions[n]->funcId]->nameSpace == ns )
			{
				if (code)
				{
					asCString str;
					if (ns->name != "")
						str = ns->name + "::" + name;
					else
						str = name;
					str.Format(TXT_NAME_CONFLICT_s_IS_FUNCTION, str.AddressOf());
					WriteError(str, code, node);
				}

				return -1;
			}
		}
	}
#else
	UNUSED_VAR(isSharedIntf);
#endif

	return 0;
}

// Returns a negative value on invalid property
// -2 incorrect prefix
// -3 invalid signature
// -4 mismatching type for get/set
// -5 name conflict
int asCBuilder::ValidateVirtualProperty(asCScriptFunction *func)
{
	asASSERT( func->IsProperty() );
	
	// A virtual property must have the prefix "get_" or "set_"
	asCString prefix = func->name.SubString(0, 4);
	if( prefix != "get_" && prefix != "set_" )
		return -2;
	
	// A getter must return a non-void type and have at most 1 argument (indexed property)
	if( prefix == "get_" && (func->returnType == asCDataType::CreatePrimitive(ttVoid, false) || func->parameterTypes.GetLength() > 1) )
		return -3;
	
	// A setter must return a void and have 1 or 2 arguments (indexed property)
	if( prefix == "set_" && (func->returnType != asCDataType::CreatePrimitive(ttVoid, false) || func->parameterTypes.GetLength() < 1 || func->parameterTypes.GetLength() > 2) )
		return -3;
	
	// Check matching getter/setter. Registered application object methods are
	// already indexed by their owning type. Scanning every engine function for
	// each property makes large native interfaces quadratic in total functions.
	// Script declarations may not yet be in the type's method list: retain the
	// original scan for script types and globals, including all scope checks.
	asCObjectType *registeredOwner = func->funcType == asFUNC_SYSTEM && func->objectType &&
		!(func->objectType->flags & asOBJ_SCRIPT_OBJECT) ? func->objectType : 0;
	const asUINT candidateCount = registeredOwner ? registeredOwner->methods.GetLength() : engine->scriptFunctions.GetLength();
	asCDataType getType, setType;
	bool found = false;
	if( prefix == "get_" )
	{
		getType = func->returnType;
		
		// Find if there is a set accessor in the same scope, and then validate the type of it
		asCString setName = "set_" + func->name.SubString(4);
		for( asUINT n = 0; n < candidateCount; n++ )
		{
			asCScriptFunction *setFunc = engine->scriptFunctions[registeredOwner ? registeredOwner->methods[n] : n];
			if( setFunc == 0 || setFunc->name != setName || !setFunc->IsProperty() )
				continue;
			
			// Is it the same scope?
			if( func->module != setFunc->module || func->nameSpace != setFunc->nameSpace || func->objectType != setFunc->objectType )
				continue;
			
			setType = setFunc->parameterTypes[setFunc->parameterTypes.GetLength() - 1];
			found = true;
			break;
		}
	}
	else
	{
		setType = func->parameterTypes[func->parameterTypes.GetLength() - 1];
		
		// Find if there is a get accessor in the same scope and then validate the type of it
		asCString getName = "get_" + func->name.SubString(4);
		for( asUINT n = 0; n < candidateCount; n++ )
		{
			asCScriptFunction *getFunc = engine->scriptFunctions[registeredOwner ? registeredOwner->methods[n] : n];
			if( getFunc == 0 || getFunc->name != getName || !getFunc->IsProperty() )
				continue;
			
			// Is it the same scope?
			if( func->module != getFunc->module || func->nameSpace != getFunc->nameSpace || func->objectType != getFunc->objectType )
				continue;
			
			getType = getFunc->returnType;
			found = true;
			break;
		}
	}
	
	if( found )
	{
		// Check that the type matches
		// It is permitted for a getter to return a handle and the setter to take a reference
		if( !getType.IsEqualExceptRefAndConst(setType) &&
			!((getType.IsObjectHandle() && !setType.IsObjectHandle()) &&
			  (getType.GetTypeInfo() == setType.GetTypeInfo())) )
		{
			return -4;
		}
	}
	
	// Check name conflict with other entities in the same scope
	// It is allowed to have a real property of the same name, in which case the virtual property hides the real one.
	int r;
	if( func->objectType )
		r = CheckNameConflictMember(func->objectType, func->name.SubString(4).AddressOf(), 0, 0, true, true);
	else
		r = CheckNameConflict(func->name.SubString(4).AddressOf(), 0, 0, func->nameSpace, true, true, false);
	if( r < 0 )
		return -5;
	
	// Everything is OK
	return 0;
}

#ifndef AS_NO_COMPILER
sMixinClass *asCBuilder::GetMixinClass(const char *name, asSNameSpace *ns)
{
	for( asUINT n = 0; n < mixinClasses.GetLength(); n++ )
		if( mixinClasses[n]->name == name &&
			mixinClasses[n]->ns == ns )
			return mixinClasses[n];

	return 0;
}

int asCBuilder::RegisterFuncDef(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns, asCObjectType *parent)
{
	// namespace and parent are exclusively mutual
	asASSERT((ns == 0 && parent) || (ns && parent == 0));

	// Skip leading 'shared' and 'external' keywords
	asCScriptNode *n = node->firstChild;
	while (n->nodeType == snIdentifier)
		n = n->next;

	// Find the name
	asASSERT( n->nodeType == snDataType );
	n = n->next->next;

	asCString name;
	name.Assign(&file->code[n->tokenPos], n->tokenLength);

	// Check for name conflict with other types
	if (ns)
	{
		int r = CheckNameConflict(name.AddressOf(), node, file, ns, true, false, false);
		if (asSUCCESS != r)
		{
			node->Destroy(engine);
			return r;
		}
	}
	else
	{
		int r = CheckNameConflictMember(parent, name.AddressOf(), node, file, false, false);
		if (asSUCCESS != r)
		{
			node->Destroy(engine);
			return r;
		}
	}

	// The function definition should be stored as a asCScriptFunction so that the application
	// can use the asIScriptFunction interface to enumerate the return type and parameters

	// The return type and parameter types aren't determined in this function. A second pass is
	// necessary after all type declarations have been identified. The second pass is implemented
	// in CompleteFuncDef().

	sFuncDef *fd = asNEW(sFuncDef);
	if( fd == 0 )
	{
		node->Destroy(engine);
		return asOUT_OF_MEMORY;
	}

	fd->name   = name;
	fd->node   = node;
	fd->script = file;
	fd->idx    = module->AddFuncDef(name, ns, parent);

	funcDefs.PushLast(fd);

	return 0;
}

void asCBuilder::CompleteFuncDef(sFuncDef *funcDef)
{
	asCArray<asCString *>      defaultArgs;
	asSFunctionTraits          funcTraits;

	asCFuncdefType *fdt = module->m_funcDefs[funcDef->idx];
	asASSERT( fdt );
	asCScriptFunction *func = fdt->funcdef;

	asSNameSpace *implicitNs = func->nameSpace ? func->nameSpace : fdt->parentClass->nameSpace;
	GetParsedFunctionDetails(funcDef->node, funcDef->script, fdt->parentClass, funcDef->name, func->returnType, func->parameterNames, func->parameterTypes, func->inOutFlags, defaultArgs, funcTraits, implicitNs);

	// There should not be any defaultArgs, but if there are any we need to delete them to avoid leaks
	for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
		if( defaultArgs[n] )
			asDELETE(defaultArgs[n], asCString);

	// All funcdefs are shared, unless one of the parameter types or return type is not shared
	bool declaredShared = funcTraits.GetTrait(asTRAIT_SHARED);
	funcTraits.SetTrait(asTRAIT_SHARED, true);
	if (func->returnType.GetTypeInfo() && !func->returnType.GetTypeInfo()->IsShared())
	{
		if (declaredShared)
		{
			asCString s;
			s.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, func->returnType.GetTypeInfo()->name.AddressOf());
			WriteError(s.AddressOf(), funcDef->script, funcDef->node);
		}
		funcTraits.SetTrait(asTRAIT_SHARED, false);
	}
	for( asUINT n = 0; funcTraits.GetTrait(asTRAIT_SHARED) && n < func->parameterTypes.GetLength(); n++ )
		if (func->parameterTypes[n].GetTypeInfo() && !func->parameterTypes[n].GetTypeInfo()->IsShared())
		{
			if (declaredShared)
			{
				asCString s;
				s.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, func->parameterTypes[n].GetTypeInfo()->name.AddressOf());
				WriteError(s.AddressOf(), funcDef->script, funcDef->node);
			}
			funcTraits.SetTrait(asTRAIT_SHARED, false);
		}
	func->SetShared(funcTraits.GetTrait(asTRAIT_SHARED));

	// Check if there is another identical funcdef from another module and if so reuse that instead
	bool found = false;
	if( func->IsShared() )
	{
		for( asUINT n = 0; n < engine->funcDefs.GetLength(); n++ )
		{
			asCFuncdefType *fdt2 = engine->funcDefs[n];
			if( fdt2 == 0 || fdt == fdt2 )
				continue;

			if( !fdt2->funcdef->IsShared() )
				continue;

			if( fdt2->name == fdt->name &&
				fdt2->nameSpace == fdt->nameSpace &&
				fdt2->parentClass == fdt->parentClass &&
				fdt2->funcdef->IsSignatureExceptNameEqual(func) )
			{
				// Replace our funcdef for the existing one
				module->ReplaceFuncDef(fdt, fdt2);
				fdt2->AddRefInternal();

				engine->funcDefs.RemoveValue(fdt);

				fdt->ReleaseInternal();
				found = true;
				break;
			}
		}
	}

	// If the funcdef was declared as external then the existing shared declaration must have been found
	if (funcTraits.GetTrait(asTRAIT_EXTERNAL) && !found)
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_NOT_FOUND, funcDef->name.AddressOf());
		WriteError(str, funcDef->script, funcDef->node);
	}

	// Remember if the type was declared as external so the saved bytecode can be flagged accordingly
	if (funcTraits.GetTrait(asTRAIT_EXTERNAL) && found)
		module->m_externalTypes.PushLast(engine->scriptFunctions[module->m_funcDefs[funcDef->idx]->GetFuncdefSignature()->GetId()]->funcdefType);
}

int asCBuilder::RegisterGlobalVar(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	// Has the application disabled global vars?
	if( engine->ep.disallowGlobalVars )
		WriteError(TXT_GLOBAL_VARS_NOT_ALLOWED, file, node);

	// What data type is it?
	asCDataType type = CreateDataTypeFromNode(node->firstChild, file, ns);

	if( !type.CanBeInstantiated() )
	{
		asCString str;
		if( type.IsAbstractClass() )
			str.Format(TXT_ABSTRACT_CLASS_s_CANNOT_BE_INSTANTIATED, type.Format(ns).AddressOf());
		else if( type.IsInterface() )
			str.Format(TXT_INTERFACE_s_CANNOT_BE_INSTANTIATED, type.Format(ns).AddressOf());
		else
			// TODO: Improve error message to explain why
			str.Format(TXT_DATA_TYPE_CANT_BE_s, type.Format(ns).AddressOf());

		WriteError(str, file, node);
	}

	asCScriptNode *n = node->firstChild->next;

	while( n )
	{
		// Verify that the name isn't taken
		asCString name(&file->code[n->tokenPos], n->tokenLength);
		CheckNameConflict(name.AddressOf(), n, file, ns, true, false, false);

		// Register the global variable
		sGlobalVariableDescription *gvar = asNEW(sGlobalVariableDescription);
		if( gvar == 0 )
		{
			node->Destroy(engine);
			return asOUT_OF_MEMORY;
		}

		gvar->script      = file;
		gvar->name        = name;
		gvar->isCompiled  = false;
		gvar->datatype    = type;
		gvar->isEnumValue = false;
		gvar->ns          = ns;

		// TODO: Give error message if wrong
		asASSERT(!gvar->datatype.IsReference());

		// Allocation is done when the variable is compiled, to allow for autos
		gvar->property = 0;
		gvar->index    = 0;

		globVariables.Put(gvar);


		gvar->declaredAtNode = n;
		n = n->next;
		gvar->declaredAtNode->DisconnectParent();
		gvar->initializationNode = 0;
		if( n &&
			( n->nodeType == snAssignment ||
			  n->nodeType == snArgList    ||
			  n->nodeType == snInitList     ) )
		{
			gvar->initializationNode = n;
			n = n->next;
			gvar->initializationNode->DisconnectParent();
		}
	}

	node->Destroy(engine);

	return 0;
}

int asCBuilder::RegisterUsingNamespace(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	asCScriptNode* n = node->firstChild;
	asCString name(&file->code[n->tokenPos], n->tokenLength);

	while (n->next)
	{
	    n = n->next;
	    name += "::" + asCString(&file->code[n->tokenPos], n->tokenLength);
	}

	asSNameSpace* visibleNamespace = engine->AddNameSpace(name.AddressOf());
	asSMapNode<asSNameSpace*, asCArray<asSNameSpace*>>* cursor = 0;

	if (namespaceVisibility.MoveTo(&cursor, ns))
	{
		asCArray<asSNameSpace*>& visibleNamespaces = namespaceVisibility.GetValue(cursor);
		if (!visibleNamespaces.Exists(visibleNamespace))
			namespaceVisibility.GetValue(cursor).PushLast(visibleNamespace);
	}
	else
	{
		asCArray<asSNameSpace*> tmp;
		tmp.PushLast(visibleNamespace);
		namespaceVisibility.Insert(ns, tmp);
	}

	node->Destroy(engine);
	return 0;
}

int asCBuilder::RegisterMixinClass(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	asCScriptNode *cl = node->firstChild;
	asASSERT( cl->nodeType == snClass );

	asCScriptNode *n = cl->firstChild;

	// Skip potential decorator tokens
	while( n->tokenType == ttIdentifier &&
		   (file->TokenEquals(n->tokenPos, n->tokenLength, FINAL_TOKEN) ||
		    file->TokenEquals(n->tokenPos, n->tokenLength, SHARED_TOKEN) ||
			file->TokenEquals(n->tokenPos, n->tokenLength, ABSTRACT_TOKEN) ||
			file->TokenEquals(n->tokenPos, n->tokenLength, EXTERNAL_TOKEN)) )
	{
		// Report error, because mixin class cannot be final or shared
		asCString msg;
		msg.Format(TXT_MIXIN_CANNOT_BE_DECLARED_AS_s, asCString(&file->code[n->tokenPos], n->tokenLength).AddressOf());
		WriteError(msg, file, n);

		asCScriptNode *tmp = n;
		n = n->next;

		// Remove the invalid node, so compilation can continue as if it wasn't there
		tmp->DisconnectParent();
		tmp->Destroy(engine);
	}

	asCString name(&file->code[n->tokenPos], n->tokenLength);

	int r, c;
	file->ConvertPosToRowCol(n->tokenPos, &r, &c);

	CheckNameConflict(name.AddressOf(), n, file, ns, true, false, false);

	sMixinClass *decl = asNEW(sMixinClass);
	if( decl == 0 )
	{
		node->Destroy(engine);
		return asOUT_OF_MEMORY;
	}

	mixinClasses.PushLast(decl);
	decl->name   = name;
	decl->ns     = ns;
	decl->node   = cl;
	decl->script = file;

	// Clean up memory
	cl->DisconnectParent();
	node->Destroy(engine);

	// Check that the mixin class doesn't contain any child types
	// TODO: Add support for child types in mixin classes
	n = cl->firstChild;
	while (n)
	{
		if (n->nodeType == snFuncDef)
		{
			WriteError(TXT_MIXIN_CANNOT_HAVE_CHILD_TYPES, file, n);
			break;
		}
		n = n->next;
	}

	return 0;
}

int asCBuilder::RegisterClass(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	asCScriptNode *n = node->firstChild;
	bool isFinal = false;
	bool isShared = false;
	bool isAbstract = false;
	bool isExternal = false;

	// Check the class modifiers
	while( n->tokenType == ttIdentifier )
	{
		if( file->TokenEquals(n->tokenPos, n->tokenLength, FINAL_TOKEN) )
		{
			if( isAbstract )
				WriteError(TXT_CLASS_CANT_BE_FINAL_AND_ABSTRACT, file, n);
			else
			{
				if( isFinal )
				{
					asCString msg;
					msg.Format(TXT_ATTR_s_INFORMED_MULTIPLE_TIMES, asCString(&file->code[n->tokenPos], n->tokenLength).AddressOf());
					WriteWarning(msg, file, n);
				}
				isFinal = true;
			}
		}
		else if( file->TokenEquals(n->tokenPos, n->tokenLength, SHARED_TOKEN) )
		{
			if( isShared )
			{
				asCString msg;
				msg.Format(TXT_ATTR_s_INFORMED_MULTIPLE_TIMES, asCString(&file->code[n->tokenPos], n->tokenLength).AddressOf());
				WriteWarning(msg, file, n);
			}
			isShared = true;
		}
		else if (file->TokenEquals(n->tokenPos, n->tokenLength, EXTERNAL_TOKEN))
		{
			if (isExternal)
			{
				asCString msg;
				msg.Format(TXT_ATTR_s_INFORMED_MULTIPLE_TIMES, asCString(&file->code[n->tokenPos], n->tokenLength).AddressOf());
				WriteWarning(msg, file, n);
			}
			isExternal = true;
		}
		else if( file->TokenEquals(n->tokenPos, n->tokenLength, ABSTRACT_TOKEN) )
		{
			if( isFinal )
				WriteError(TXT_CLASS_CANT_BE_FINAL_AND_ABSTRACT, file, n);
			else
			{
				if( isAbstract )
				{
					asCString msg;
					msg.Format(TXT_ATTR_s_INFORMED_MULTIPLE_TIMES, asCString(&file->code[n->tokenPos], n->tokenLength).AddressOf());
					WriteWarning(msg, file, n);
				}
				isAbstract = true;
			}
		}
		else
		{
			// This is the name of the class
			break;
		}

		n = n->next;
	}

	asCString name(&file->code[n->tokenPos], n->tokenLength);

	int r, c;
	file->ConvertPosToRowCol(n->tokenPos, &r, &c);

	CheckNameConflict(name.AddressOf(), n, file, ns, true, false, false);

	sClassDeclaration *decl = asNEW(sClassDeclaration);
	if( decl == 0 )
	{
		node->Destroy(engine);
		return asOUT_OF_MEMORY;
	}

	classDeclarations.PushLast(decl);
	decl->name             = name;
	decl->script           = file;
	decl->node             = node;

	// External shared interfaces must not try to redefine the interface
	if (isExternal && (n->next == 0 || n->next->tokenType != ttEndStatement))
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_CANNOT_REDEF, name.AddressOf());
		WriteError(str, file, n);
	}
	else if (!isExternal && n->next && n->next->tokenType == ttEndStatement)
	{
		asCString str;
		str.Format(TXT_MISSING_DEFINITION_OF_s, name.AddressOf());
		WriteError(str, file, n);
	}

	// If this type is shared and there already exist another shared
	// type of the same name, then that one should be used instead of
	// creating a new one.
	asCObjectType *st = 0;
	if( isShared )
	{
		for( asUINT i = 0; i < engine->sharedScriptTypes.GetLength(); i++ )
		{
			st = CastToObjectType(engine->sharedScriptTypes[i]);
			if( st &&
				st->IsShared() &&
				st->name == name &&
				st->nameSpace == ns &&
				!st->IsInterface() )
			{
				// We'll use the existing type
				decl->isExistingShared = true;
				decl->typeInfo         = st;
				module->AddClassType(st);
				st->AddRefInternal();
				break;
			}
		}
	}

	// If the class was declared as external then it must have been compiled in a different module first
	if (isExternal && decl->typeInfo == 0)
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_NOT_FOUND, name.AddressOf());
		WriteError(str, file, n);
	}

	// Remember if the class was declared as external so the saved bytecode can be flagged accordingly
	if (isExternal)
		module->m_externalTypes.PushLast(st);

	if (!decl->isExistingShared)
	{
		// Create a new object type for this class
		st = asNEW(asCObjectType)(engine);
		if (st == 0)
			return asOUT_OF_MEMORY;

		// By default all script classes are marked as garbage collected.
		// Only after the complete structure and relationship between classes
		// is known, can the flag be cleared for those objects that truly cannot
		// form circular references. This is important because a template
		// callback may be called with a script class before the compilation
		// completes, and until it is known, the callback must assume the class
		// is garbage collected.
		st->flags = asOBJ_REF | asOBJ_SCRIPT_OBJECT | asOBJ_GC;
		if (engine->ep.disableScriptClassGC)
			st->flags &= ~asOBJ_GC;

		if (isShared)
			st->flags |= asOBJ_SHARED;

		if (isFinal)
			st->flags |= asOBJ_NOINHERIT;

		if (isAbstract)
			st->flags |= asOBJ_ABSTRACT;

		if (node->tokenType == ttHandle)
			st->flags |= asOBJ_IMPLICIT_HANDLE;

		st->size = sizeof(asCScriptObject);
		st->name = name;
		st->nameSpace = ns;
		st->module = module;
		module->AddClassType(st);
		if (isShared)
		{
			engine->sharedScriptTypes.PushLast(st);
			st->AddRefInternal();
		}
		decl->typeInfo = st;

		// Use the default script class behaviours
		st->beh = engine->scriptTypeBehaviours.beh;

		// TODO: Move this to asCObjectType so that the asCRestore can reuse it
		engine->scriptFunctions[st->beh.addref]->AddRefInternal();
		engine->scriptFunctions[st->beh.release]->AddRefInternal();
		engine->scriptFunctions[st->beh.gcEnumReferences]->AddRefInternal();
		engine->scriptFunctions[st->beh.gcGetFlag]->AddRefInternal();
		engine->scriptFunctions[st->beh.gcGetRefCount]->AddRefInternal();
		engine->scriptFunctions[st->beh.gcReleaseAllReferences]->AddRefInternal();
		engine->scriptFunctions[st->beh.gcSetFlag]->AddRefInternal();
		engine->scriptFunctions[st->beh.copy]->AddRefInternal();
		engine->scriptFunctions[st->beh.factory]->AddRefInternal();
		engine->scriptFunctions[st->beh.construct]->AddRefInternal();
		// TODO: weak: Should not do this if the class has been declared with noweak
		engine->scriptFunctions[st->beh.getWeakRefFlag]->AddRefInternal();

		// Skip to the content of the class
		while (n && n->nodeType == snIdentifier)
			n = n->next;
	}

	// Register possible child types
	while (n)
	{
		node = n->next;
		if (n->nodeType == snFuncDef)
		{
			n->DisconnectParent();
			if (!decl->isExistingShared)
				RegisterFuncDef(n, file, 0, st);
			else
			{
				// Destroy the node, since it won't be used
				// TODO: Should verify that the funcdef is identical to the one in the existing shared class
				n->Destroy(engine);
			}
		}
		n = node;
	}

	return 0;
}

int asCBuilder::RegisterInterface(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	asCScriptNode *n = node->firstChild;

	bool isShared = false;
	bool isExternal = false;
	while( n->nodeType == snIdentifier )
	{
		if (file->TokenEquals(n->tokenPos, n->tokenLength, SHARED_TOKEN))
			isShared = true;
		else if (file->TokenEquals(n->tokenPos, n->tokenLength, EXTERNAL_TOKEN))
			isExternal = true;
		else
			break;
		n = n->next;
	}

	int r, c;
	file->ConvertPosToRowCol(n->tokenPos, &r, &c);

	asCString name;
	name.Assign(&file->code[n->tokenPos], n->tokenLength);
	CheckNameConflict(name.AddressOf(), n, file, ns, true, false, isShared);

	sClassDeclaration *decl = asNEW(sClassDeclaration);
	if( decl == 0 )
	{
		node->Destroy(engine);
		return asOUT_OF_MEMORY;
	}

	interfaceDeclarations.PushLast(decl);
	decl->name             = name;
	decl->script           = file;
	decl->node             = node;

	// External shared interfaces must not try to redefine the interface
	if (isExternal && (n->next == 0 || n->next->tokenType != ttEndStatement) )
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_CANNOT_REDEF, name.AddressOf());
		WriteError(str, file, n);
	}
	else if (!isExternal && n->next && n->next->tokenType == ttEndStatement)
	{
		asCString str;
		str.Format(TXT_MISSING_DEFINITION_OF_s, name.AddressOf());
		WriteError(str, file, n);
	}

	// If this type is shared and there already exist another shared
	// type of the same name, then that one should be used instead of
	// creating a new one.
	if( isShared )
	{
		for( asUINT i = 0; i < engine->sharedScriptTypes.GetLength(); i++ )
		{
			asCObjectType *st = CastToObjectType(engine->sharedScriptTypes[i]);
			if( st &&
				st->IsShared() &&
				st->name == name &&
				st->nameSpace == ns &&
				st->IsInterface() )
			{
				// We'll use the existing type
				decl->isExistingShared = true;
				decl->typeInfo = st;
				module->AddClassType(st);
				st->AddRefInternal();

				// Remember if the interface was declared as external so the saved bytecode can be flagged accordingly
				if (isExternal)
					module->m_externalTypes.PushLast(st);

				return 0;
			}
		}
	}

	// If the interface was declared as external then it must have been compiled in a different module first
	if (isExternal)
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_NOT_FOUND, name.AddressOf());
		WriteError(str, file, n);
	}

	// Register the object type for the interface
	asCObjectType *st = asNEW(asCObjectType)(engine);
	if( st == 0 )
		return asOUT_OF_MEMORY;

	st->flags = asOBJ_REF | asOBJ_SCRIPT_OBJECT;

	if( isShared )
		st->flags |= asOBJ_SHARED;

	st->size = 0; // Cannot be instantiated
	st->name = name;
	st->nameSpace = ns;
	st->module = module;
	module->AddClassType(st);
	if( isShared )
	{
		engine->sharedScriptTypes.PushLast(st);
		st->AddRefInternal();
	}
	decl->typeInfo = st;

	// Use the default script class behaviours
	st->beh.construct = 0;
	st->beh.addref = engine->scriptTypeBehaviours.beh.addref;
	engine->scriptFunctions[st->beh.addref]->AddRefInternal();
	st->beh.release = engine->scriptTypeBehaviours.beh.release;
	engine->scriptFunctions[st->beh.release]->AddRefInternal();
	st->beh.copy = 0;

	return 0;
}

void asCBuilder::CompileGlobalVariables()
{
	bool compileSucceeded = true;

	// Store state of compilation (errors, warning, output)
	int currNumErrors   = numErrors;
	int currNumWarnings = numWarnings;

	// Backup the original message stream
	bool                       msgCallback     = engine->msgCallback;
	asSSystemFunctionInterface msgCallbackFunc = engine->msgCallbackFunc;
	void                      *msgCallbackObj  = engine->msgCallbackObj;

	// Set the new temporary message stream
	asCOutputBuffer outBuffer;
	engine->SetMessageCallback(asFUNCTION(asCOutputBuffer::CDeclCallback), &outBuffer, asCALL_CDECL_OBJLAST);

	asCOutputBuffer finalOutput;
	asCScriptFunction *initFunc = 0;

	asCSymbolTable<asCGlobalProperty> initOrder;

	// We first try to compile all the primitive global variables, and only after that
	// compile the non-primitive global variables. This permits the constructors
	// for the complex types to use the already initialized variables of primitive
	// type. Note, we currently don't know which global variables are used in the
	// constructors, so we cannot guarantee that variables of complex types are
	// initialized in the correct order, so we won't reorder those.
	bool compilingPrimitives = true;

	// Compile each global variable
	while( compileSucceeded )
	{
		compileSucceeded = false;

		int accumErrors = 0;
		int accumWarnings = 0;

		// Restore state of compilation
		finalOutput.Clear();
		asCSymbolTable<sGlobalVariableDescription>::iterator it = globVariables.List();
		for( ; it; it++ )
		{
			sGlobalVariableDescription *gvar = *it;
			if( gvar->isCompiled )
				continue;

			asCByteCode init(engine);
			numWarnings = 0;
			numErrors = 0;
			outBuffer.Clear();

			// Skip this for now if we're not compiling complex types yet
			if( compilingPrimitives && !gvar->datatype.IsPrimitive() )
				continue;

			if( gvar->declaredAtNode )
			{
				int r, c;
				gvar->script->ConvertPosToRowCol(gvar->declaredAtNode->tokenPos, &r, &c);
				asCString str = gvar->datatype.Format(gvar->ns);
				str += " " + gvar->name;
				str.Format(TXT_COMPILING_s, str.AddressOf());
				WriteInfo(gvar->script->name, str, r, c, true);
			}

			if( gvar->isEnumValue )
			{
				int r;
				if( gvar->initializationNode )
				{
					asCCompiler comp(engine);
					asCScriptFunction func(engine, module, asFUNC_SCRIPT);

					// Set the namespace that should be used during the compilation
					func.nameSpace = gvar->datatype.GetTypeInfo()->nameSpace;

					// Temporarily switch the type of the variable to int so it can be compiled properly
					asCDataType saveType;
					saveType = gvar->datatype;
					gvar->datatype = asCDataType::CreatePrimitive(ttInt, true);
					r = comp.CompileGlobalVariable(this, gvar->script, gvar->initializationNode, gvar, &func);
					gvar->datatype = saveType;

					// Make the function a dummy so it doesn't try to release objects while destroying the function
					func.funcType = asFUNC_DUMMY;
				}
				else
				{
					r = 0;

					// When there is no assignment the value is the last + 1
					int enumVal = 0;
					asCSymbolTable<sGlobalVariableDescription>::iterator prev_it = it;
					prev_it--;
					if( prev_it )
					{
						sGlobalVariableDescription *gvar2 = *prev_it;
						if(gvar2->datatype == gvar->datatype )
						{
							enumVal = int(gvar2->constantValue) + 1;

							if( !gvar2->isCompiled )
							{
								int row, col;
								gvar->script->ConvertPosToRowCol(gvar->declaredAtNode->tokenPos, &row, &col);

								asCString str = gvar->datatype.Format(gvar->ns);
								str += " " + gvar->name;
								str.Format(TXT_COMPILING_s, str.AddressOf());
								WriteInfo(gvar->script->name, str, row, col, true);

								str.Format(TXT_UNINITIALIZED_GLOBAL_VAR_s, gvar2->name.AddressOf());
								WriteError(gvar->script->name, str, row, col);
								r = -1;
							}
						}
					}

					gvar->constantValue = enumVal;
				}

				if( r >= 0 )
				{
					// Set the value as compiled
					gvar->isCompiled = true;
					compileSucceeded = true;
				}
			}
			else
			{
				// Compile the global variable
				initFunc = asNEW(asCScriptFunction)(engine, module, asFUNC_SCRIPT);
				if( initFunc == 0 )
				{
					// Out of memory
					return;
				}

				// Set the namespace that should be used for this function
				initFunc->nameSpace = gvar->ns;

				asCCompiler comp(engine);
				int r = comp.CompileGlobalVariable(this, gvar->script, gvar->initializationNode, gvar, initFunc);
				if( r >= 0 )
				{
					// Compilation succeeded
					gvar->isCompiled = true;
					compileSucceeded = true;
				}
				else
				{
					// Compilation failed
					initFunc->funcType = asFUNC_DUMMY;
					asDELETE(initFunc, asCScriptFunction);
					initFunc = 0;
				}
			}

			if( gvar->isCompiled )
			{
				// Add warnings for this constant to the total build
				if( numWarnings )
				{
					currNumWarnings += numWarnings;
					if( msgCallback )
						outBuffer.SendToCallback(engine, &msgCallbackFunc, msgCallbackObj);
				}

				// Determine order of variable initializations
				if( gvar->property && !gvar->isEnumValue )
					initOrder.Put(gvar->property);

				// Does the function contain more than just a SUSPEND followed by a RET instruction?
				if( initFunc && initFunc->scriptData->byteCode.GetLength() > 2 )
				{
					// Create the init function for this variable
					initFunc->id = engine->GetNextScriptFunctionId();
					engine->AddScriptFunction(initFunc);

					// Finalize the init function for this variable
					initFunc->returnType = asCDataType::CreatePrimitive(ttVoid, false);
					initFunc->scriptData->scriptSectionIdx = engine->GetScriptSectionNameIndex(gvar->script->name.AddressOf());
					if( gvar->declaredAtNode )
					{
						int row, col;
						gvar->script->ConvertPosToRowCol(gvar->declaredAtNode->tokenPos, &row, &col);
						initFunc->scriptData->declaredAt = (row & 0xFFFFF)|((col & 0xFFF)<<20);
					}

					gvar->property->SetInitFunc(initFunc);

					initFunc->ReleaseInternal();
					initFunc = 0;
				}
				else if( initFunc )
				{
					// Destroy the function as it won't be used
					initFunc->funcType = asFUNC_DUMMY;
					asDELETE(initFunc, asCScriptFunction);
					initFunc = 0;
				}

				// Convert enums to true enum values, so subsequent compilations can access it as an enum
				if( gvar->isEnumValue )
				{
					asCEnumType *enumType = CastToEnumType(gvar->datatype.GetTypeInfo());
					asASSERT(NULL != enumType);

					asSEnumValue *e = asNEW(asSEnumValue);
					if( e == 0 )
					{
						// Out of memory
						numErrors++;
						return;
					}

					e->name = gvar->name;
					e->value = int(gvar->constantValue);

					enumType->enumValues.PushLast(e);
				}
			}
			else
			{
				// Add output to final output
				finalOutput.Append(outBuffer);
				accumErrors += numErrors;
				accumWarnings += numWarnings;
			}

			engine->preMessage.isSet = false;
		}

		if( !compileSucceeded )
		{
			if( compilingPrimitives )
			{
				// No more primitives could be compiled, so
				// switch to compiling the complex variables
				compilingPrimitives = false;
				compileSucceeded    = true;
			}
			else
			{
				// No more variables can be compiled
				// Add errors and warnings to total build
				currNumWarnings += accumWarnings;
				currNumErrors   += accumErrors;
				if( msgCallback )
					finalOutput.SendToCallback(engine, &msgCallbackFunc, msgCallbackObj);
			}
		}
	}

	// Restore states
	engine->msgCallback     = msgCallback;
	engine->msgCallbackFunc = msgCallbackFunc;
	engine->msgCallbackObj  = msgCallbackObj;

	numWarnings = currNumWarnings;
	numErrors   = currNumErrors;

	// Set the correct order of initialization
	if( numErrors == 0 )
	{
		// If the length of the arrays are not the same, then this is the compilation
		// of a single variable, in which case the initialization order of the previous
		// variables must be preserved.
		if( module->m_scriptGlobals.GetSize() == initOrder.GetSize() )
			module->m_scriptGlobals.SwapWith(initOrder);
	}

	CleanupEnumValues();
}

void asCBuilder::CleanupEnumValues()
{
	// Delete the enum expressions
	asCSymbolTableIterator<sGlobalVariableDescription> it = globVariables.List();
	while (it)
	{
		sGlobalVariableDescription *gvar = *it;
		if (gvar->isEnumValue)
		{
			// Remove from symboltable. This has to be done prior to freeing the memeory
			globVariables.Erase(it.GetIndex());

			// Destroy the gvar property
			if (gvar->declaredAtNode)
			{
				gvar->declaredAtNode->Destroy(engine);
				gvar->declaredAtNode = 0;
			}
			if (gvar->initializationNode)
			{
				gvar->initializationNode->Destroy(engine);
				gvar->initializationNode = 0;
			}
			if (gvar->property)
			{
				asDELETE(gvar->property, asCGlobalProperty);
				gvar->property = 0;
			}

			asDELETE(gvar, sGlobalVariableDescription);
		}
		else
			it++;
	}
}

int asCBuilder::GetNamespaceAndNameFromNode(asCScriptNode *n, asCScriptCode *script, asSNameSpace *implicitNs, asSNameSpace *&outNs, asCString &outName, bool *isExplicitNs)
{
	asASSERT( n->nodeType == snIdentifier );

	// Get the optional scope from the node
	asSNameSpace *ns = GetNameSpaceFromNode(n->firstChild, script, implicitNs, 0, 0, isExplicitNs);
	if( ns == 0 )
		return -1;

	// Get the name
	asCString name(&script->code[n->lastChild->tokenPos], n->lastChild->tokenLength);

	outNs = ns;
	outName = name;

	return 0;
}

void asCBuilder::AddInterfaceFromMixinToClass(sClassDeclaration *decl, asCScriptNode *errNode, sMixinClass *mixin)
{
	// Determine what interfaces that the mixin implements
	asCScriptNode *node = mixin->node;
	asASSERT(node->nodeType == snClass);

	// Skip the name of the mixin
	node = node->firstChild->next;


	while( node && node->nodeType == snIdentifier )
	{
		bool ok = true;
		asSNameSpace *ns;
		asCString name;
		if( GetNamespaceAndNameFromNode(node, mixin->script, mixin->ns, ns, name) < 0 )
			ok = false;
		else
		{
			// Find the object type for the interface
			asCObjectType *objType = GetObjectType(name.AddressOf(), ns);

			// Check that the object type is an interface
			if( objType && objType->IsInterface() )
			{
				// Only add the interface if the class doesn't already implement it
				if( !decl->typeInfo->Implements(objType) )
					AddInterfaceToClass(decl, errNode, objType);
			}
			else
			{
				WriteError(TXT_MIXIN_CLASS_CANNOT_INHERIT, mixin->script, node);
				ok = false;
			}
		}

		if( !ok )
		{
			// Remove this node so the error isn't reported again
			asCScriptNode *delNode = node;
			node = node->prev;
			delNode->DisconnectParent();
			delNode->Destroy(engine);
		}

		node = node->next;
	}
}

void asCBuilder::AddInterfaceToClass(sClassDeclaration *decl, asCScriptNode *errNode, asCObjectType *intfType)
{
	// A shared type may only implement from shared interfaces
	if( decl->typeInfo->IsShared() && !intfType->IsShared() )
	{
		asCString msg;
		msg.Format(TXT_SHARED_CANNOT_IMPLEMENT_NON_SHARED_s, intfType->name.AddressOf());
		WriteError(msg, decl->script, errNode);
		return;
	}

	if( decl->isExistingShared )
	{
		// If the class is an existing shared class, then just check if the
		// interface exists in the original declaration too
		if( !decl->typeInfo->Implements(intfType) )
		{
			asCString str;
			str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, decl->typeInfo->GetName());
			WriteError(str, decl->script, errNode);
			return;
		}
	}
	else
	{
		// If the interface is already in the class then don't add it again
		if( decl->typeInfo->Implements(intfType) )
			return;

		// Add the interface to the class
		CastToObjectType(decl->typeInfo)->interfaces.PushLast(intfType);

		// Add the inherited interfaces too
		// For interfaces this will be done outside to handle out-of-order declarations
		if( !CastToObjectType(decl->typeInfo)->IsInterface() )
		{
			for( asUINT n = 0; n < intfType->interfaces.GetLength(); n++ )
				AddInterfaceToClass(decl, errNode, intfType->interfaces[n]);
		}
	}
}

void asCBuilder::CompileInterfaces()
{
	asUINT n;

	// Order the interfaces with inheritances so that the inherited
	// of inherited interfaces can be added properly
	for( n = 0; n < interfaceDeclarations.GetLength(); n++ )
	{
		sClassDeclaration *intfDecl = interfaceDeclarations[n];
		asCObjectType *intfType = CastToObjectType(intfDecl->typeInfo);

		if( intfType->interfaces.GetLength() == 0 ) continue;

		// If any of the derived interfaces are found after this interface, then move this to the end of the list
		for( asUINT m = n+1; m < interfaceDeclarations.GetLength(); m++ )
		{
			if( intfType != interfaceDeclarations[m]->typeInfo &&
				intfType->Implements(interfaceDeclarations[m]->typeInfo) )
			{
				interfaceDeclarations.RemoveIndex(n);
				interfaceDeclarations.PushLast(intfDecl);

				// Decrease index so that we don't skip an entry
				n--;
				break;
			}
		}
	}

	// Now recursively add the additional inherited interfaces
	for( n = 0; n < interfaceDeclarations.GetLength(); n++ )
	{
		sClassDeclaration *intfDecl = interfaceDeclarations[n];
		if( intfDecl->isExistingShared )
		{
			// Set the declaration as validated already, so that other
			// types that contain this will accept this type
			intfDecl->validState = 1;
			continue;
		}
		
		asCObjectType *intfType = CastToObjectType(intfDecl->typeInfo);

		// TODO: Is this really at the correct place? Hasn't the vfTableIdx already been set here?
		// Co-opt the vfTableIdx value in our own methods to indicate the
		// index the function should have in the table chunk for this interface.
		for( asUINT d = 0; d < intfType->methods.GetLength(); d++ )
		{
			asCScriptFunction *func = GetFunctionDescription(intfType->methods[d]);
			func->vfTableIdx = d;

			asASSERT(func->objectType == intfType);
		}

		// As new interfaces will be added to the end of the list, all
		// interfaces will be traversed the same as recursively
		for( asUINT m = 0; m < intfType->interfaces.GetLength(); m++ )
		{
			asCObjectType *base = intfType->interfaces[m];

			// Add any interfaces not already implemented
			for( asUINT l = 0; l < base->interfaces.GetLength(); l++ )
				AddInterfaceToClass(intfDecl, intfDecl->node, base->interfaces[l]);

			// Add the methods from the implemented interface
			for( asUINT l = 0; l < base->methods.GetLength(); l++ )
			{
				// If the derived interface implements the same method, then don't add the base interface' method
				asCScriptFunction *baseFunc = GetFunctionDescription(base->methods[l]);
				asCScriptFunction *derivedFunc = 0;
				bool found = false;
				for( asUINT d = 0; d < intfType->methods.GetLength(); d++ )
				{
					derivedFunc = GetFunctionDescription(intfType->methods[d]);
					if( derivedFunc->IsSignatureEqual(baseFunc) )
					{
						found = true;
						break;
					}
				}

				if( !found )
				{
					// Add the method
					intfType->methods.PushLast(baseFunc->id);
					baseFunc->AddRefInternal();
				}
			}
		}
	}
}

void asCBuilder::DetermineTypeRelations()
{
	// Determine inheritance between interfaces
	for (asUINT n = 0; n < interfaceDeclarations.GetLength(); n++)
	{
		sClassDeclaration *intfDecl = interfaceDeclarations[n];
		asCObjectType *intfType = CastToObjectType(intfDecl->typeInfo);

		asCScriptNode *node = intfDecl->node;
		asASSERT(node && node->nodeType == snInterface);
		node = node->firstChild;

		// Skip the 'shared' & 'external' keywords
		while( node->nodeType == snIdentifier &&
			   (intfDecl->script->TokenEquals(node->tokenPos, node->tokenLength, SHARED_TOKEN) ||
				intfDecl->script->TokenEquals(node->tokenPos, node->tokenLength, EXTERNAL_TOKEN)) )
			node = node->next;

		// Skip the name
		node = node->next;

		// Verify the inherited interfaces
		while (node && node->nodeType == snIdentifier)
		{
			asSNameSpace *ns;
			asCString name;
			bool isExplicitNs = false;
			if (GetNamespaceAndNameFromNode(node, intfDecl->script, intfType->nameSpace, ns, name, &isExplicitNs) < 0)
			{
				node = node->next;
				continue;
			}

			asCObjectType* objType = 0;
			bool ok = FindObjectTypeOrMixinInNsHierarchy(name, ns, isExplicitNs, node, intfDecl->script, &objType, 0);

			// Check that the object type is an interface
			if (ok)
			{
				if (objType && objType->IsInterface())
				{
					// Check that the implemented interface is shared if the base interface is shared
					if (intfType->IsShared() && !objType->IsShared())
					{
						asCString str;
						str.Format(TXT_SHARED_CANNOT_IMPLEMENT_NON_SHARED_s, objType->GetName());
						WriteError(str, intfDecl->script, node);
						ok = false;
					}
				}
				else
				{
					WriteError(TXT_INTERFACE_CAN_ONLY_IMPLEMENT_INTERFACE, intfDecl->script, node);
					ok = false;
				}
			}

			if (ok)
			{
				// Make sure none of the implemented interfaces implement from this one
				asCObjectType *base = objType;
				while (base != 0)
				{
					if (base == intfType)
					{
						WriteError(TXT_CANNOT_IMPLEMENT_SELF, intfDecl->script, node);
						ok = false;
						break;
					}

					// At this point there is at most one implemented interface
					if (base->interfaces.GetLength())
						base = base->interfaces[0];
					else
						break;
				}
			}

			if (ok)
				AddInterfaceToClass(intfDecl, node, objType);

			// Remove the nodes so they aren't parsed again
			asCScriptNode *delNode = node;
			node = node->next;
			delNode->DisconnectParent();
			delNode->Destroy(engine);
		}
	}

	// Determine class inheritances and interfaces
	for (asUINT n = 0; n < classDeclarations.GetLength(); n++)
	{
		sClassDeclaration *decl = classDeclarations[n];
		asCScriptCode *file = decl->script;

		// Find the base class that this class inherits from
		bool multipleInheritance = false;
		asCScriptNode *node = decl->node->firstChild;

		while (file->TokenEquals(node->tokenPos, node->tokenLength, FINAL_TOKEN) ||
			file->TokenEquals(node->tokenPos, node->tokenLength, SHARED_TOKEN) ||
			file->TokenEquals(node->tokenPos, node->tokenLength, ABSTRACT_TOKEN) ||
			file->TokenEquals(node->tokenPos, node->tokenLength, EXTERNAL_TOKEN))
		{
			node = node->next;
		}

		// Skip the name of the class
		asASSERT(node->tokenType == ttIdentifier);
		node = node->next;

		while (node && node->nodeType == snIdentifier)
		{
			asSNameSpace *ns;
			asCString name;
			bool isExplicitNs = false;
			if (GetNamespaceAndNameFromNode(node, file, decl->typeInfo->nameSpace, ns, name, &isExplicitNs) < 0)
			{
				node = node->next;
				continue;
			}

			// Find the object type for the interface
			asCObjectType *objType = 0;
			sMixinClass *mixin = 0;
			asSNameSpace *origNs = ns;

			bool ok = FindObjectTypeOrMixinInNsHierarchy(name, ns, isExplicitNs, node, file, &objType, &mixin);

			if (ok)
			{
				if (objType == 0 && mixin == 0)
				{
					asCString str;
					if (origNs->name == "")
						str.Format(TXT_IDENTIFIER_s_NOT_DATA_TYPE_IN_GLOBAL_NS, name.AddressOf());
					else
						str.Format(TXT_IDENTIFIER_s_NOT_DATA_TYPE_IN_NS_s, name.AddressOf(), origNs->name.AddressOf());
					WriteError(str, file, node);
				}
				else if (mixin)
				{
					AddInterfaceFromMixinToClass(decl, node, mixin);
				}
				else if (!(objType->flags & asOBJ_SCRIPT_OBJECT) ||
					(objType->flags & asOBJ_NOINHERIT))
				{
					// Either the class is not a script class or interface
					// or the class has been declared as 'final'
					asCString str;
					str.Format(TXT_CANNOT_INHERIT_FROM_s_FINAL, objType->name.AddressOf());
					WriteError(str, file, node);
				}
				else if (objType->size != 0)
				{
					// The class inherits from another script class
					if (!decl->isExistingShared && CastToObjectType(decl->typeInfo)->derivedFrom != 0)
					{
						if (!multipleInheritance)
						{
							WriteError(TXT_CANNOT_INHERIT_FROM_MULTIPLE_CLASSES, file, node);
							multipleInheritance = true;
						}
					}
					else
					{
						// Make sure none of the base classes inherit from this one
						asCObjectType* base = objType;
						bool error = false;
						while (base != 0)
						{
							if (base == decl->typeInfo)
							{
								WriteError(TXT_CANNOT_INHERIT_FROM_SELF, file, node);
								error = true;
								break;
							}

							base = base->derivedFrom;
						}

						if (!error)
						{
							// A shared type may only inherit from other shared types
							if ((decl->typeInfo->IsShared()) && !(objType->IsShared()))
							{
								asCString msg;
								msg.Format(TXT_SHARED_CANNOT_INHERIT_FROM_NON_SHARED_s, objType->name.AddressOf());
								WriteError(msg, file, node);
								error = true;
							}
						}

						if (!error)
						{
							if (decl->isExistingShared)
							{
								// Verify that the base class is the same as the original shared type
								if (CastToObjectType(decl->typeInfo)->derivedFrom != objType)
								{
									asCString str;
									str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, decl->typeInfo->GetName());
									WriteError(str, file, node);
								}
							}
							else
							{
								// Set the base class
								CastToObjectType(decl->typeInfo)->derivedFrom = objType;
								objType->AddRefInternal();
							}
						}
					}
				}
				else
				{
					// The class implements an interface
					AddInterfaceToClass(decl, node, objType);
				}
			}

			node = node->next;
		}
	}
}

// numTempl is the number of template instances that existed in the engine before the build begun
void asCBuilder::CompileClasses(asUINT numTempl)
{
	asUINT n;
	asCArray<sClassDeclaration*> toValidate((int)classDeclarations.GetLength());

	// Order class declarations so that base classes are compiled before derived classes.
	// This will allow the derived classes to copy properties and methods in the next step.
	for( n = 0; n < classDeclarations.GetLength(); n++ )
	{
		sClassDeclaration *decl = classDeclarations[n];
		asCObjectType *derived = CastToObjectType(decl->typeInfo);
		asCObjectType *base = derived->derivedFrom;

		if( base == 0 ) continue;

		// If the base class is found after the derived class, then move the derived class to the end of the list
		for( asUINT m = n+1; m < classDeclarations.GetLength(); m++ )
		{
			sClassDeclaration *declBase = classDeclarations[m];
			if( base == declBase->typeInfo )
			{
				classDeclarations.RemoveIndex(n);
				classDeclarations.PushLast(decl);

				// Decrease index so that we don't skip an entry
				n--;
				break;
			}
		}
	}

	// Go through each of the classes and register the object type descriptions
	for( n = 0; n < classDeclarations.GetLength(); n++ )
	{
		sClassDeclaration *decl = classDeclarations[n];
		asCObjectType *ot = CastToObjectType(decl->typeInfo);
		if( decl->isExistingShared )
		{
			// Set the declaration as validated already, so that other
			// types that contain this will accept this type
			decl->validState = 1;

			// We'll still validate the declaration to make sure nothing new is
			// added to the shared class that wasn't there in the previous
			// compilation. We do not care if something that is there in the previous
			// declaration is not included in the new declaration though.

			asASSERT( ot->interfaces.GetLength() == ot->interfaceVFTOffsets.GetLength() );
		}

		// Methods included from mixin classes should take precedence over inherited methods
		IncludeMethodsFromMixins(decl);

		// Add all properties and methods from the base class
		if( !decl->isExistingShared && ot->derivedFrom )
		{
			asCObjectType *baseType = ot->derivedFrom;

			// The derived class inherits all interfaces from the base class
			for( unsigned int m = 0; m < baseType->interfaces.GetLength(); m++ )
			{
				if( !ot->Implements(baseType->interfaces[m]) )
					ot->interfaces.PushLast(baseType->interfaces[m]);
			}

			// TODO: Need to check for name conflict with new class methods

			// Copy properties from base class to derived class
			for( asUINT p = 0; p < baseType->properties.GetLength(); p++ )
			{
				asCObjectProperty *prop = AddPropertyToClass(decl, baseType->properties[p]->name, baseType->properties[p]->type, baseType->properties[p]->isPrivate, baseType->properties[p]->isProtected, true);

				// The properties must maintain the same offset
				asASSERT(prop && prop->byteOffset == baseType->properties[p]->byteOffset); UNUSED_VAR(prop);
			}

			// Copy methods from base class to derived class
			for( asUINT m = 0; m < baseType->methods.GetLength(); m++ )
			{
				// If the derived class implements the same method, then don't add the base class' method
				asCScriptFunction *baseFunc = GetFunctionDescription(baseType->methods[m]);
				asCScriptFunction *derivedFunc = 0;
				bool found = false;
				for( asUINT d = 0; d < ot->methods.GetLength(); d++ )
				{
					derivedFunc = GetFunctionDescription(ot->methods[d]);
					if( baseFunc->name == "opConv" || baseFunc->name == "opImplConv" ||
						baseFunc->name == "opCast" || baseFunc->name == "opImplCast" )
					{
						// For the opConv and opCast methods, the return type can differ if they are different methods
						if( derivedFunc->name == baseFunc->name &&
							derivedFunc->IsSignatureExceptNameEqual(baseFunc) )
						{
							if( baseFunc->IsFinal() )
							{
								asCString msg;
								msg.Format(TXT_METHOD_CANNOT_OVERRIDE_s, baseFunc->GetDeclaration());
								WriteError(msg, decl->script, decl->node);
							}

							// Move the function from the methods array to the virtualFunctionTable
							ot->methods.RemoveIndex(d);
							ot->virtualFunctionTable.PushLast(derivedFunc);
							found = true;
							break;
						}
					}
					else
					{
						if( derivedFunc->name == baseFunc->name &&
							derivedFunc->IsSignatureExceptNameAndReturnTypeEqual(baseFunc) )
						{
							if( baseFunc->returnType != derivedFunc->returnType )
							{
								asCString msg;
								msg.Format(TXT_DERIVED_METHOD_MUST_HAVE_SAME_RETTYPE_s, baseFunc->GetDeclaration());
								WriteError(msg, decl->script, decl->node);
							}

							if( baseFunc->IsFinal() )
							{
								asCString msg;
								msg.Format(TXT_METHOD_CANNOT_OVERRIDE_s, baseFunc->GetDeclaration());
								WriteError(msg, decl->script, decl->node);
							}

							// Move the function from the methods array to the virtualFunctionTable
							ot->methods.RemoveIndex(d);
							ot->virtualFunctionTable.PushLast(derivedFunc);
							found = true;
							break;
						}
					}
				}

				if( !found )
				{
					// Push the base class function on the virtual function table
					ot->virtualFunctionTable.PushLast(baseType->virtualFunctionTable[m]);
					baseType->virtualFunctionTable[m]->AddRefInternal();

					CheckForConflictsDueToDefaultArgs(decl->script, decl->node, baseType->virtualFunctionTable[m], ot);
				}

				ot->methods.PushLast(baseType->methods[m]);
				engine->scriptFunctions[baseType->methods[m]]->AddRefInternal();
			}
		}

		if( !decl->isExistingShared )
		{
			// Move this class' methods into the virtual function table
			for( asUINT m = 0; m < ot->methods.GetLength(); m++ )
			{
				asCScriptFunction *func = GetFunctionDescription(ot->methods[m]);
				if( func->funcType != asFUNC_VIRTUAL )
				{
					// Move the reference from the method list to the virtual function list
					ot->methods.RemoveIndex(m);
					ot->virtualFunctionTable.PushLast(func);

					// Substitute the function description in the method list for a virtual method
					// Make sure the methods are in the same order as the virtual function table
					ot->methods.PushLast(CreateVirtualFunction(func, (int)ot->virtualFunctionTable.GetLength() - 1));
					m--;
				}
			}

			// Make virtual function table chunks for each implemented interface
			for( asUINT m = 0; m < ot->interfaces.GetLength(); m++ )
			{
				asCObjectType *intf = ot->interfaces[m];

				// Add all the interface's functions to the virtual function table
				asUINT offset = asUINT(ot->virtualFunctionTable.GetLength());
				ot->interfaceVFTOffsets.PushLast(offset);

				for( asUINT j = 0; j < intf->methods.GetLength(); j++ )
				{
					asCScriptFunction *intfFunc = GetFunctionDescription(intf->methods[j]);

					// Only create the table for functions that are explicitly from this interface,
					// inherited interface methods will be put in that interface's table.
					if( intfFunc->objectType != intf )
						continue;

					asASSERT((asUINT)intfFunc->vfTableIdx == j);

					//Find the interface function in the list of methods
					asCScriptFunction *realFunc = 0;
					for( asUINT p = 0; p < ot->methods.GetLength(); p++ )
					{
						asCScriptFunction *func = GetFunctionDescription(ot->methods[p]);

						if( func->signatureId == intfFunc->signatureId )
						{
							if( func->funcType == asFUNC_VIRTUAL )
							{
								realFunc = ot->virtualFunctionTable[func->vfTableIdx];
							}
							else
							{
								// This should not happen, all methods were moved into the virtual table
								asASSERT(false);
							}
							break;
						}
					}

					// If realFunc is still null, the interface was not
					// implemented and we error out later in the checks.
					ot->virtualFunctionTable.PushLast(realFunc);
					if( realFunc )
						realFunc->AddRefInternal();
				}
			}
		}

		// Enumerate each of the declared properties
		asCScriptNode *node = decl->node->firstChild->next;

		// Skip list of classes and interfaces
		while( node && node->nodeType == snIdentifier )
			node = node->next;

		while( node && node->nodeType == snDeclaration )
		{
			asCScriptNode *nd = node->firstChild;

			// Is the property declared as private or protected?
			bool isPrivate = false, isProtected = false;
			if( nd && nd->tokenType == ttPrivate )
			{
				isPrivate = true;
				nd = nd->next;
			}
			else if( nd && nd->tokenType == ttProtected )
			{
				isProtected = true;
				nd = nd->next;
			}

			// Determine the type of the property
			asCScriptCode *file = decl->script;
			asCDataType dt = CreateDataTypeFromNode(nd, file, ot->nameSpace, false, ot);
			if( ot->IsShared() && dt.GetTypeInfo() && !dt.GetTypeInfo()->IsShared() )
			{
				asCString msg;
				msg.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, dt.GetTypeInfo()->name.AddressOf());
				WriteError(msg, file, node);
			}

			if( dt.IsReadOnly() )
				WriteError(TXT_PROPERTY_CANT_BE_CONST, file, node);

			// Multiple properties can be declared separated by ,
			nd = nd->next;
			while( nd )
			{
				asCString name(&file->code[nd->tokenPos], nd->tokenLength);

				if( !decl->isExistingShared )
				{
					CheckNameConflictMember(ot, name.AddressOf(), nd, file, true, false);
					AddPropertyToClass(decl, name, dt, isPrivate, isProtected, false, file, nd);
				}
				else
				{
					// Verify that the property exists in the original declaration
					bool found = false;
					for( asUINT p = 0; p < ot->properties.GetLength(); p++ )
					{
						asCObjectProperty *prop = ot->properties[p];
						if( prop->isPrivate == isPrivate &&
							prop->isProtected == isProtected &&
							prop->name == name &&
							prop->type.IsEqualExceptRef(dt) )
						{
							found = true;
							break;
						}
					}
					if( !found )
					{
						asCString str;
						str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, ot->GetName());
						WriteError(str, file, nd);
					}
				}

				// Skip the initialization node
				if( nd->next && nd->next->nodeType != snIdentifier )
					nd = nd->next;

				nd = nd->next;
			}

			node = node->next;
		}

		// Add properties from included mixin classes that don't conflict with existing properties
		IncludePropertiesFromMixins(decl);

		if( !decl->isExistingShared )
			toValidate.PushLast(decl);

		asASSERT( ot->interfaces.GetLength() == ot->interfaceVFTOffsets.GetLength() );
	}

	// TODO: Warn if a method overrides a base method without marking it as 'override'.
	//       It must be possible to turn off this warning through engine property.

	// TODO: A base class should be able to mark a method as 'abstract'. This will
	//       allow a base class to provide a partial implementation, but still force
	//       derived classes to implement specific methods.

	// Verify that all interface methods are implemented in the classes
	// We do this here so the base class' methods have already been inherited
	for( n = 0; n < classDeclarations.GetLength(); n++ )
	{
		sClassDeclaration *decl = classDeclarations[n];
		if( decl->isExistingShared ) continue;

		asCObjectType *ot = CastToObjectType(decl->typeInfo);
		asCArray<bool> overrideValidations(ot->GetMethodCount());
		for( asUINT k = 0; k < ot->methods.GetLength(); k++ )
			overrideValidations.PushLast( !static_cast<asCScriptFunction*>(ot->GetMethodByIndex(k, false))->IsOverride() );

		for( asUINT m = 0; m < ot->interfaces.GetLength(); m++ )
		{
			asCObjectType *objType = ot->interfaces[m];
			for( asUINT i = 0; i < objType->methods.GetLength(); i++ )
			{
				// Only check the interface methods that was explicitly declared in this interface
				// Methods that was inherited from other interfaces will be checked in those interfaces
				if( objType != engine->scriptFunctions[objType->methods[i]]->objectType )
					continue;

				asUINT overrideIndex;
				if( !DoesMethodExist(ot, objType->methods[i], &overrideIndex) )
				{
					asCString str;
					str.Format(TXT_MISSING_IMPLEMENTATION_OF_s,
						engine->GetFunctionDeclaration(objType->methods[i]).AddressOf());
					WriteError(str, decl->script, decl->node);
				}
				else
					overrideValidations[overrideIndex] = true;
			}
		}

		bool hasBaseClass = ot->derivedFrom != 0;

		for( asUINT j = 0; j < overrideValidations.GetLength(); j++ )
		{
			if( !overrideValidations[j] && (!hasBaseClass || !DoesMethodExist(ot->derivedFrom, ot->methods[j])) )
			{
				asCString msg;
				msg.Format(TXT_METHOD_s_DOES_NOT_OVERRIDE, ot->GetMethodByIndex(j, false)->GetDeclaration());
				WriteError(msg, decl->script, decl->node);
			}
		}
	}

	// Verify that the declared structures are valid, e.g. that the structure
	// doesn't contain a member of its own type directly or indirectly
	while( toValidate.GetLength() > 0 )
	{
		asUINT numClasses = (asUINT)toValidate.GetLength();

		asCArray<sClassDeclaration*> toValidateNext((int)toValidate.GetLength());
		while( toValidate.GetLength() > 0 )
		{
			sClassDeclaration *decl = toValidate[toValidate.GetLength()-1];
			asCObjectType *ot = CastToObjectType(decl->typeInfo);
			int validState = 1;
			for( n = 0; n < ot->properties.GetLength(); n++ )
			{
				// A valid structure is one that uses only primitives or other valid objects
				asCObjectProperty *prop = ot->properties[n];
				asCDataType dt = prop->type;

				// TODO: Add this check again, once solving the issues commented below
				/*
				if( dt.IsTemplate() )
				{
					// TODO: This must verify all sub types, not just the first one
					// TODO: Just because the subtype is not a handle doesn't mean the template will actually instance the object
					//       this it shouldn't automatically raise an error for this, e.g. weakref<Object> should be legal as member
					//       of the Object class
					asCDataType sub = dt;
					while( sub.IsTemplate() && !sub.IsObjectHandle() )
						sub = sub.GetSubType();

					dt = sub;
				}
				*/

				if( dt.IsObject() && !dt.IsObjectHandle() )
				{
					// Find the class declaration
					sClassDeclaration *pdecl = 0;
					for( asUINT p = 0; p < classDeclarations.GetLength(); p++ )
					{
						if( classDeclarations[p]->typeInfo == dt.GetTypeInfo() )
						{
							pdecl = classDeclarations[p];
							break;
						}
					}

					if( pdecl )
					{
						if( pdecl->typeInfo == decl->typeInfo )
						{
							WriteError(TXT_ILLEGAL_MEMBER_TYPE, decl->script, decl->node);
							validState = 2;
							break;
						}
						else if( pdecl->validState != 1 )
						{
							validState = pdecl->validState;
							break;
						}
					}
				}
			}

			if( validState == 1 )
			{
				decl->validState = 1;
				toValidate.PopLast();
			}
			else if( validState == 2 )
			{
				decl->validState = 2;
				toValidate.PopLast();
			}
			else
			{
				toValidateNext.PushLast(toValidate.PopLast());
			}
		}

		toValidate = toValidateNext;
		toValidateNext.SetLength(0);

		if( numClasses == toValidate.GetLength() )
		{
			WriteError(TXT_ILLEGAL_MEMBER_TYPE, toValidate[0]->script, toValidate[0]->node);
			break;
		}
	}

	if( numErrors > 0 ) return;

	// Iterate through classes and verify if any automatically generated copy constructor can really be compiled,
	// i.e. all members and the base class can be copied. If not, then remove the auto generated copy constructor 
	// so it will not be compiled. This is done here when it is known the order of class inheritance, and all classes
	// have received their members from mixins, etc.
	for (n = 0; n < classDeclarations.GetLength(); n++)
	{
		sClassDeclaration* decl = classDeclarations[n];
		if (decl->isExistingShared)
			continue;

		asCObjectType* ot = CastToObjectType(decl->typeInfo);
		if (ot == 0 || ot->beh.copyconstruct == 0)
			continue;

		// Find the sFunctionDescriptor to determine if the copy constructor is auto generated or not
		asUINT funcDesc = asUINT(-1);
		for (asUINT d = 0; d < functions.GetLength(); d++)
			if (functions[d] && functions[d]->funcId == ot->beh.copyconstruct)
			{
				funcDesc = d;
				break;
			}

		if (funcDesc != asUINT(-1) && functions[funcDesc]->node == 0)
		{
			// The copy constructor is auto generated. Check if all members and base class can be copied
			bool canBeCopied = true;
			asCScriptFunction *func = engine->scriptFunctions[ot->beh.copyconstruct];
			asCObjectType* objType = func->objectType;
			if (objType->derivedFrom)
			{
				if (objType->derivedFrom->beh.copyconstruct == 0 &&
					(objType->derivedFrom->beh.construct == 0 || objType->derivedFrom->beh.copy == 0))
					canBeCopied = false;
				else
				{
					// The base class can be copied. Check if it can be done from a const object
					// TODO: Should allow it anyway, but the copy constructor should be generated as taking a non-const reference
					asCScriptFunction* baseFunc = engine->scriptFunctions[objType->derivedFrom->beh.copyconstruct];
					if (baseFunc && !baseFunc->parameterTypes[0].IsObjectConst())
						canBeCopied = false;
					else if (!baseFunc)
					{
						baseFunc = engine->scriptFunctions[objType->derivedFrom->beh.copy];
						if (baseFunc && !baseFunc->parameterTypes[0].IsObjectConst())
							canBeCopied = false;
					}
				}
			}

			for (asUINT m = 0; canBeCopied && m < objType->properties.GetLength(); m++)
			{
				asCObjectProperty* prop = objType->properties[m];

				// Check if the property is inherited
				asCScriptNode* declNode = 0;
				for (asUINT p = 0; p < decl->propInits.GetLength(); p++)
				{
					if (decl->propInits[p].name == prop->name)
					{
						declNode = decl->propInits[p].declNode;
						break;
					}
				}

				// If declNode is null then the property was inherited
				if (declNode == 0)
					continue;

				// Check if the property can be copied
				asCObjectType* propObjType = CastToObjectType(prop->type.GetTypeInfo());
				if (propObjType && !prop->type.IsObjectHandle() && propObjType->beh.copyconstruct == 0 &&
					(propObjType->beh.construct == 0 || propObjType->beh.copy == 0))
					canBeCopied = false;
				else if(propObjType && !prop->type.IsObjectHandle())
				{
					// The property can be copied. Check if it can be done from a const object
					// TODO: Should allow it anyway, but the copy constructor should be generated as taking a non-const reference
					asCScriptFunction* baseFunc = engine->scriptFunctions[propObjType->beh.copyconstruct];
					if (baseFunc && !baseFunc->parameterTypes[0].IsObjectConst())
						canBeCopied = false;
					else if (!baseFunc)
					{
						baseFunc = engine->scriptFunctions[propObjType->beh.copy];
						if (baseFunc && !baseFunc->parameterTypes[0].IsObjectConst())
							canBeCopied = false;
					}
				}
			}

			if (!canBeCopied)
			{
				// Remove the function descriptor and the script function that was prepared for compilation
				asDELETE(functions[funcDesc], sFunctionDescription);
				functions.RemoveIndex(funcDesc);

				module->m_scriptFunctions.RemoveValue(func);
				func->module = 0;
				func->ReleaseInternal();
				ot->beh.constructors.RemoveValue(ot->beh.copyconstruct);
				ot->beh.copyconstruct = 0;
				func->ReleaseInternal();

				// Do the same for the copyfactory
				for (asUINT d = 0; d < functions.GetLength(); d++)
					if (functions[d] && functions[d]->funcId == ot->beh.copyfactory)
					{
						asDELETE(functions[d], sFunctionDescription);
						functions.RemoveIndex(d);
						break;
					}
				func = engine->scriptFunctions[ot->beh.copyfactory];
				module->m_scriptFunctions.RemoveValue(func);
				func->module = 0;
				func->ReleaseInternal();
				ot->beh.factories.RemoveValue(ot->beh.copyfactory);
				ot->beh.copyfactory = 0;
				func->ReleaseInternal();
			}
		}
	}

	// Verify which script classes can really form circular references, and mark only those as garbage collected.
	// This must be done in the correct order, so that a class that contains another class isn't needlessly marked
	// as garbage collected, just because the contained class was evaluated afterwards.

	// TODO: runtime optimize: This algorithm can be further improved by checking the types that inherits from
	//                         a base class. If the base class is not shared all the classes that derive from it
	//                         are known at compile time, and can thus be checked for potential circular references too.
	//
	//                         Observe, that doing this would conflict with another potential future feature, which is to
	//                         allow incremental builds, i.e. allow application to add or replace classes in an
	//                         existing module. However, the applications that want to use that should use a special
	//                         build flag to not finalize the module.

	if( !engine->ep.disableScriptClassGC )
	{
		asCArray<asCObjectType*> typesToValidate;
		for( n = 0; n < classDeclarations.GetLength(); n++ )
		{
			// Existing shared classes won't need evaluating, nor interfaces
			sClassDeclaration *decl = classDeclarations[n];
			if( decl->isExistingShared ) continue;
	
			asCObjectType *ot = CastToObjectType(decl->typeInfo);
			if( ot->IsInterface() ) continue;
	
			typesToValidate.PushLast(ot);
		}
	
		asUINT numReevaluations = 0;
		while( typesToValidate.GetLength() )
		{
			if( numReevaluations > typesToValidate.GetLength() )
			{
				// No types could be completely evaluated in the last iteration so
				// we consider the remaining types in the array as garbage collected
				break;
			}
	
			asCObjectType *type = typesToValidate[0];
			typesToValidate.RemoveIndex(0);
	
			// If the type inherits from another type that is yet to be validated, then reinsert it at the end
			if( type->derivedFrom && typesToValidate.Exists(type->derivedFrom) )
			{
				typesToValidate.PushLast(type);
				numReevaluations++;
				continue;
			}
	
			// If the type inherits from a known garbage collected type, then this type must also be garbage collected
			if( type->derivedFrom && (type->derivedFrom->flags & asOBJ_GC) )
			{
				type->flags |= asOBJ_GC;
				continue;
			}
	
			// Evaluate template instances (silently) before verifying each of the classes, since it is possible that
			// a class will be marked as non-garbage collected, which in turn will mark the template instance that uses
			// it as non-garbage collected, which in turn means the class that contains the array also do not have to be
			// garbage collected
			EvaluateTemplateInstances(numTempl, true);
	
			// Is there some path in which this structure is involved in circular references?
			// If the type contains a member of a type that is yet to be validated, then reinsert it at the end
			bool mustReevaluate = false;
			bool gc = false;
			for( asUINT p = 0; p < type->properties.GetLength(); p++ )
			{
				asCDataType dt = type->properties[p]->type;
	
				if (dt.IsFuncdef())
				{
					// If a class holds a function pointer as member then the class must be garbage collected as the
					// function pointer can form circular references with the class through use of a delegate. Example:
					//
					//   class A { B @b; void f(); }
					//   class B { F @f; }
					//   funcdef void F();
					//
					//   A a;
					//   @a.b = B();       // instance of A refers to instance of B
					//   @a.b.f = F(a.f);  // instance of B refers to delegate that refers to instance of A
					//
					gc = true;
					break;
				}
	
				if( !dt.IsObject() )
					continue;
	
				if( typesToValidate.Exists(CastToObjectType(dt.GetTypeInfo())) )
					mustReevaluate = true;
				else
				{
					if( dt.IsTemplate() )
					{
						// Check if any of the subtypes are yet to be evaluated
						bool skip = false;
						for( asUINT s = 0; s < dt.GetTypeInfo()->GetSubTypeCount(); s++ )
						{
							asCObjectType *t = reinterpret_cast<asCObjectType*>(dt.GetTypeInfo()->GetSubType(s));
							if( typesToValidate.Exists(t) )
							{
								mustReevaluate = true;
								skip = true;
								break;
							}
						}
						if( skip )
							continue;
					}
	
					if( dt.IsObjectHandle() )
					{
						// If it is known that the handle can't be involved in a circular reference
						// then this object doesn't need to be marked as garbage collected.
						asCObjectType *prop = CastToObjectType(dt.GetTypeInfo());
	
						if( prop->flags & asOBJ_SCRIPT_OBJECT )
						{
							// For script objects, treat non-final classes as if they can contain references
							// as it is not known what derived classes might do. For final types, check all
							// properties to determine if any of those can cause a circular reference with this
							// class.
							if( prop->flags & asOBJ_NOINHERIT )
							{
								for( asUINT sp = 0; sp < prop->properties.GetLength(); sp++ )
								{
									asCDataType sdt = prop->properties[sp]->type;
	
									if( sdt.IsObject() )
									{
										if( sdt.IsObjectHandle() )
										{
											// TODO: runtime optimize: If the handle is again to a final class, then we can recursively check if the circular reference can occur
											if( sdt.GetTypeInfo()->flags & (asOBJ_SCRIPT_OBJECT | asOBJ_GC) )
											{
												gc = true;
												break;
											}
										}
										else if( sdt.GetTypeInfo()->flags & asOBJ_GC )
										{
											// TODO: runtime optimize: Just because the member type is a potential circle doesn't mean that this one is.
											//                         Only if the object is of a type that can reference this type, either directly or indirectly
											gc = true;
											break;
										}
									}
								}
	
								if( gc )
									break;
							}
							else
							{
								// Assume it is garbage collected as it is not known at compile time what might inherit from this type
								gc = true;
								break;
							}
						}
						else if( prop->flags & asOBJ_GC )
						{
							// If a type is not a script object, adopt its GC flag
							// TODO: runtime optimize: Just because an application registered class is garbage collected, doesn't mean it
							//                         can form a circular reference with this script class. Perhaps need a flag to tell
							//                         if the script classes that contains the type should be garbage collected or not.
							gc = true;
							break;
						}
					}
					else if( dt.GetTypeInfo()->flags & asOBJ_GC )
					{
						// TODO: runtime optimize: Just because the member type is a potential circle doesn't mean that this one is.
						//                         Only if the object is of a type that can reference this type, either directly or indirectly
						gc = true;
						break;
					}
				}
			}
	
			// If the class wasn't found to require garbage collection, but it
			// contains another type that has yet to be evaluated then it must be
			// re-evaluated.
			if( !gc && mustReevaluate )
			{
				typesToValidate.PushLast(type);
				numReevaluations++;
				continue;
			}
	
			// Update the flag in the object type
			if( gc )
				type->flags |= asOBJ_GC;
			else
				type->flags &= ~asOBJ_GC;
	
			// Reset the counter
			numReevaluations = 0;
		}
	}
}

void asCBuilder::IncludeMethodsFromMixins(sClassDeclaration *decl)
{
	asCScriptNode *node = decl->node->firstChild;

	// Skip the class attributes
	while( node->nodeType == snIdentifier &&
		   !decl->script->TokenEquals(node->tokenPos, node->tokenLength, decl->name.AddressOf()) )
		node = node->next;

	// Skip the name of the class
	node = node->next;

	// Find the included mixin classes
	while( node && node->nodeType == snIdentifier )
	{
		asSNameSpace *ns;
		asCString name;
		if( GetNamespaceAndNameFromNode(node, decl->script, decl->typeInfo->nameSpace, ns, name) < 0 )
		{
			node = node->next;
			continue;
		}

		sMixinClass *mixin = 0;
		while( ns )
		{
			// Need to make sure the name is not an object type
			asCObjectType *objType = GetObjectType(name.AddressOf(), ns);
			if( objType == 0 )
				mixin = GetMixinClass(name.AddressOf(), ns);

			if( objType || mixin )
				break;

			ns = engine->GetParentNameSpace(ns);
		}

		if( mixin )
		{
			// Find methods from mixin declaration
			asCScriptNode *n = mixin->node->firstChild;

			// Skip to the member declarations
			// Possible keywords 'final' and 'shared' are removed in RegisterMixinClass so we don't need to worry about those here
			while( n && n->nodeType == snIdentifier )
				n = n->next;

			// Add methods from the mixin that are not already existing in the class
			while( n )
			{
				if( n->nodeType == snFunction )
				{
					// Instead of disconnecting the node, we need to clone it, otherwise other
					// classes that include the same mixin will not see the methods
					asCScriptNode *copy = n->CreateCopy(engine);

					// Register the method, but only if it doesn't already exist in the class
					RegisterScriptFunctionFromNode(copy, mixin->script, CastToObjectType(decl->typeInfo), false, false, mixin->ns, false, true);
				}
				else if( n->nodeType == snVirtualProperty )
				{
					// TODO: mixin: Support virtual properties too
					WriteError("The virtual property syntax is currently not supported for mixin classes", mixin->script, n);
					//RegisterVirtualProperty(node, decl->script, decl->objType, false, false);
				}

				n = n->next;
			}
		}

		node = node->next;
	}
}

void asCBuilder::IncludePropertiesFromMixins(sClassDeclaration *decl)
{
	asCScriptNode *node = decl->node->firstChild;

	// Skip the class attributes
	while( node->nodeType == snIdentifier &&
		   !decl->script->TokenEquals(node->tokenPos, node->tokenLength, decl->name.AddressOf()) )
		node = node->next;

	// Skip the name of the class
	node = node->next;

	// Find the included mixin classes
	while( node && node->nodeType == snIdentifier )
	{
		asSNameSpace *ns;
		asCString name;
		if( GetNamespaceAndNameFromNode(node, decl->script, decl->typeInfo->nameSpace, ns, name) < 0 )
		{
			node = node->next;
			continue;
		}

		sMixinClass *mixin = 0;
		while( ns )
		{
			// Need to make sure the name is not an object type
			asCObjectType *objType = GetObjectType(name.AddressOf(), ns);
			if( objType == 0 )
				mixin = GetMixinClass(name.AddressOf(), ns);

			if( objType || mixin )
				break;

			ns = engine->GetParentNameSpace(ns);
		}

		if( mixin )
		{
			// Find properties from mixin declaration
			asCScriptNode *n = mixin->node->firstChild;

			// Skip to the member declarations
			// Possible keywords 'final' and 'shared' are removed in RegisterMixinClass so we don't need to worry about those here
			while( n && n->nodeType == snIdentifier )
				n = n->next;

			// Add properties from the mixin that are not already existing in the class
			while( n )
			{
				if( n->nodeType == snDeclaration )
				{
					asCScriptNode *n2 = n->firstChild;
					bool isPrivate = false, isProtected = false;
					if( n2 && n2->tokenType == ttPrivate )
					{
						isPrivate = true;
						n2 = n2->next;
					}
					else if( n2 && n2->tokenType == ttProtected )
					{
						isProtected = true;
						n2 = n2->next;
					}

					asCScriptCode *file = mixin->script;
					asCDataType dt = CreateDataTypeFromNode(n2, file, mixin->ns);

					if( decl->typeInfo->IsShared() && dt.GetTypeInfo() && !dt.GetTypeInfo()->IsShared() )
					{
						asCString msg;
						msg.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, dt.GetTypeInfo()->name.AddressOf());
						WriteError(msg, file, n);
						WriteInfo(TXT_WHILE_INCLUDING_MIXIN, decl->script, node);
					}

					if( dt.IsReadOnly() )
						WriteError(TXT_PROPERTY_CANT_BE_CONST, file, n);

					n2 = n2->next;
					while( n2 )
					{
						name.Assign(&file->code[n2->tokenPos], n2->tokenLength);

						// Add the property only if it doesn't already exist in the class
						bool exists = false;
						asCObjectType *ot = CastToObjectType(decl->typeInfo);
						for( asUINT p = 0; p < ot->properties.GetLength(); p++ )
							if( ot->properties[p]->name == name )
							{
								exists = true;
								break;
							}

						if( !exists )
						{
							if( !decl->isExistingShared )
							{
								// It must not conflict with the name of methods
								int r = CheckNameConflictMember(ot, name.AddressOf(), n2, file, true, false);
								if( r < 0 )
									WriteInfo(TXT_WHILE_INCLUDING_MIXIN, decl->script, node);

								AddPropertyToClass(decl, name, dt, isPrivate, isProtected, false, file, n2);
							}
							else
							{
								// Verify that the property exists in the original declaration
								bool found = false;
								for( asUINT p = 0; p < ot->properties.GetLength(); p++ )
								{
									asCObjectProperty *prop = ot->properties[p];
									if( prop->isPrivate == isPrivate &&
										prop->isProtected == isProtected &&
										prop->name == name &&
										prop->type == dt )
									{
										found = true;
										break;
									}
								}
								if( !found )
								{
									asCString str;
									str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, ot->GetName());
									WriteError(str, decl->script, decl->node);
									WriteInfo(TXT_WHILE_INCLUDING_MIXIN, decl->script, node);
								}
							}
						}

						// Skip the initialization expression
						if( n2->next && n2->next->nodeType != snIdentifier )
							n2 = n2->next;

						n2 = n2->next;
					}
				}

				n = n->next;
			}
		}

		node = node->next;
	}
}

int asCBuilder::CreateVirtualFunction(asCScriptFunction *func, int idx)
{
	asCScriptFunction *vf = asNEW(asCScriptFunction)(engine, module, asFUNC_VIRTUAL);
	if( vf == 0 )
		return asOUT_OF_MEMORY;

	vf->name             = func->name;
	vf->nameSpace        = func->nameSpace;
	vf->returnType       = func->returnType;
	vf->parameterTypes   = func->parameterTypes;
	vf->inOutFlags       = func->inOutFlags;
	vf->id               = engine->GetNextScriptFunctionId();
	vf->objectType       = func->objectType;
	vf->objectType->AddRefInternal();
	vf->signatureId      = func->signatureId;
	vf->vfTableIdx       = idx;
	vf->traits           = func->traits;

	// Clear the shared trait since the virtual function should not have that
	vf->SetShared(false);

	// It is not necessary to copy the default args, as they have no meaning in the virtual function

	module->AddScriptFunction(vf);

	// Add a dummy to the builder so that it doesn't mix up function ids
	functions.PushLast(0);

	return vf->id;
}

asCObjectProperty *asCBuilder::AddPropertyToClass(sClassDeclaration *decl, const asCString &name, const asCDataType &dt, bool isPrivate, bool isProtected, bool isInherited, asCScriptCode *file, asCScriptNode *node)
{
	if( node )
	{
		asASSERT(!isInherited);

		// Check if the property is allowed
		if( !dt.CanBeInstantiated() )
		{
			if( file && node )
			{
				asCString str;
				if( dt.IsAbstractClass() )
					str.Format(TXT_ABSTRACT_CLASS_s_CANNOT_BE_INSTANTIATED, dt.Format(decl->typeInfo->nameSpace).AddressOf());
				else if( dt.IsInterface() )
					str.Format(TXT_INTERFACE_s_CANNOT_BE_INSTANTIATED, dt.Format(decl->typeInfo->nameSpace).AddressOf());
				else
					// TODO: Improve error message to explain why
					str.Format(TXT_DATA_TYPE_CANT_BE_s, dt.Format(decl->typeInfo->nameSpace).AddressOf());
				WriteError(str, file, node);
			}
			return 0;
		}

		// Register the initialization expression (if any) to be compiled later
		asCScriptNode *declNode = node;
		asCScriptNode *initNode = 0;
		if( node->next && node->next->nodeType != snIdentifier )
		{
			asASSERT( node->next->nodeType == snAssignment );
			initNode = node->next;
		}

		sPropertyInitializer p(name, declNode, initNode, file);
		decl->propInits.PushLast(p);
	}
	else
	{
		// If the declaration node is not given, then
		// this property is inherited from a base class
		asASSERT(isInherited);
	}

	// Add the property to the object type
	return CastToObjectType(decl->typeInfo)->AddPropertyToClass(name, dt, isPrivate, isProtected, isInherited);
}

bool asCBuilder::DoesMethodExist(asCObjectType *objType, int methodId, asUINT *methodIndex)
{
	asCScriptFunction *method = GetFunctionDescription(methodId);

	for( asUINT n = 0; n < objType->methods.GetLength(); n++ )
	{
		asCScriptFunction *m = GetFunctionDescription(objType->methods[n]);

		if( m->name           != method->name           ) continue;
		if( m->returnType     != method->returnType     ) continue;
		if( m->IsReadOnly()   != method->IsReadOnly()   ) continue;
		if( m->parameterTypes != method->parameterTypes ) continue;
		if( m->inOutFlags     != method->inOutFlags     ) continue;

		if( methodIndex )
			*methodIndex = n;

		return true;
	}

	return false;
}

void asCBuilder::AddDefaultCopyConstructor(asCObjectType* objType, asCScriptCode* file)
{
	// Check if a copy constructor does not already exist, in which case this should be skipped
	if (objType->beh.copyconstruct != 0)
		return;

	int funcId = engine->GetNextScriptFunctionId();

	asCDataType returnType = asCDataType::CreatePrimitive(ttVoid, false);
	asCArray<asCDataType> parameterTypes;
	asCArray<asETypeModifiers> inOutFlags;
	asCArray<asCString*> defaultArgs;
	asCArray<asCString> parameterNames;

	parameterTypes.PushLast(asCDataType::CreateType(objType, true));
	parameterTypes[0].MakeReference(true);
	parameterNames.PushLast("other");
	inOutFlags.PushLast(asTM_INOUTREF);
	defaultArgs.PushLast(0);

	// Add the script function
	// TODO: declaredAt should be set to where the class has been declared
	module->AddScriptFunction(file->idx, 0, funcId, objType->name, returnType, parameterTypes, parameterNames, inOutFlags, defaultArgs, false, objType, false, asSFunctionTraits(), objType->nameSpace);

	// Add it as constructor
	objType->beh.constructors.PushLast(funcId);
	asASSERT(objType->beh.copyconstruct == 0);
	objType->beh.copyconstruct = funcId;
	engine->scriptFunctions[funcId]->AddRefInternal();

	// The bytecode for the default constructor will be generated
	// only after the potential inheritance has been established
	sFunctionDescription* func = asNEW(sFunctionDescription);
	if (func == 0)
	{
		// Out of memory
		return;
	}

	functions.PushLast(func);

	func->script = file;
	func->node = 0;
	func->name = objType->name;
	func->objType = objType;
	func->funcId = funcId;
	func->isExistingShared = false;

	// Add a factory as well
	funcId = engine->GetNextScriptFunctionId();
	objType->beh.factories.PushLast(funcId);
	objType->beh.copyfactory = funcId;
	returnType = asCDataType::CreateObjectHandle(objType, false);
	// TODO: should be the same as the constructor
	module->AddScriptFunction(file->idx, 0, funcId, objType->name, returnType, parameterTypes, parameterNames, inOutFlags, defaultArgs, false);
	functions.PushLast(0);
	asCCompiler compiler(engine);
	compiler.CompileFactory(this, file, engine->scriptFunctions[funcId]);
	engine->scriptFunctions[funcId]->AddRefInternal();

	// If the object is shared, then the factory must also be marked as shared
	if (objType->flags & asOBJ_SHARED)
		engine->scriptFunctions[funcId]->SetShared(true);
}

void asCBuilder::AddDefaultConstructor(asCObjectType *objType, asCScriptCode *file)
{
	// Check if a default constructor does not already exist, in which case this should be skipped
	if (objType->beh.construct != engine->scriptTypeBehaviours.beh.construct)
		return;

	int funcId = engine->GetNextScriptFunctionId();

	asCDataType returnType = asCDataType::CreatePrimitive(ttVoid, false);
	asCArray<asCDataType> parameterTypes;
	asCArray<asETypeModifiers> inOutFlags;
	asCArray<asCString *> defaultArgs;
	asCArray<asCString> parameterNames;

	// Add the script function
	// TODO: declaredAt should be set to where the class has been declared
	module->AddScriptFunction(file->idx, 0, funcId, objType->name, returnType, parameterTypes, parameterNames, inOutFlags, defaultArgs, false, objType, false, asSFunctionTraits(), objType->nameSpace);

	// Set it as default constructor
	if( objType->beh.construct )
		engine->scriptFunctions[objType->beh.construct]->ReleaseInternal();
	objType->beh.construct = funcId;
	objType->beh.constructors[0] = funcId;
	engine->scriptFunctions[funcId]->AddRefInternal();

	// The bytecode for the default constructor will be generated
	// only after the potential inheritance has been established
	sFunctionDescription *func = asNEW(sFunctionDescription);
	if( func == 0 )
	{
		// Out of memory
		return;
	}

	functions.PushLast(func);

	func->script            = file;
	func->node              = 0;
	func->name              = objType->name;
	func->objType           = objType;
	func->funcId            = funcId;
	func->isExistingShared  = false;

	// Add a default factory as well
	funcId = engine->GetNextScriptFunctionId();
	if( objType->beh.factory )
		engine->scriptFunctions[objType->beh.factory]->ReleaseInternal();
	objType->beh.factory = funcId;
	objType->beh.factories[0] = funcId;
	returnType = asCDataType::CreateObjectHandle(objType, false);
	// TODO: should be the same as the constructor
	module->AddScriptFunction(file->idx, 0, funcId, objType->name, returnType, parameterTypes, parameterNames, inOutFlags, defaultArgs, false);
	functions.PushLast(0);
	asCCompiler compiler(engine);
	compiler.CompileFactory(this, file, engine->scriptFunctions[funcId]);
	engine->scriptFunctions[funcId]->AddRefInternal();

	// If the object is shared, then the factory must also be marked as shared
	if( objType->flags & asOBJ_SHARED )
		engine->scriptFunctions[funcId]->SetShared(true);
}

int asCBuilder::RegisterEnum(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	// Is it a shared enum?
	bool isShared = false;
	bool isExternal = false;
	asCEnumType *existingSharedType = 0;
	asCScriptNode *tmp = node->firstChild;
	while( tmp->nodeType == snIdentifier )
	{
		if (file->TokenEquals(tmp->tokenPos, tmp->tokenLength, SHARED_TOKEN))
			isShared = true;
		else if (file->TokenEquals(tmp->tokenPos, tmp->tokenLength, EXTERNAL_TOKEN))
			isExternal = true;
		else
			break;
		tmp = tmp->next;
	}

	// Grab the name of the enumeration
	asCString name;
	asASSERT(snDataType == tmp->nodeType);
	asASSERT(snIdentifier == tmp->firstChild->nodeType);
	name.Assign(&file->code[tmp->firstChild->tokenPos], tmp->firstChild->tokenLength);

	if( isShared )
	{
		// Look for a pre-existing shared enum with the same signature
		for( asUINT n = 0; n < engine->sharedScriptTypes.GetLength(); n++ )
		{
			asCTypeInfo *o = engine->sharedScriptTypes[n];
			if( o &&
				o->IsShared() &&
				(o->flags & asOBJ_ENUM) &&
				o->name == name &&
				o->nameSpace == ns )
			{
				existingSharedType = CastToEnumType(o);
				break;
			}
		}
	}

	// If the enum was declared as external then it must have been compiled in a different module first
	if (isExternal && existingSharedType == 0)
	{
		asCString str;
		str.Format(TXT_EXTERNAL_SHARED_s_NOT_FOUND, name.AddressOf());
		WriteError(str, file, tmp);
	}

	// Remember if the type was declared as external so the saved bytecode can be flagged accordingly
	if (isExternal && existingSharedType)
		module->m_externalTypes.PushLast(existingSharedType);

	// Check the name and add the enum
	int r = CheckNameConflict(name.AddressOf(), tmp->firstChild, file, ns, true, false, false);
	if( asSUCCESS == r )
	{
		asCEnumType *st;

		if( existingSharedType )
		{
			st = existingSharedType;
			st->AddRefInternal();
		}
		else
		{
			st = asNEW(asCEnumType)(engine);
			if( st == 0 )
				return asOUT_OF_MEMORY;

			st->flags     = asOBJ_ENUM;
			if( isShared )
				st->flags |= asOBJ_SHARED;
			st->size      = 4;
			st->name      = name;
			st->nameSpace = ns;
			st->module    = module;
		}
		module->AddEnumType(st);

		if( !existingSharedType && isShared )
		{
			engine->sharedScriptTypes.PushLast(st);
			st->AddRefInternal();
		}

		// Store the location of this declaration for reference in name collisions
		sClassDeclaration *decl = asNEW(sClassDeclaration);
		if( decl == 0 )
			return asOUT_OF_MEMORY;

		decl->name             = name;
		decl->script           = file;
		decl->typeInfo         = st;
		namedTypeDeclarations.PushLast(decl);

		asCDataType type = CreateDataTypeFromNode(tmp, file, ns);
		asASSERT(!type.IsReference());

		// External shared enums must not redeclare the enum values
		if (isExternal && (tmp->next == 0 || tmp->next->tokenType != ttEndStatement) )
		{
			asCString str;
			str.Format(TXT_EXTERNAL_SHARED_s_CANNOT_REDEF, name.AddressOf());
			WriteError(str, file, tmp);
		}
		else if (!isExternal && tmp->next && tmp->next->tokenType == ttEndStatement)
		{
			asCString str;
			str.Format(TXT_MISSING_DEFINITION_OF_s, name.AddressOf());
			WriteError(str, file, tmp);
		}

		// Register the enum values
		tmp = tmp->next;
		while( tmp && tmp->nodeType == snIdentifier )
		{
			name.Assign(&file->code[tmp->tokenPos], tmp->tokenLength);

			if( existingSharedType )
			{
				// If this is a pre-existent shared enum, then just double check
				// that the value is already defined in the original declaration
				bool found = false;
				for( asUINT n = 0; n < st->enumValues.GetLength(); n++ )
					if( st->enumValues[n]->name == name )
					{
						found = true;
						break;
					}

				if( !found )
				{
					asCString str;
					str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, st->GetName());
					WriteError(str, file, tmp);
					break;
				}

				tmp = tmp->next;
				if( tmp && tmp->nodeType == snAssignment )
					tmp = tmp->next;
				continue;
			}
			else
			{
				// Check for name conflict errors with other values in the enum
				if( globVariables.GetFirst(ns, name, asCCompGlobVarType(type)) )
				{
					asCString str;
					str.Format(TXT_NAME_CONFLICT_s_ALREADY_USED, name.AddressOf());
					WriteError(str, file, tmp);

					tmp = tmp->next;
					if( tmp && tmp->nodeType == snAssignment )
						tmp = tmp->next;
					continue;
				}

				// Check for assignment
				asCScriptNode *asnNode = tmp->next;
				if( asnNode && snAssignment == asnNode->nodeType )
					asnNode->DisconnectParent();
				else
					asnNode = 0;

				// Create the global variable description so the enum value can be evaluated
				sGlobalVariableDescription *gvar = asNEW(sGlobalVariableDescription);
				if( gvar == 0 )
					return asOUT_OF_MEMORY;

				gvar->script             = file;
				gvar->declaredAtNode     = tmp;
				tmp = tmp->next;
				gvar->declaredAtNode->DisconnectParent();
				gvar->initializationNode = asnNode;
				gvar->name               = name;
				gvar->datatype           = type;
				gvar->ns                 = ns;
				// No need to allocate space on the global memory stack since the values are stored in the asCObjectType
				// Set the index to a negative to allow compiler to diferentiate from ordinary global var when compiling the initialization
				gvar->index              = -1;
				gvar->isCompiled         = false;
				gvar->isPureConstant     = true;
				gvar->isEnumValue        = true;
				gvar->constantValue      = 0xdeadbeef;

				// Allocate dummy property so we can compile the value.
				// This will be removed later on so we don't add it to the engine.
				gvar->property = asNEW(asCGlobalProperty);
				if( gvar->property == 0 )
					return asOUT_OF_MEMORY;

				gvar->property->name      = name;
				gvar->property->nameSpace = ns;
				gvar->property->type      = gvar->datatype;
				gvar->property->id        = 0;

				globVariables.Put(gvar);
			}
		}
	}

	node->Destroy(engine);

	return r;
}

int asCBuilder::RegisterTypedef(asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	// Get the native data type
	asCScriptNode *tmp = node->firstChild;
	asASSERT(NULL != tmp && snDataType == tmp->nodeType);
	asCDataType dataType;
	dataType.CreatePrimitive(tmp->tokenType, false);
	dataType.SetTokenType(tmp->tokenType);
	tmp = tmp->next;

	// Grab the name of the typedef
	asASSERT(NULL != tmp && NULL == tmp->next);
	asCString name;
	name.Assign(&file->code[tmp->tokenPos], tmp->tokenLength);

	// If the name is not already in use add it
 	int r = CheckNameConflict(name.AddressOf(), tmp, file, ns, true, false, false);

	asCTypedefType *st = 0;
	if( asSUCCESS == r )
	{
		// Create the new type
		st = asNEW(asCTypedefType)(engine);
		if( st == 0 )
			r = asOUT_OF_MEMORY;
	}

	if( asSUCCESS == r )
	{
		st->flags           = asOBJ_TYPEDEF;
		st->size            = dataType.GetSizeInMemoryBytes();
		st->name            = name;
		st->nameSpace       = ns;
		st->aliasForType    = dataType;
		st->module          = module;

		module->AddTypeDef(st);

		// Store the location of this declaration for reference in name collisions
		sClassDeclaration *decl = asNEW(sClassDeclaration);
		if( decl == 0 )
			r = asOUT_OF_MEMORY;
		else
		{
			decl->name             = name;
			decl->script           = file;
			decl->typeInfo         = st;
			namedTypeDeclarations.PushLast(decl);
		}
	}

	node->Destroy(engine);

	return r;
}

void asCBuilder::GetParsedFunctionDetails(asCScriptNode *node, asCScriptCode *file, asCObjectType *objType, asCString &name, asCDataType &returnType, asCArray<asCString> &parameterNames, asCArray<asCDataType> &parameterTypes, asCArray<asETypeModifiers> &inOutFlags, asCArray<asCString *> &defaultArgs, asSFunctionTraits &funcTraits, asSNameSpace *implicitNamespace)
{
	node = node->firstChild;

	// Is the function shared?
	funcTraits.SetTrait(asTRAIT_SHARED, false);
	funcTraits.SetTrait(asTRAIT_EXTERNAL, false);
	while (node->tokenType == ttIdentifier)
	{
		if (file->TokenEquals(node->tokenPos, node->tokenLength, SHARED_TOKEN))
			funcTraits.SetTrait(asTRAIT_SHARED, true);
		else if (file->TokenEquals(node->tokenPos, node->tokenLength, EXTERNAL_TOKEN))
			funcTraits.SetTrait(asTRAIT_EXTERNAL, true);
		else
			break;
		node = node->next;
	}

	// Is the function a private or protected class method?
	funcTraits.SetTrait(asTRAIT_PRIVATE, false);
	funcTraits.SetTrait(asTRAIT_PROTECTED, false);
	if( node->tokenType == ttPrivate )
	{
		funcTraits.SetTrait(asTRAIT_PRIVATE, true);
		node = node->next;
	}
	else if( node->tokenType == ttProtected )
	{
		funcTraits.SetTrait(asTRAIT_PROTECTED, true);
		node = node->next;
	}

	// Find the name
	funcTraits.SetTrait(asTRAIT_CONSTRUCTOR, false);
	funcTraits.SetTrait(asTRAIT_DESTRUCTOR, false);
	asCScriptNode *n = 0;
	if( node->nodeType == snDataType )
		n = node->next->next;
	else
	{
		// If the first node is a ~ token, then we know it is a destructor
		if( node->tokenType == ttBitNot )
		{
			n = node->next;
			funcTraits.SetTrait(asTRAIT_DESTRUCTOR, true);
		}
		else
		{
			n = node;
			funcTraits.SetTrait(asTRAIT_CONSTRUCTOR, true);
		}
	}
	name.Assign(&file->code[n->tokenPos], n->tokenLength);

	if( !funcTraits.GetTrait(asTRAIT_CONSTRUCTOR) && !funcTraits.GetTrait(asTRAIT_DESTRUCTOR) )
	{
		returnType = CreateDataTypeFromNode(node, file, implicitNamespace, false, objType);
		returnType = ModifyDataTypeFromNode(returnType, node->next, file, 0, 0);

		if( engine->ep.disallowValueAssignForRefType &&
			returnType.GetTypeInfo() &&
			(returnType.GetTypeInfo()->flags & asOBJ_REF) &&
			!(returnType.GetTypeInfo()->flags & asOBJ_SCOPED) &&
			!returnType.IsReference() &&
			!returnType.IsObjectHandle() )
		{
			WriteError(TXT_REF_TYPE_CANT_BE_RETURNED_BY_VAL, file, node);
		}
	}
	else
		returnType = asCDataType::CreatePrimitive(ttVoid, false);

	funcTraits.SetTrait(asTRAIT_CONST, false);
	funcTraits.SetTrait(asTRAIT_FINAL, false);
	funcTraits.SetTrait(asTRAIT_OVERRIDE, false);
	funcTraits.SetTrait(asTRAIT_EXPLICIT, false);
	funcTraits.SetTrait(asTRAIT_PROPERTY, false);

	if( n->next->next )
	{
		asCScriptNode *decorator = n->next->next;

		// Is this a const method?
		if( objType && decorator->tokenType == ttConst )
		{
			funcTraits.SetTrait(asTRAIT_CONST, true);
			decorator = decorator->next;
		}

		while( decorator && decorator->tokenType == ttIdentifier )
		{
			if (objType && file->TokenEquals(decorator->tokenPos, decorator->tokenLength, FINAL_TOKEN))
				funcTraits.SetTrait(asTRAIT_FINAL, true);
			else if (objType && file->TokenEquals(decorator->tokenPos, decorator->tokenLength, OVERRIDE_TOKEN))
				funcTraits.SetTrait(asTRAIT_OVERRIDE, true);
			else if (objType && file->TokenEquals(decorator->tokenPos, decorator->tokenLength, EXPLICIT_TOKEN))
				funcTraits.SetTrait(asTRAIT_EXPLICIT, true);
			else if (file->TokenEquals(decorator->tokenPos, decorator->tokenLength, PROPERTY_TOKEN))
				funcTraits.SetTrait(asTRAIT_PROPERTY, true);
			else if (file->TokenEquals(decorator->tokenPos, decorator->tokenLength, DELETE_TOKEN))
				funcTraits.SetTrait(asTRAIT_DELETED, true);
			else
			{
				asCString msg(&file->code[decorator->tokenPos], decorator->tokenLength);
				msg.Format(TXT_UNEXPECTED_TOKEN_s, msg.AddressOf());
				WriteError(msg.AddressOf(), file, decorator);
			}			
			
			decorator = decorator->next;
		}
	}

	// Count the number of parameters
	int count = 0;
	asCScriptNode *c = n->next->firstChild;
	while( c )
	{
		count++;
		c = c->next->next;
		if( c && c->nodeType == snIdentifier )
			c = c->next;
		if( c && c->nodeType == snExpression )
			c = c->next;
	}

	// Get the parameter types
	parameterNames.Allocate(count, false);
	parameterTypes.Allocate(count, false);
	inOutFlags.Allocate(count, false);
	defaultArgs.Allocate(count, false);
	n = n->next->firstChild;
	while( n )
	{
		asETypeModifiers inOutFlag;
		asCDataType type = CreateDataTypeFromNode(n, file, implicitNamespace, false, objType);
		type = ModifyDataTypeFromNode(type, n->next, file, &inOutFlag, 0);

		if( engine->ep.disallowValueAssignForRefType &&
			type.GetTypeInfo() &&
			(type.GetTypeInfo()->flags & asOBJ_REF) &&
			!(type.GetTypeInfo()->flags & asOBJ_SCOPED) &&
			!type.IsReference() &&
			!type.IsObjectHandle() )
		{
			WriteError(TXT_REF_TYPE_CANT_BE_PASSED_BY_VAL, file, node);
		}

		// Store the parameter type
		parameterTypes.PushLast(type);
		inOutFlags.PushLast(inOutFlag);

		// Move to next parameter
		n = n->next->next;
		if( n && n->nodeType == snIdentifier )
		{
			asCString paramName(&file->code[n->tokenPos], n->tokenLength);
			parameterNames.PushLast(paramName);
			n = n->next;
		}
		else
		{
			// No name was given for the parameter
			parameterNames.PushLast(asCString());
		}

		if( n && n->nodeType == snExpression )
		{
			// Strip out white space and comments to better share the string
			asCString *defaultArgStr = asNEW(asCString);
			if( defaultArgStr )
				*defaultArgStr = GetCleanExpressionString(n, file);
			defaultArgs.PushLast(defaultArgStr);

			n = n->next;
		}
		else
			defaultArgs.PushLast(0);
	}
}
#endif

asCString asCBuilder::GetCleanExpressionString(asCScriptNode *node, asCScriptCode *file)
{
	asASSERT(node && node->nodeType == snExpression);

	asCString str;
	str.Assign(file->code + node->tokenPos, node->tokenLength);

	asCString cleanStr;
	for( asUINT n = 0; n < str.GetLength(); )
	{
		asUINT len = 0;
		asETokenClass tok = engine->ParseToken(str.AddressOf() + n, str.GetLength() - n, &len);

		// Replace comments and whitespace with a single space
		if (tok == asTC_COMMENT || tok == asTC_WHITESPACE)
		{
			if (cleanStr.GetLength() && cleanStr[cleanStr.GetLength() - 1] != ' ')
				cleanStr += " ";
		}
		else
			cleanStr.Concatenate(str.AddressOf() + n, len);

		n += len;
	}

	return cleanStr;
}

#ifndef AS_NO_COMPILER
int asCBuilder::RegisterScriptFunctionFromNode(asCScriptNode *node, asCScriptCode *file, asCObjectType *objType, bool isInterface, bool isGlobalFunction, asSNameSpace *ns, bool isExistingShared, bool isMixin, sClassDeclaration* decl)
{
	asCString                  name;
	asCDataType                returnType;
	asCArray<asCString>        parameterNames;
	asCArray<asCDataType>      parameterTypes;
	asCArray<asETypeModifiers> inOutFlags;
	asCArray<asCString *>      defaultArgs;
	asSFunctionTraits          funcTraits;

	asASSERT( (objType && ns == 0) || isGlobalFunction || isMixin );

	// Set the default namespace
	if( ns == 0 )
	{
		if( objType )
			ns = objType->nameSpace;
		else
			ns = engine->nameSpaces[0];
	}

	GetParsedFunctionDetails(node, file, objType, name, returnType, parameterNames, parameterTypes, inOutFlags, defaultArgs, funcTraits, ns);

	return RegisterScriptFunction(node, file, objType, isInterface, isGlobalFunction, ns, isExistingShared, isMixin, name, returnType, parameterNames, parameterTypes, inOutFlags, defaultArgs, funcTraits, decl);
}

asCScriptFunction *asCBuilder::RegisterLambda(asCScriptNode *node, asCScriptCode *file, asCScriptFunction *funcDef, const asCString &name, asSNameSpace *ns, bool isShared)
{
	// Get the parameter names from the node
	asCArray<asCString> parameterNames;
	asCArray<asCString*> defaultArgs;
	asCScriptNode *args = node->firstChild;
	while( args && args->nodeType != snStatementBlock )
	{
		if (args->nodeType == snUndefined)
		{
			if (args->lastChild && args->lastChild->nodeType == snIdentifier)
			{
				asCString argName;
				argName.Assign(&file->code[args->lastChild->tokenPos], args->lastChild->tokenLength);
				parameterNames.PushLast(argName);
			}
			else
				parameterNames.PushLast("");
			defaultArgs.PushLast(0);
		}
		args = args->next;
	}

	// The statement block for the function must be disconnected, as the builder is going to be the owner of it
	args->DisconnectParent();

	// Get the return and parameter types from the funcDef
	asCString funcName = name;
	asSFunctionTraits traits;
	traits.SetTrait(asTRAIT_SHARED, isShared);
	int r = RegisterScriptFunction(args, file, 0, 0, true, ns, false, false, funcName, funcDef->returnType, parameterNames, funcDef->parameterTypes, funcDef->inOutFlags, defaultArgs, traits, 0);
	if( r < 0 )
		return 0;

	// Return the function that was just created (but that will be compiled later)
	return engine->scriptFunctions[functions[functions.GetLength()-1]->funcId];
}

int asCBuilder::RegisterScriptFunction(asCScriptNode *node, asCScriptCode *file, asCObjectType *objType, bool isInterface, bool isGlobalFunction, asSNameSpace *ns, bool isExistingShared, bool isMixin, asCString &name, asCDataType &returnType, asCArray<asCString> &parameterNames, asCArray<asCDataType> &parameterTypes, asCArray<asETypeModifiers> &inOutFlags, asCArray<asCString *> &defaultArgs, asSFunctionTraits funcTraits, sClassDeclaration* classDecl)
{
	// Determine default namespace if not specified
	if( ns == 0 )
	{
		if( objType )
			ns = objType->nameSpace;
		else
			ns = engine->nameSpaces[0];
	}

	if( isExistingShared )
	{
		asASSERT( objType );

		// Should validate that the function really exists in the class/interface
		bool found = false;
		if(funcTraits.GetTrait(asTRAIT_CONSTRUCTOR) || funcTraits.GetTrait(asTRAIT_DESTRUCTOR) )
		{
			// TODO: shared: Should check the existance of these too
			found = true;
		}
		else
		{
			for( asUINT n = 0; n < objType->methods.GetLength(); n++ )
			{
				asCScriptFunction *func = engine->scriptFunctions[objType->methods[n]];
				if( func->name == name &&
					func->IsSignatureExceptNameEqual(returnType, parameterTypes, inOutFlags, objType, funcTraits.GetTrait(asTRAIT_CONST)) )
				{
					// Add the shared function in this module too
					module->AddScriptFunction(func);

					found = true;
					break;
				}
			}
		}

		if( !found )
		{
			asCString str;
			str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, objType->GetName());
			WriteError(str, file, node);
		}

		// Free the default args
		for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
			if( defaultArgs[n] )
				asDELETE(defaultArgs[n], asCString);

		node->Destroy(engine);
		return 0;
	}

	// Check for name conflicts
	if( !funcTraits.GetTrait(asTRAIT_CONSTRUCTOR) && !funcTraits.GetTrait(asTRAIT_DESTRUCTOR) )
	{
		if( objType )
		{
			CheckNameConflictMember(objType, name.AddressOf(), node, file, false, false);

			if( name == objType->name )
				WriteError(TXT_METHOD_CANT_HAVE_NAME_OF_CLASS, file, node);
		}
		else
			CheckNameConflict(name.AddressOf(), node, file, ns, false, false, false);
	}
	else
	{
		if( isMixin )
		{
			// Mixins cannot implement constructors/destructors
			WriteError(TXT_MIXIN_CANNOT_HAVE_CONSTRUCTOR, file, node);

			// Free the default args
			for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
				if( defaultArgs[n] )
					asDELETE(defaultArgs[n], asCString);

			node->Destroy(engine);
			return 0;
		}

		// Verify that the name of the constructor/destructor is the same as the class
		if( name != objType->name )
		{
			asCString str;
			if(funcTraits.GetTrait(asTRAIT_DESTRUCTOR) )
				str.Format(TXT_DESTRUCTOR_s_s_NAME_ERROR, objType->name.AddressOf(), name.AddressOf());
			else
				str.Format(TXT_METHOD_s_s_HAS_NO_RETURN_TYPE, objType->name.AddressOf(), name.AddressOf());
			WriteError(str, file, node);
		}

		if(funcTraits.GetTrait(asTRAIT_DESTRUCTOR))
			name = "~" + name;
	}

	// Validate virtual properties signature
	if( funcTraits.GetTrait(asTRAIT_PROPERTY) )
	{
		asCScriptFunction func(engine, module, asFUNC_SCRIPT);
		func.name           = name;
		func.nameSpace      = ns;
		func.objectType     = objType;
		if( objType )
			objType->AddRefInternal();
		func.traits         = funcTraits;
		func.returnType     = returnType;
		func.parameterTypes = parameterTypes;
		
		int r = ValidateVirtualProperty(&func);
		if( r < 0 )
		{
			asCString str;
			if( r == -2 || r == -3 )
				str.Format(TXT_INVALID_SIG_FOR_VIRTPROP);
			else if( r == -4 )
				str.Format(TXT_GET_SET_ACCESSOR_TYPE_MISMATCH_FOR_s, name.SubString(4).AddressOf());
			else if( r == -5 )
				str.Format(TXT_NAME_CONFLICT_s_ALREADY_USED, name.SubString(4).AddressOf());
			WriteError(str, file, node);
		}
		
		// Set the type as dummy so the clean-up works correctly
		func.funcType = asFUNC_DUMMY;
	}
	
	// If the method is deleted, then don't register it
	if (funcTraits.GetTrait(asTRAIT_DELETED))
	{
		// Delete can only be used for default constructor, default copy constructor, and default copy operator
		if (classDecl && funcTraits.GetTrait(asTRAIT_CONSTRUCTOR) && parameterTypes.GetLength() == 0)
		{
			classDecl->isDefaultConstructorDeleted = true;

			if (objType->beh.construct && objType->beh.construct != engine->scriptTypeBehaviours.beh.construct)
				WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
		}
		else if (classDecl && funcTraits.GetTrait(asTRAIT_CONSTRUCTOR) && parameterTypes.GetLength() == 1 && parameterTypes[0].GetTypeInfo() == objType)
		{
			classDecl->isDefaultCopyConstructorDeleted = true;

			if (objType->beh.copyconstruct)
				WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
		}
		else if (classDecl && name == "opAssign" && parameterTypes.GetLength() == 1 && parameterTypes[0].GetTypeInfo() == objType)
		{
			classDecl->isDefaultCopyDeleted = true;

			if (objType->beh.copy && objType->beh.copy != engine->scriptTypeBehaviours.beh.copy)
				WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
		}
		else
			WriteError(TXT_CANNOT_DELETE_NON_AUTO_FUNC, file, node);

		// Deleted functions must not have implementation
		if (node && node->lastChild && node->lastChild->nodeType == snStatementBlock)
			WriteError(TXT_DELETED_FUNC_CANT_HAVE_IMPL, file, node);

		// Free the default args
		for (asUINT n = 0; n < defaultArgs.GetLength(); n++)
			if (defaultArgs[n])
				asDELETE(defaultArgs[n], asCString);

		node->Destroy(engine);

		return 0;
	}

	isExistingShared = false;
	int funcId = engine->GetNextScriptFunctionId();
	if( !isInterface )
	{
		sFunctionDescription *func = asNEW(sFunctionDescription);
		if( func == 0 )
		{
			// Free the default args
			for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
				if( defaultArgs[n] )
					asDELETE(defaultArgs[n], asCString);

			return asOUT_OF_MEMORY;
		}

		functions.PushLast(func);

		func->script            = file;
		func->node              = node;
		func->name              = name;
		func->objType           = objType;
		func->funcId            = funcId;
		func->isExistingShared  = false;
		func->paramNames        = parameterNames;

		if(funcTraits.GetTrait(asTRAIT_SHARED))
		{
			// Look for a pre-existing shared function with the same signature
			for( asUINT n = 0; n < engine->scriptFunctions.GetLength(); n++ )
			{
				asCScriptFunction *f = engine->scriptFunctions[n];
				if( f &&
					f->IsShared() &&
					f->name == name &&
					f->nameSpace == ns &&
					f->objectType == objType &&
					f->IsSignatureExceptNameEqual(returnType, parameterTypes, inOutFlags, 0, false) )
				{
					funcId = func->funcId = f->id;
					isExistingShared = func->isExistingShared = true;
					break;
				}
			}
		}

		// Remember if the function was declared as external so the saved bytecode can be flagged accordingly
		if (funcTraits.GetTrait(asTRAIT_EXTERNAL) && func->isExistingShared)
			module->m_externalFunctions.PushLast(engine->scriptFunctions[func->funcId]);

		if (funcTraits.GetTrait(asTRAIT_EXTERNAL) && !func->isExistingShared)
		{
			// Mark it as existing shared to avoid compiling it
			func->isExistingShared = true;

			asCString str;
			str.Format(TXT_EXTERNAL_SHARED_s_NOT_FOUND, name.AddressOf());
			WriteError(str, file, node);
		}

		// External shared function must not try to redefine the interface
		if (funcTraits.GetTrait(asTRAIT_EXTERNAL) && !(node == 0 || node->tokenType == ttEndStatement || (node->lastChild && node->lastChild->tokenType == ttEndStatement)))
		{
			asCString str;
			str.Format(TXT_EXTERNAL_SHARED_s_CANNOT_REDEF, name.AddressOf());
			WriteError(str, file, node);
		}
		else if (!funcTraits.GetTrait(asTRAIT_EXTERNAL) && (node == 0 || !(node->nodeType == snStatementBlock || (node->lastChild && node->lastChild->nodeType == snStatementBlock))) )
		{
			asCString str;
			str.Format(TXT_MISSING_DEFINITION_OF_s, name.AddressOf());
			WriteError(str, file, node);
		}
	}

	// Destructors may not have any parameters
	if (funcTraits.GetTrait(asTRAIT_DESTRUCTOR) && parameterTypes.GetLength() > 0)
		WriteError(TXT_DESTRUCTOR_MAY_NOT_HAVE_PARM, file, node);

	// If a function, class, or interface is shared then only shared types may be used in the signature
	if( (objType && objType->IsShared()) || funcTraits.GetTrait(asTRAIT_SHARED))
	{
		asCTypeInfo *ti = returnType.GetTypeInfo();
		if( ti && !ti->IsShared() )
		{
			asCString msg;
			msg.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, ti->name.AddressOf());
			WriteError(msg, file, node);
		}

		for( asUINT p = 0; p < parameterTypes.GetLength(); ++p )
		{
			ti = parameterTypes[p].GetTypeInfo();
			if( ti && !ti->IsShared() )
			{
				asCString msg;
				msg.Format(TXT_SHARED_CANNOT_USE_NON_SHARED_TYPE_s, ti->name.AddressOf());
				WriteError(msg, file, node);
			}
		}
	}

	// Check that the same function hasn't been registered already in the namespace
	asCArray<int> funcs;
	if( objType )
		GetObjectMethodDescriptions(name.AddressOf(), objType, funcs, false);
	else
		GetFunctionDescriptions(name.AddressOf(), funcs, ns);
	if( objType && (name == "opConv" || name == "opImplConv" || name == "opCast" || name == "opImplCast") && parameterTypes.GetLength() == 0 )
	{
		// opConv and opCast are special methods used for type casts
		for( asUINT n = 0; n < funcs.GetLength(); ++n )
		{
			asCScriptFunction *func = GetFunctionDescription(funcs[n]);
			if( func->IsSignatureExceptNameEqual(returnType, parameterTypes, inOutFlags, objType, funcTraits.GetTrait(asTRAIT_CONST)) )
			{
				// TODO: clean up: Reuse the same error handling for both opConv and normal methods
				if( isMixin )
				{
					// Clean up the memory, as the function will not be registered
					if( node )
						node->Destroy(engine);
					sFunctionDescription *funcDesc = functions.PopLast();
					asDELETE(funcDesc, sFunctionDescription);

					// Free the default args
					for( n = 0; n < defaultArgs.GetLength(); n++ )
						if( defaultArgs[n] )
							asDELETE(defaultArgs[n], asCString);

					return 0;
				}

				WriteError(TXT_FUNCTION_ALREADY_EXIST, file, node);
				break;
			}
		}
	}
	else
	{
		for( asUINT n = 0; n < funcs.GetLength(); ++n )
		{
			asCScriptFunction *func = GetFunctionDescription(funcs[n]);
			if( func->IsSignatureExceptNameAndReturnTypeEqual(parameterTypes, inOutFlags, objType, funcTraits.GetTrait(asTRAIT_CONST)) )
			{
				if( isMixin )
				{
					// Clean up the memory, as the function will not be registered
					if( node )
						node->Destroy(engine);
					sFunctionDescription *funcDesc = functions.PopLast();
					asDELETE(funcDesc, sFunctionDescription);

					// Free the default args
					for( n = 0; n < defaultArgs.GetLength(); n++ )
						if( defaultArgs[n] )
							asDELETE(defaultArgs[n], asCString);

					return 0;
				}

				WriteError(TXT_FUNCTION_ALREADY_EXIST, file, node);
				break;
			}
		}
	}

	// Register the function
	if( isExistingShared )
	{
		// Delete the default args as they won't be used anymore
		for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
			if( defaultArgs[n] )
				asDELETE(defaultArgs[n], asCString);

		asCScriptFunction *f = engine->scriptFunctions[funcId];
		module->AddScriptFunction(f);

		// TODO: clean up: This should be done by AddScriptFunction() itself
		module->m_globalFunctions.Put(f);
	}
	else
	{
		int row = 0, col = 0;
		if( node )
			file->ConvertPosToRowCol(node->tokenPos, &row, &col);
		module->AddScriptFunction(file->idx, (row&0xFFFFF)|((col&0xFFF)<<20), funcId, name, returnType, parameterTypes, parameterNames, inOutFlags, defaultArgs, isInterface, objType, isGlobalFunction, funcTraits, ns);
	}

	// Make sure the default args are declared correctly
	ValidateDefaultArgs(file, node, engine->scriptFunctions[funcId]);
	CheckForConflictsDueToDefaultArgs(file, node, engine->scriptFunctions[funcId], objType);

	if( objType )
	{
		asASSERT( !isExistingShared );

		engine->scriptFunctions[funcId]->AddRefInternal();
		if(funcTraits.GetTrait(asTRAIT_CONSTRUCTOR))
		{
			int factoryId = engine->GetNextScriptFunctionId();
			if( parameterTypes.GetLength() == 0 )
			{
				// Overload the default constructor
				engine->scriptFunctions[objType->beh.construct]->ReleaseInternal();
				objType->beh.construct = funcId;
				objType->beh.constructors[0] = funcId;

				// Register the default factory as well
				engine->scriptFunctions[objType->beh.factory]->ReleaseInternal();
				objType->beh.factory = factoryId;
				objType->beh.factories[0] = factoryId;

				if( classDecl->isDefaultConstructorDeleted )
					WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
			}
			else
			{
				// The copy constructor needs to be marked for easy finding
				if( parameterTypes.GetLength() == 1 && 
				    parameterTypes[0].GetTypeInfo() == objType && 
					(parameterTypes[0].IsReference() || parameterTypes[0].IsObjectHandle()) )
				{
					// Verify that there are not multiple options matching the copy constructor
					// TODO: Need a better message, since the parameters can be slightly different, e.g. & vs @
					if( objType->beh.copyconstruct )
						WriteError(TXT_FUNCTION_ALREADY_EXIST, file, node);

					objType->beh.copyconstruct = funcId;
					objType->beh.copyfactory = factoryId;

					if (classDecl->isDefaultCopyConstructorDeleted)
						WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
				}

				// Register as a normal constructor
				objType->beh.constructors.PushLast(funcId);

				// Register the factory as well
				objType->beh.factories.PushLast(factoryId);
			}

			// We must copy the default arg strings to avoid deleting the same object multiple times
			for( asUINT n = 0; n < defaultArgs.GetLength(); n++ )
				if( defaultArgs[n] )
					defaultArgs[n] = asNEW(asCString)(*defaultArgs[n]);

			asCDataType dt = asCDataType::CreateObjectHandle(objType, false);
			module->AddScriptFunction(file->idx, engine->scriptFunctions[funcId]->scriptData->declaredAt, factoryId, name, dt, parameterTypes, parameterNames, inOutFlags, defaultArgs, false, 0, false, funcTraits, objType->nameSpace);

			// If the object is shared, then the factory must also be marked as shared
			if( objType->flags & asOBJ_SHARED )
				engine->scriptFunctions[factoryId]->SetShared(true);

			// Add a dummy function to the builder so that it doesn't mix up the fund Ids
			functions.PushLast(0);

			// Compile the factory immediately
			asCCompiler compiler(engine);
			compiler.CompileFactory(this, file, engine->scriptFunctions[factoryId]);
			engine->scriptFunctions[factoryId]->AddRefInternal();
		}
		else if(funcTraits.GetTrait(asTRAIT_DESTRUCTOR))
			objType->beh.destruct = funcId;
		else
		{
			// If the method is the assignment operator we need to replace the default implementation
			asCScriptFunction *f = engine->scriptFunctions[funcId];
			if( f->name == "opAssign" && f->parameterTypes.GetLength() == 1 &&
				f->parameterTypes[0].GetTypeInfo() == f->objectType &&
				(f->inOutFlags[0] & asTM_INREF) )
			{
				engine->scriptFunctions[objType->beh.copy]->ReleaseInternal();
				objType->beh.copy = funcId;
				f->AddRefInternal();

				if (classDecl->isDefaultCopyDeleted)
					WriteError(TXT_CANNOT_DEFINE_FUNC_THAT_IS_DELETED, file, node);
			}

			objType->methods.PushLast(funcId);
		}
	}

	// We need to delete the node already if this is an interface method
	if( isInterface && node )
		node->Destroy(engine);

	return 0;
}

int asCBuilder::RegisterVirtualProperty(asCScriptNode *node, asCScriptCode *file, asCObjectType *objType, bool isInterface, bool isGlobalFunction, asSNameSpace *ns, bool isExistingShared)
{
	if( engine->ep.propertyAccessorMode < 2 )
	{
		WriteError(TXT_PROPERTY_ACCESSOR_DISABLED, file, node);
		node->Destroy(engine);
		return 0;
	}

	asASSERT( (objType && ns == 0) || isGlobalFunction );

	if( ns == 0 )
	{
		if( objType )
			ns = objType->nameSpace;
		else
			ns = engine->nameSpaces[0];
	}

	bool isPrivate = false, isProtected = false;
	asCString emulatedName;
	asCDataType emulatedType;

	asCScriptNode *mainNode = node;
	node = node->firstChild;

	if( !isGlobalFunction && node->tokenType == ttPrivate )
	{
		isPrivate = true;
		node = node->next;
	}
	else if( !isGlobalFunction && node->tokenType == ttProtected )
	{
		isProtected = true;
		node = node->next;
	}

	emulatedType = CreateDataTypeFromNode(node, file, ns);
	emulatedType = ModifyDataTypeFromNode(emulatedType, node->next, file, 0, 0);
	node = node->next->next;
	emulatedName.Assign(&file->code[node->tokenPos], node->tokenLength);

	if( node->next == 0 )
		WriteError(TXT_PROPERTY_WITHOUT_ACCESSOR, file, node);

	node = node->next;
	while (node)
	{
		asCScriptNode             *next = node->next;
		asCScriptNode             *funcNode = 0;
		bool                       success = false;
		asSFunctionTraits          funcTraits;
		asCDataType                returnType;
		asCArray<asCString>        paramNames;
		asCArray<asCDataType>      paramTypes;
		asCArray<asETypeModifiers> paramModifiers;
		asCArray<asCString*>       defaultArgs;
		asCString                  name;

		funcTraits.SetTrait(asTRAIT_PRIVATE, isPrivate);
		funcTraits.SetTrait(asTRAIT_PROTECTED, isProtected);
		funcTraits.SetTrait(asTRAIT_PROPERTY, true);

		if (node->firstChild->nodeType == snIdentifier && file->TokenEquals(node->firstChild->tokenPos, node->firstChild->tokenLength, GET_TOKEN))
			name = "get_";
		else if (node->firstChild->nodeType == snIdentifier && file->TokenEquals(node->firstChild->tokenPos, node->firstChild->tokenLength, SET_TOKEN))
			name = "set_";
		else
			WriteError(TXT_UNRECOGNIZED_VIRTUAL_PROPERTY_NODE, file, node);

		if (name != "")
		{
			success = true;
			funcNode = node->firstChild->next;

			if (funcNode && funcNode->tokenType == ttConst)
			{
				funcTraits.SetTrait(asTRAIT_CONST, true);
				funcNode = funcNode->next;
			}

			while (funcNode && (funcNode->nodeType != snStatementBlock && funcNode->tokenType != ttEndStatement))
			{
				if (funcNode->tokenType == ttIdentifier && file->TokenEquals(funcNode->tokenPos, funcNode->tokenLength, FINAL_TOKEN))
					funcTraits.SetTrait(asTRAIT_FINAL, true);
				else if (funcNode->tokenType == ttIdentifier && file->TokenEquals(funcNode->tokenPos, funcNode->tokenLength, OVERRIDE_TOKEN))
					funcTraits.SetTrait(asTRAIT_OVERRIDE, true);
				else
				{
					asCString msg(&file->code[funcNode->tokenPos], funcNode->tokenLength);;
					msg.Format(TXT_UNEXPECTED_TOKEN_s, msg.AddressOf());
					WriteError(msg.AddressOf(), file, node);
				}

				funcNode = funcNode->next;
			}

			if (funcNode)
				funcNode->DisconnectParent();

			if ((funcNode == 0 || funcNode->nodeType != snStatementBlock) && (objType == 0 || !objType->IsInterface()))
			{
				// TODO: getset: If no implementation is supplied the builder should provide an automatically generated implementation
				//               The compiler needs to be able to handle the different types, primitive, value type, and handle
				//               The code is also different for global property accessors
				WriteError(TXT_PROPERTY_ACCESSOR_MUST_BE_IMPLEMENTED, file, node);
			}

			if (name == "get_")
			{
				// Setup the signature for the get accessor method
				returnType = emulatedType;
				name = "get_" + emulatedName;
			}
			else if (name == "set_")
			{
				// Setup the signature for the set accessor method
				returnType = asCDataType::CreatePrimitive(ttVoid, false);
				paramModifiers.PushLast(asTM_NONE);
				paramNames.PushLast("value");
				paramTypes.PushLast(emulatedType);
				defaultArgs.PushLast(0);
				name = "set_" + emulatedName;
			}
		}

		if( success )
		{
			if( !isExistingShared )
				RegisterScriptFunction(funcNode ? funcNode : node, file, objType, isInterface, isGlobalFunction, ns, false, false, name, returnType, paramNames, paramTypes, paramModifiers, defaultArgs, funcTraits, 0);
			else
			{
				// Free the funcNode as it won't be used
				if( funcNode ) funcNode->Destroy(engine);

				// Should validate that the function really exists in the class/interface
				bool found = false;
				for( asUINT n = 0; n < objType->methods.GetLength(); n++ )
				{
					asCScriptFunction *func = engine->scriptFunctions[objType->methods[n]];
					if( func->name == name &&
						func->IsSignatureExceptNameEqual(returnType, paramTypes, paramModifiers, objType, funcTraits.GetTrait(asTRAIT_CONST)) )
					{
						found = true;
						break;
					}
				}

				if( !found )
				{
					asCString str;
					str.Format(TXT_SHARED_s_DOESNT_MATCH_ORIGINAL, objType->GetName());
					WriteError(str, file, node);
				}
			}
		}

		node = next;
	};

	mainNode->Destroy(engine);

	return 0;
}

int asCBuilder::RegisterImportedFunction(int importID, asCScriptNode *node, asCScriptCode *file, asSNameSpace *ns)
{
	asCString                  name;
	asCDataType                returnType;
	asCArray<asCString>        parameterNames;
	asCArray<asCDataType>      parameterTypes;
	asCArray<asETypeModifiers> inOutFlags;
	asCArray<asCString *>      defaultArgs;
	asSFunctionTraits          funcTraits;

	if( ns == 0 )
		ns = engine->nameSpaces[0];

	GetParsedFunctionDetails(node->firstChild, file, 0, name, returnType, parameterNames, parameterTypes, inOutFlags, defaultArgs, funcTraits, ns);
	CheckNameConflict(name.AddressOf(), node, file, ns, false, false, false);

	// Check that the same function hasn't been registered already in the namespace
	asCArray<int> funcs;
	GetFunctionDescriptions(name.AddressOf(), funcs, ns);
	for( asUINT n = 0; n < funcs.GetLength(); ++n )
	{
		asCScriptFunction *func = GetFunctionDescription(funcs[n]);
		if( func->IsSignatureExceptNameAndReturnTypeEqual(parameterTypes, inOutFlags, 0, false) )
		{
			WriteError(TXT_FUNCTION_ALREADY_EXIST, file, node);
			break;
		}
	}

	// Check for invalid function traits
	if (funcTraits.GetTrait(asTRAIT_DELETED))
		WriteError(TXT_CANNOT_DELETE_NON_AUTO_FUNC, file, node);

	// Read the module name as well
	asCScriptNode *nd = node->lastChild;
	asASSERT( nd->nodeType == snConstant && nd->tokenType == ttStringConstant );
	asCString moduleName;
	moduleName.Assign(&file->code[nd->tokenPos+1], nd->tokenLength-2);

	node->Destroy(engine);

	// Register the function
	module->AddImportedFunction(importID, name, returnType, parameterTypes, inOutFlags, defaultArgs, funcTraits, ns, moduleName);

	return 0;
}

asCScriptFunction *asCBuilder::GetFunctionDescription(int id)
{
	// TODO: import: This should be improved when the imported functions are removed
	// Get the description from the engine
	if( (id & FUNC_IMPORTED) == 0 )
		return engine->scriptFunctions[id];
	else
		return engine->importedFunctions[id & ~FUNC_IMPORTED]->importedFunctionSignature;
}

void asCBuilder::GetFunctionDescriptions(const char *name, asCArray<int> &funcs, asSNameSpace *ns)
{
	asUINT n;

	// Get the script declared global functions
	const asCArray<unsigned int> &idxs = module->m_globalFunctions.GetIndexes(ns, name);
	for( n = 0; n < idxs.GetLength(); n++ )
	{
		const asCScriptFunction *f = module->m_globalFunctions.Get(idxs[n]);
		asASSERT( f->objectType == 0 );
		funcs.PushLast(f->id);
	}

	// Add the imported functions
	// TODO: optimize: Linear search: This is probably not that critial. Also bindInformation will probably be removed in near future
	for( n = 0; n < module->m_bindInformations.GetLength(); n++ )
	{
		if( module->m_bindInformations[n]->importedFunctionSignature->name == name &&
			module->m_bindInformations[n]->importedFunctionSignature->nameSpace == ns )
			funcs.PushLast(module->m_bindInformations[n]->importedFunctionSignature->id);
	}

	// Add the registered global functions
	const asCArray<unsigned int> &idxs2 = engine->registeredGlobalFuncs.GetIndexes(ns, name);
	for( n = 0; n < idxs2.GetLength(); n++ )
	{
		asCScriptFunction *f = engine->registeredGlobalFuncs.Get(idxs2[n]);

		// Verify if the module has access to the function
		if( module->m_accessMask & f->accessMask )
		{
			funcs.PushLast(f->id);
		}
	}
}

// scope is only informed when looking for a base class' method
void asCBuilder::GetObjectMethodDescriptions(const char *name, asCObjectType *objectType, asCArray<int> &methods, bool objIsConst, const asCString &scope, asCScriptNode *errNode, asCScriptCode *script)
{
	asASSERT(objectType);

	if( scope != "" )
	{
		// If searching with a scope informed, then the node and script must also be informed for potential error reporting
		asASSERT( errNode && script );

		// If the scope contains ::identifier, then use the last identifier as the class name and the rest of it as the namespace
		// TODO: child funcdef: A scope can include a template type, e.g. array<ns::type>
		int n = scope.FindLast("::");
		asCString className = n >= 0 ? scope.SubString(n+2) : scope;
		asCString nsName = n >= 0 ? scope.SubString(0, n) : asCString("");

		// If a namespace was specifically defined, then this must be used
		asSNameSpace *ns = 0;
		if (n >= 0)
		{
			if (nsName == "")
				ns = engine->nameSpaces[0];
			else
				ns = GetNameSpaceByString(nsName, objectType->nameSpace, errNode, script, 0, false);

			// If the namespace isn't found return silently and let the calling
			// function report the error if it cannot resolve the symbol
			if (ns == 0)
				return;
		}

		// Find the base class with the specified scope
		while (objectType)
		{
			// If the name and namespace matches it is the correct class. If no
			// specific namespace was given, then don't compare the namespace
			if (objectType->name == className && (ns == 0 || objectType->nameSpace == ns))
				break;

			objectType = objectType->derivedFrom;
		}

		// If the scope is not any of the base classes, then return no methods
		if( objectType == 0 )
			return;
	}

	// Find the methods in the object that match the name
	// TODO: optimize: Improve linear search
	for( asUINT n = 0; n < objectType->methods.GetLength(); n++ )
	{
		asCScriptFunction *func = engine->scriptFunctions[objectType->methods[n]];
		if( func->name == name &&
			(!objIsConst || func->IsReadOnly()) &&
			(func->accessMask & module->m_accessMask) )
		{
			// When the scope is defined the returned methods should be the true methods, not the virtual method stubs
			if( scope == "" )
				methods.PushLast(engine->scriptFunctions[objectType->methods[n]]->id);
			else
			{
				asCScriptFunction *f = engine->scriptFunctions[objectType->methods[n]];
				if( f && f->funcType == asFUNC_VIRTUAL )
					f = objectType->virtualFunctionTable[f->vfTableIdx];
				methods.PushLast(f->id);
			}
		}
	}
}
#endif

void asCBuilder::WriteInfo(const asCString &scriptname, const asCString &message, int r, int c, bool pre)
{
	// Need to store the pre message in a structure
	if( pre )
	{
		engine->preMessage.isSet      = true;
		engine->preMessage.c          = c;
		engine->preMessage.r          = r;
		engine->preMessage.message    = message;
		engine->preMessage.scriptname = scriptname;
	}
	else
	{
		engine->preMessage.isSet = false;

		if( !silent )
			engine->WriteMessage(scriptname.AddressOf(), r, c, asMSGTYPE_INFORMATION, message.AddressOf());
	}
}

void asCBuilder::WriteInfo(const asCString &message, asCScriptCode *file, asCScriptNode *node)
{
	int r = 0, c = 0;
	if( node )
		file->ConvertPosToRowCol(node->tokenPos, &r, &c);

	WriteInfo(file->name, message, r, c, false);
}

void asCBuilder::WriteError(const asCString &message, asCScriptCode *file, asCScriptNode *node)
{
	int r = 0, c = 0;
	if( node && file )
		file->ConvertPosToRowCol(node->tokenPos, &r, &c);

	WriteError(file ? file->name : asCString(""), message, r, c);
}

void asCBuilder::WriteError(const asCString &scriptname, const asCString &message, int r, int c)
{
	numErrors++;

	if( !silent )
		engine->WriteMessage(scriptname.AddressOf(), r, c, asMSGTYPE_ERROR, message.AddressOf());
}

void asCBuilder::WriteWarning(const asCString &scriptname, const asCString &message, int r, int c)
{
	if( engine->ep.compilerWarnings )
	{
		numWarnings++;

		if( !silent )
			engine->WriteMessage(scriptname.AddressOf(), r, c, asMSGTYPE_WARNING, message.AddressOf());
	}
}

void asCBuilder::WriteWarning(const asCString &message, asCScriptCode *file, asCScriptNode *node)
{
	int r = 0, c = 0;
	if( node && file )
		file->ConvertPosToRowCol(node->tokenPos, &r, &c);

	WriteWarning(file ? file->name : asCString(""), message, r, c);
}

// TODO: child funcdef: Should try to eliminate this function. GetNameSpaceFromNode is more complete
asCString asCBuilder::GetScopeFromNode(asCScriptNode *node, asCScriptCode *script, asCScriptNode **next)
{
	if (node->nodeType != snScope)
	{
		if (next)
			*next = node;
		return "";
	}

	asCString scope;
	asCScriptNode *sn = node->firstChild;
	if( sn->tokenType == ttScope )
	{
		scope = "::";
		sn = sn->next;
	}

	// TODO: child funcdef: A scope can have a template type as the innermost
	while( sn && sn->next && sn->next->tokenType == ttScope )
	{
		asCString tmp;
		tmp.Assign(&script->code[sn->tokenPos], sn->tokenLength);
		if( scope != "" && scope != "::" )
			scope += "::";
		scope += tmp;
		sn = sn->next->next;
	}

	if( next )
		*next = node->next;

	return scope;
}

bool asCBuilder::FindObjectTypeOrMixinInNsHierarchy(const asCString& name, asSNameSpace* startNs, bool isExplicitNs, asCScriptNode *errNode, asCScriptCode *script, asCObjectType **outObjType, sMixinClass **outMixin)
{
	if (outObjType) *outObjType = 0;
	if (outMixin) *outMixin = 0;

	asSNameSpace* ns = startNs;

	// If not explicit scope, then search the visible namespaces, then search the parent namespaces
	asCArray<asSNameSpace*> pendingNamespaces;
	asCArray<asSNameSpace*> visitedNamespaces;
	bool checkAmbiguousSymbols = !isExplicitNs;
	asSNameSpace* parentNs = engine->GetParentNameSpace(ns);

	asCObjectType* objType = 0;
	sMixinClass* mixin = 0;
	while (ns)
	{
		if (!visitedNamespaces.Exists(ns))
		{
			visitedNamespaces.PushLast(ns);

			if (!checkAmbiguousSymbols)
			{
				objType = GetObjectType(name.AddressOf(), ns);
#ifndef AS_NO_COMPILER
				if (objType == 0 && outMixin)
					mixin = GetMixinClass(name.AddressOf(), ns);
#endif

				if (objType || mixin)
					break;
			}
			else
			{
				asCObjectType* ot = GetObjectType(name.AddressOf(), ns);
				sMixinClass* m = 0;
#ifndef AS_NO_COMPILER
				if (ot == 0 && outMixin)
					m = GetMixinClass(name.AddressOf(), ns);
#endif
				if (ot || m)
				{
					if (objType || mixin)
					{
						asCString msg;
						msg.Format(TXT_AMBIGUOUS_SYMBOL_NAME_s, name.AddressOf());
						WriteError(msg, script, errNode);
						return false;
					}

					objType = ot;
					mixin = m;
				}
			}
		}

		AddVisibleNamespaces(ns, visitedNamespaces, pendingNamespaces);
		ns = FindNextVisibleNamespace(visitedNamespaces, pendingNamespaces, parentNs, &checkAmbiguousSymbols);
		if (ns == parentNs)
		{
			// Only move to the parent namespace if the object type hasn't been found yet
			if (objType || mixin)
				break;

			parentNs = engine->GetParentNameSpace(ns);
		}
	}

	if (outObjType) *outObjType = objType;
	if (outMixin) *outMixin = mixin;
	return true;
}

asSNameSpace *asCBuilder::GetNameSpaceFromNode(asCScriptNode *node, asCScriptCode *script, asSNameSpace *implicitNs, asCScriptNode **next, asCObjectType **objType, bool *isExplicitNs)
{
	if (objType)
		*objType = 0;

	// If no scope has been informed, then return the implicit namespace
	if (node->nodeType != snScope)
	{
		if (next)
			*next = node;
		if (isExplicitNs) 
			*isExplicitNs = false;
		return implicitNs ? implicitNs : engine->nameSpaces[0];
	}

	if (isExplicitNs)
		*isExplicitNs = true;

	if (next)
		*next = node->next;

	asCString scope;
	asCScriptNode *sn = node->firstChild;
	if (sn && sn->tokenType == ttScope)
	{
		scope = "::";
		sn = sn->next;
	}

	while (sn)
	{
		if (sn->next->tokenType == ttScope)
		{
			asCString tmp;
			tmp.Assign(&script->code[sn->tokenPos], sn->tokenLength);
			if (scope != "" && scope != "::")
				scope += "::";
			scope += tmp;
			sn = sn->next->next;
		}
		else
		{
			// This is a template type
			asASSERT(sn->next->nodeType == snDataType);

			asSNameSpace *ns = implicitNs;
			if (scope != "")
				ns = engine->FindNameSpace(scope.AddressOf());

			asCString templateName(&script->code[sn->tokenPos], sn->tokenLength);

			asCObjectType* templateType = 0;
			bool ok = FindObjectTypeOrMixinInNsHierarchy(templateName, ns, scope != "", sn->next, script, &templateType, 0);

			if (!ok || templateType == 0 || (templateType->flags & asOBJ_TEMPLATE) == 0)
			{
				// TODO: child funcdef: Report error
				return ns;
			}

			if (objType)
				*objType = GetTemplateInstanceFromNode(sn, script, templateType, implicitNs, 0);

			// Return no namespace, since this is an object type
			return 0;
		}
	}

	// If the scope doesn't start with '::' then prepend the implicit scope to give the full scope
	asCString originalScope = scope;
	if (scope.SubString(0, 2) != "::" && implicitNs && implicitNs->name.GetLength() > 0 )
		scope = implicitNs->name + "::" + scope;

	asCTypeInfo *ti = 0;
	asSNameSpace *ns = GetNameSpaceByString(scope, implicitNs ? implicitNs : engine->nameSpaces[0], node, script, &ti, scope == originalScope);

	// If the namespace wasn't found try again with the original scope as it may be a fully identified scope
	if (scope != originalScope && ns == 0)
		ns = GetNameSpaceByString(originalScope, implicitNs ? implicitNs : engine->nameSpaces[0], node, script, &ti, true);

	if (ti && objType)
		*objType = CastToObjectType(ti);

	return ns;
}

asSNameSpace *asCBuilder::GetNameSpaceByString(const asCString &nsName, asSNameSpace *implicitNs, asCScriptNode *errNode, asCScriptCode *script, asCTypeInfo **scopeType, bool isRequired)
{
	if( scopeType )
		*scopeType = 0;

	asSNameSpace *ns = implicitNs;
	if( nsName == "::" )
		ns = engine->nameSpaces[0];
	else if( nsName != "" )
	{
		ns = engine->FindNameSpace(nsName.AddressOf());
		if (ns == 0 && scopeType)
		{
			asCString typeName;
			asCString searchNs;

			// Split the scope with at the inner most ::
			int pos = nsName.FindLast("::");
			bool recursive = false;
			if (pos >= 0)
			{
				// Fully qualified namespace
				typeName = nsName.SubString(pos + 2);
				searchNs = nsName.SubString(0, pos);
			}
			else
			{
				// Partially qualified, use the implicit namespace and then search recursively for the type
				typeName = nsName;
				searchNs = implicitNs->name;
				recursive = true;
			}

			asSNameSpace *nsTmp = searchNs == "::" ? engine->nameSpaces[0] : engine->FindNameSpace(searchNs.AddressOf());
			asCTypeInfo *ti = 0;
			while( !ti && nsTmp )
			{
				// Check if the typeName is an existing type in the namespace
				ti = GetType(typeName.AddressOf(), nsTmp, 0);
				if (ti)
				{
					// The informed scope is not a namespace, but it does match a type
					*scopeType = ti;
					return 0;
				}
				nsTmp = recursive ? engine->GetParentNameSpace(nsTmp) : 0;
			}
		}

		if (ns == 0 && isRequired)
		{
			asCString msg;
			msg.Format(TXT_NAMESPACE_s_DOESNT_EXIST, nsName.AddressOf());
			WriteError(msg, script, errNode);
		}
	}

	return ns;
}

asCDataType asCBuilder::CreateDataTypeFromNode(asCScriptNode *node, asCScriptCode *file, asSNameSpace *implicitNamespace, bool acceptHandleForScope, asCObjectType *currentType, bool reportError, bool *isValid, asCArray<asCDataType> *templSubTypes, asCArray<asSNameSpace*>* scopeVisibleNamespaces)
{
	asASSERT(node->nodeType == snDataType || node->nodeType == snIdentifier || node->nodeType == snScope );

	asCDataType dt;

	asCScriptNode *n = node->firstChild;

	if (isValid)
		*isValid = true;

	// If the informed node is an identifier or scope, then the
	// datatype should be identified directly from that
	if (node->nodeType != snDataType)
		n = node;

	bool isConst = false;
	bool isImplicitHandle = false;
	if( n->tokenType == ttConst )
	{
		isConst = true;
		n = n->next;
	}

	// Determine namespace (or parent type) to search for the data type in
	asCObjectType *parentType = 0;
	asSNameSpace *ns = GetNameSpaceFromNode(n, file, implicitNamespace, &n, &parentType);
	if( ns == 0 && parentType == 0 )
	{
		// The namespace and parent type doesn't exist. Return a dummy type instead.
		dt = asCDataType::CreatePrimitive(ttInt, false);
		if (isValid)
			*isValid = false;
		return dt;
	}

	if( n->tokenType == ttIdentifier )
	{
		bool found = false;

		asCString str;
		str.Assign(&file->code[n->tokenPos], n->tokenLength);

		// Recursively search parent namespaces for matching type
		asSNameSpace *origNs = ns;
		asCObjectType *origParentType = parentType;
		asCArray<asSNameSpace*> pendingNamespaces;
		asCArray<asSNameSpace*> visitedNamespaces;
		bool checkAmbiguousSymbols = ns == engine->nameSpaces[0];

		if (scopeVisibleNamespaces)
		{
			pendingNamespaces = *scopeVisibleNamespaces;
		}

		asSNameSpace* parentNs = engine->GetParentNameSpace(ns);
		while( (ns || parentType) && (!found || checkAmbiguousSymbols))
		{
			if (!visitedNamespaces.Exists(ns) || parentType)
			{
				visitedNamespaces.PushLast(ns);
				asCTypeInfo* ti = 0;
				bool canContinue = true;

				if (templSubTypes)
				{
					// If a list of templSubTypes is provided then first check if the identifier matches any of them
					for (asUINT subtypeIndex = 0; subtypeIndex < templSubTypes->GetLength(); subtypeIndex++)
					{
						asCTypeInfo* type = ((*templSubTypes)[subtypeIndex]).GetTypeInfo();
						if (type && str == type->name)
						{
							ti = type;
							break;
						}
					}
				}
				if (ti == 0 && currentType)
				{
					// If this is for a template type, then we must first determine if the
					// identifier matches any of the template subtypes
					if (currentType->flags & asOBJ_TEMPLATE)
					{
						for (asUINT subtypeIndex = 0; subtypeIndex < currentType->templateSubTypes.GetLength(); subtypeIndex++)
						{
							asCTypeInfo* type = currentType->templateSubTypes[subtypeIndex].GetTypeInfo();
							if (type && str == type->name)
							{
								ti = type;
								break;
							}
						}
					}

					if (ti == 0)
					{
						// Check if the type is a child type of the current type
						ti = GetFuncDef(str.AddressOf(), 0, currentType);
						if (ti)
						{
							if (dt.GetTypeInfo() != 0)
							{
								asCString msg;
								msg.Format(TXT_AMBIGUOUS_SYMBOL_NAME_s, str.AddressOf());
								WriteError(msg, file, node);

								// Return a dummy
								return asCDataType::CreatePrimitive(ttInt, isConst);
							}

							dt = asCDataType::CreateType(ti, false);
							found = true;
							canContinue = false;
						}
					}
				}

				if (ti == 0)
					ti = GetType(str.AddressOf(), ns, parentType);
				if (ti == 0 && !module && currentType)
					ti = GetTypeFromTypesKnownByObject(str.AddressOf(), currentType);

				if (ti && (!found || checkAmbiguousSymbols) && canContinue)
				{
					if (dt.GetTypeInfo() != 0)
					{
						asCString msg;
						msg.Format(TXT_AMBIGUOUS_SYMBOL_NAME_s, str.AddressOf());
						WriteError(msg, file, node);

						// Return a dummy
						return asCDataType::CreatePrimitive(ttInt, isConst);
					}

					found = true;

					if (ti->flags & asOBJ_IMPLICIT_HANDLE)
						isImplicitHandle = true;

					// Make sure the module has access to the object type
					if (!module || (module->m_accessMask & ti->accessMask))
					{
						if (asOBJ_TYPEDEF == (ti->flags & asOBJ_TYPEDEF))
						{
							// TODO: typedef: A typedef should be considered different from the original type (though with implicit conversions between the two)
							// Create primitive data type based on object flags
							dt = CastToTypedefType(ti)->aliasForType;
							dt.MakeReadOnly(isConst);
						}
						else
						{
							if (ti->flags & asOBJ_TEMPLATE)
							{
								ti = GetTemplateInstanceFromNode(n, file, CastToObjectType(ti), implicitNamespace, currentType, &n);
								if (ti == 0)
								{
									if (isValid)
										*isValid = false;

									// Return a dummy
									return asCDataType::CreatePrimitive(ttInt, false);
								}
							}
							else if (n && n->next && n->next->nodeType == snDataType)
							{
								if (reportError)
								{
									asCString msg;
									msg.Format(TXT_TYPE_s_NOT_TEMPLATE, ti->name.AddressOf());
									WriteError(msg, file, n);
								}
								if (isValid)
									*isValid = false;

								// Return a dummy
								return asCDataType::CreatePrimitive(ttInt, false);
							}

							// Create object data type
							if (ti)
								dt = asCDataType::CreateType(ti, isConst);
							else
								dt = asCDataType::CreatePrimitive(ttInt, isConst);
						}
					}
					else
					{
						if (reportError)
						{
							asCString msg;
							msg.Format(TXT_TYPE_s_NOT_AVAILABLE_FOR_MODULE, (const char*)str.AddressOf());
							WriteError(msg, file, n);
						}

						dt.SetTokenType(ttInt);
						if (isValid)
							*isValid = false;

						// Return a dummy
						return asCDataType::CreatePrimitive(ttInt, false);
					}
				}
			}

			if( !found || checkAmbiguousSymbols )
			{
				AddVisibleNamespaces(ns, visitedNamespaces, pendingNamespaces);
				
				// Try to find it in the parent namespace
				ns = FindNextVisibleNamespace(visitedNamespaces, pendingNamespaces, parentNs, &checkAmbiguousSymbols);
				if (parentNs == ns)
					parentNs = engine->GetParentNameSpace(ns);
				
				if (parentType)
					parentType = 0;
			}
		}

		if( !found )
		{
			if (reportError)
			{
				asCString msg;
				if (origNs && origNs->name == "")
					msg.Format(TXT_IDENTIFIER_s_NOT_DATA_TYPE_IN_GLOBAL_NS, str.AddressOf());
				else if (origNs)
					msg.Format(TXT_IDENTIFIER_s_NOT_DATA_TYPE_IN_NS_s, str.AddressOf(), origNs->name.AddressOf());
				else
				{
					// TODO: child funcdef: Message should explain that the identifier is not a type of the parent type
					asCDataType pt = asCDataType::CreateType(origParentType, false);
					msg.Format(TXT_IDENTIFIER_s_NOT_DATA_TYPE_IN_NS_s, str.AddressOf(), pt.Format(origParentType->nameSpace, false).AddressOf());
				}
				WriteError(msg, file, n);
			}

			dt = asCDataType::CreatePrimitive(ttInt, isConst);
			if (isValid)
				*isValid = false;
			return dt;
		}
	}
	else if( n->tokenType == ttAuto )
	{
		dt = asCDataType::CreateAuto(isConst);
	}
	else
	{
		// Create primitive data type
		dt = asCDataType::CreatePrimitive(n->tokenType, isConst);
	}

	// Determine array dimensions and object handles
	n = n->next;
	while( n && (n->tokenType == ttOpenBracket || n->tokenType == ttHandle) )
	{
		if( n->tokenType == ttOpenBracket )
		{
			if (isImplicitHandle)
			{
				// Make the type a handle
				if (dt.MakeHandle(true, acceptHandleForScope) < 0)
				{
					if (reportError)
						WriteError(TXT_OBJECT_HANDLE_NOT_SUPPORTED, file, n);
					if (isValid)
						*isValid = false;

					// Return a dummy
					return asCDataType::CreatePrimitive(ttInt, false);
				}
				isImplicitHandle = false;
			}

			// Make sure the sub type can be instantiated
			if( !dt.CanBeInstantiated() || dt.IsAuto() )
			{
				if (reportError)
				{
					asCString str;
					if (dt.IsAbstractClass())
						str.Format(TXT_ABSTRACT_CLASS_s_CANNOT_BE_INSTANTIATED, dt.Format(ns).AddressOf());
					else if (dt.IsInterface())
						str.Format(TXT_INTERFACE_s_CANNOT_BE_INSTANTIATED, dt.Format(ns).AddressOf());
					else
						// TODO: Improve error message to explain why
						str.Format(TXT_DATA_TYPE_CANT_BE_s, dt.Format(ns).AddressOf());

					WriteError(str, file, n);
				}
				if (isValid)
					*isValid = false;

				// Return a dummy
				return asCDataType::CreatePrimitive(ttInt, false);
			}

			// Make the type an array (or multidimensional array)
			// TODO: If reportError is false, the errors in MakeArray must be suppressed
			asCDataType origDt = dt;
			int r = dt.MakeArray(engine, module);
			if( r < 0 )
			{
				if (reportError && r == asINVALID_TYPE)
					WriteError(TXT_NO_DEFAULT_ARRAY_TYPE, file, n);
				else
				{
					asCString msg;
					msg.Format(TXT_CANNOT_FORM_ARRAY_OF_s, origDt.Format(ns).AddressOf());
					WriteError(msg, file, n);
				}
				if (isValid)
					*isValid = false;

				// Return a dummy
				return asCDataType::CreatePrimitive(ttInt, false);
			}
		}
		else
		{
			// Make the type a handle
			if( dt.IsObjectHandle() )
			{
				if( reportError )
					WriteError(TXT_HANDLE_OF_HANDLE_IS_NOT_ALLOWED, file, n);
				if (isValid)
					*isValid = false;

				// Return a dummy
				return asCDataType::CreatePrimitive(ttInt, false);
			}
			else
			{
				if( dt.MakeHandle(true, acceptHandleForScope) < 0 )
				{
					if( reportError )
						WriteError(TXT_OBJECT_HANDLE_NOT_SUPPORTED, file, n);
					if (isValid)
						*isValid = false;

					// Return a dummy
					return asCDataType::CreatePrimitive(ttInt, false);
				}
				
				// Check if the handle should be read-only
				if( n && n->next && n->next->tokenType == ttConst )
					dt.MakeReadOnly(true);
			}
		}
		n = n->next;
	}

	if( isImplicitHandle )
	{
		// Make the type a handle
		if (dt.MakeHandle(true, acceptHandleForScope) < 0)
		{
			if( reportError )
				WriteError(TXT_OBJECT_HANDLE_NOT_SUPPORTED, file, n);
			if (isValid)
				*isValid = false;

			// Return a dummy
			return asCDataType::CreatePrimitive(ttInt, false);
		}
	}

	return dt;
}

asCObjectType *asCBuilder::GetTemplateInstanceFromNode(asCScriptNode *node, asCScriptCode *file, asCObjectType *templateType, asSNameSpace *implicitNamespace, asCObjectType *currentType, asCScriptNode **next)
{
	// Check if the subtype is a type or the template's subtype
	// if it is the template's subtype then this is the actual template type,
	// orderwise it is a template instance.
	// Only do this for application registered interface, as the
	// scripts cannot implement templates.
	asCArray<asCDataType> subTypes;
	asUINT subtypeIndex;
	asCScriptNode *n = node;
	while (n && n->next && n->next->nodeType == snDataType)
	{
		n = n->next;

		// When parsing function definitions for template registrations (currentType != 0) it is necessary
		// to pass in the current template type to the recursive call since it is this ones sub-template types
		// that should be allowed.
		asCDataType subType = CreateDataTypeFromNode(n, file, implicitNamespace, false, module ? 0 : (currentType ? currentType : templateType));
		subTypes.PushLast(subType);
	}

	if (next)
		*next = n;

	if (subTypes.GetLength() != templateType->templateSubTypes.GetLength())
	{
		asCString msg;
		msg.Format(TXT_TMPL_s_EXPECTS_d_SUBTYPES, templateType->name.AddressOf(), int(templateType->templateSubTypes.GetLength()));
		WriteError(msg, file, node);

		// Return a dummy
		return 0;
	}

	// Check if any of the given subtypes are different from the template's declared subtypes
	bool isDifferent = false;
	for (subtypeIndex = 0; subtypeIndex < subTypes.GetLength(); subtypeIndex++)
	{
		if (subTypes[subtypeIndex].GetTypeInfo() != templateType->templateSubTypes[subtypeIndex].GetTypeInfo())
		{
			isDifferent = true;
			break;
		}
	}

	if (isDifferent)
	{
		// This is a template instance
		// Need to find the correct object type
		asCObjectType *otInstance = engine->GetTemplateInstanceType(templateType, subTypes, module);

		if (otInstance && otInstance->scriptSectionIdx < 0)
		{
			// If this is the first time the template instance is used, store where it was declared from
			otInstance->scriptSectionIdx = engine->GetScriptSectionNameIndex(file->name.AddressOf());
			int row, column;
			file->ConvertPosToRowCol(n->tokenPos, &row, &column);
			otInstance->declaredAt = (row & 0xFFFFF) | (column << 20);
		}

		if (!otInstance)
		{
			asCString sub = subTypes[0].Format(templateType->nameSpace);
			for (asUINT s = 1; s < subTypes.GetLength(); s++)
			{
				sub += ",";
				sub += subTypes[s].Format(templateType->nameSpace);
			}
			asCString msg;
			msg.Format(TXT_INSTANCING_INVLD_TMPL_TYPE_s_s, templateType->name.AddressOf(), sub.AddressOf());
			WriteError(msg, file, n);
		}

		return otInstance;
	}

	return templateType;
}

asCDataType asCBuilder::ModifyDataTypeFromNode(const asCDataType &type, asCScriptNode *node, asCScriptCode *file, asETypeModifiers *inOutFlags, bool *autoHandle)
{
	asCDataType dt = type;

	if( inOutFlags ) *inOutFlags = asTM_NONE;

	// Is the argument sent by reference?
	asCScriptNode *n = node->firstChild;
	if( n && n->tokenType == ttAmp )
	{
		if (dt.GetTokenType() == ttVoid)
		{
			asCString msg;
			msg.Format(TXT_TYPE_s_CANNOT_BE_REFERENCE, type.Format(0).AddressOf());
			WriteError(msg, file, node->firstChild);
			return dt;
		}

		dt.MakeReference(true);
		n = n->next;

		if( n )
		{
			if( inOutFlags )
			{
				if( n->tokenType == ttIn )
					*inOutFlags = asTM_INREF;
				else if( n->tokenType == ttOut )
					*inOutFlags = asTM_OUTREF;
				else if( n->tokenType == ttInOut )
					*inOutFlags = asTM_INOUTREF;
				else
					asASSERT(false);
			}

			n = n->next;
		}
		else
		{
			if( inOutFlags )
				*inOutFlags = asTM_INOUTREF; // ttInOut
		}

		if( !engine->ep.allowUnsafeReferences &&
			inOutFlags && *inOutFlags == asTM_INOUTREF &&
			!(dt.GetTypeInfo() && (dt.GetTypeInfo()->flags & asOBJ_TEMPLATE_SUBTYPE)) )
		{
			// Verify that the base type support &inout parameter types
			if( !dt.IsObject() || dt.IsObjectHandle() || 
				!((dt.GetTypeInfo()->flags & asOBJ_NOCOUNT) || (CastToObjectType(dt.GetTypeInfo())->beh.addref && CastToObjectType(dt.GetTypeInfo())->beh.release)) )
				WriteError(TXT_ONLY_OBJECTS_MAY_USE_REF_INOUT, file, node->firstChild);
		}
	}

	if( autoHandle ) *autoHandle = false;

	if( n && n->tokenType == ttPlus )
	{
		// Autohandles are not supported for types with NOCOUNT
		// If the type is not a handle then there was an error with building the type, but
		// this error would already have been reported so no need to report another error here
		if( dt.IsObjectHandle() && (dt.GetTypeInfo()->flags & asOBJ_NOCOUNT) )
			WriteError(TXT_AUTOHANDLE_CANNOT_BE_USED_FOR_NOCOUNT, file, node->firstChild);

		if( autoHandle ) *autoHandle = true;
	}

	if (n && n->tokenType == ttIdentifier)
	{
		asCString str;
		str.Assign(&file->code[n->tokenPos], n->tokenLength);
		if (str == IF_HANDLE_TOKEN)
			dt.SetIfHandleThenConst(true);
		else
		{
			// TODO: Should give error if not currently parsing template registration
			asCString msg;
			msg.Format(TXT_UNEXPECTED_TOKEN_s, str.AddressOf());
			WriteError(msg, file, node->firstChild);
		}
	}

	return dt;
}

asCTypeInfo *asCBuilder::GetType(const char *type, asSNameSpace *ns, asCObjectType *parentType)
{
	asASSERT((ns == 0 && parentType) || (ns && parentType == 0));

	if (ns)
	{
		asCTypeInfo *ti = engine->GetRegisteredType(type, ns);
		if (!ti && module)
			ti = module->GetType(type, ns);
		return ti;
	}
	else
	{
		// Recursively check base classes
		asCObjectType *currType = parentType;
		while (currType)
		{
			for (asUINT n = 0; n < currType->childFuncDefs.GetLength(); n++)
			{
				asCFuncdefType *funcDef = currType->childFuncDefs[n];
				if (funcDef && funcDef->name == type)
					return funcDef;
			}
			currType = currType->derivedFrom;
		}
	}

	return 0;
}

asCObjectType *asCBuilder::GetObjectType(const char *type, asSNameSpace *ns)
{
	return CastToObjectType(GetType(type, ns, 0));
}

#ifndef AS_NO_COMPILER
// This function will return true if there are any types in the engine or module
// with the given name. The namespace is ignored in this verification.
bool asCBuilder::DoesTypeExist(const asCString &type)
{
	asUINT n;

	// This function is only used when parsing expressions for building bytecode
	// and this is only done after all types are known. For this reason the types
	// can be safely cached in a map for quick lookup. Once the builder is released
	// the cache will also be destroyed thus avoiding unnecessary memory consumption.
	if( !hasCachedKnownTypes )
	{
		// Only do this once
		hasCachedKnownTypes = true;

		// Add registered types
		asSMapNode<asSNameSpaceNamePair, asCTypeInfo*> *cursor;
		engine->allRegisteredTypes.MoveFirst(&cursor);
		while( cursor )
		{
			if( !knownTypes.MoveTo(0, cursor->key.name) )
				knownTypes.Insert(cursor->key.name, true);

			engine->allRegisteredTypes.MoveNext(&cursor, cursor);
		}

		if (module)
		{
			// Add script classes and interfaces
			for (n = 0; n < module->m_classTypes.GetLength(); n++)
				if (!knownTypes.MoveTo(0, module->m_classTypes[n]->name))
					knownTypes.Insert(module->m_classTypes[n]->name, true);

			// Add script enums
			for (n = 0; n < module->m_enumTypes.GetLength(); n++)
				if (!knownTypes.MoveTo(0, module->m_enumTypes[n]->name))
					knownTypes.Insert(module->m_enumTypes[n]->name, true);

			// Add script typedefs
			for (n = 0; n < module->m_typeDefs.GetLength(); n++)
				if (!knownTypes.MoveTo(0, module->m_typeDefs[n]->name))
					knownTypes.Insert(module->m_typeDefs[n]->name, true);

			// Add script funcdefs
			for (n = 0; n < module->m_funcDefs.GetLength(); n++)
				if (!knownTypes.MoveTo(0, module->m_funcDefs[n]->name))
					knownTypes.Insert(module->m_funcDefs[n]->name, true);
		}
	}

	// Check if the type is known
	return knownTypes.MoveTo(0, type);
}
#endif

asCTypeInfo *asCBuilder::GetTypeFromTypesKnownByObject(const char *type, asCObjectType *currentType)
{
	if (currentType->name == type)
		return currentType;

	asUINT n;

	asCTypeInfo *found = 0;

	for (n = 0; found == 0 && n < currentType->properties.GetLength(); n++)
		if (currentType->properties[n]->type.GetTypeInfo() &&
			currentType->properties[n]->type.GetTypeInfo()->name == type)
			found = currentType->properties[n]->type.GetTypeInfo();

	for (n = 0; found == 0 && n < currentType->methods.GetLength(); n++)
	{
		asCScriptFunction *func = engine->scriptFunctions[currentType->methods[n]];
		if (func->returnType.GetTypeInfo() &&
			func->returnType.GetTypeInfo()->name == type)
			found = func->returnType.GetTypeInfo();

		for (asUINT f = 0; found == 0 && f < func->parameterTypes.GetLength(); f++)
			if (func->parameterTypes[f].GetTypeInfo() &&
				func->parameterTypes[f].GetTypeInfo()->name == type)
				found = func->parameterTypes[f].GetTypeInfo();
	}

	if (found)
	{
		// In case we find a template instance it mustn't be returned
		// because it is not known if the subtype is really matching
		if (found->flags & asOBJ_TEMPLATE)
			return 0;
	}

	return found;
}

asCFuncdefType *asCBuilder::GetFuncDef(const char *type, asSNameSpace *ns, asCObjectType *parentType)
{
	asASSERT((ns == 0 && parentType) || (ns && parentType == 0));

	if (ns)
	{
		for (asUINT n = 0; n < engine->registeredFuncDefs.GetLength(); n++)
		{
			asCFuncdefType *funcDef = engine->registeredFuncDefs[n];
			// TODO: access: Only return the definitions that the module has access to
			if (funcDef && funcDef->nameSpace == ns && funcDef->name == type)
				return funcDef;
		}

		if (module)
		{
			for (asUINT n = 0; n < module->m_funcDefs.GetLength(); n++)
			{
				asCFuncdefType *funcDef = module->m_funcDefs[n];
				if (funcDef && funcDef->nameSpace == ns && funcDef->name == type)
					return funcDef;
			}
		}
	}
	else
	{
		// Recursively check base classes
		asCObjectType *currType = parentType;
		while (currType)
		{
			for (asUINT n = 0; n < currType->childFuncDefs.GetLength(); n++)
			{
				asCFuncdefType *funcDef = currType->childFuncDefs[n];
				if (funcDef && funcDef->name == type)
					return funcDef;
			}
			currType = currType->derivedFrom;
		}
	}

	return 0;
}

#ifndef AS_NO_COMPILER

int asCBuilder::GetEnumValueFromType(asCEnumType *type, const char *name, asCDataType &outDt, asDWORD &outValue)
{
	if( !type || !(type->flags & asOBJ_ENUM) )
		return 0;

	for( asUINT n = 0; n < type->enumValues.GetLength(); ++n )
	{
		if( type->enumValues[n]->name == name )
		{
			outDt = asCDataType::CreateType(type, true);
			outValue = type->enumValues[n]->value;
			return 1;
		}
	}

	return 0;
}

int asCBuilder::GetEnumValue(const char *name, asCDataType &outDt, asDWORD &outValue, asSNameSpace *ns)
{
	bool found = false;

	// Search all available enum types
	asUINT t;
	for( t = 0; t < engine->registeredEnums.GetLength(); t++ )
	{
		asCEnumType *et = engine->registeredEnums[t];
		if( ns != et->nameSpace ) continue;

		// Don't bother with types the module doesn't have access to
		if( (et->accessMask & module->m_accessMask) == 0 )
			continue;

		if( GetEnumValueFromType(et, name, outDt, outValue) )
		{
			if( !found )
				found = true;
			else
			{
				// Found more than one value in different enum types
				return 2;
			}
		}
	}

	for( t = 0; t < module->m_enumTypes.GetLength(); t++ )
	{
		asCEnumType *et = module->m_enumTypes[t];
		if( ns != et->nameSpace ) continue;

		if( GetEnumValueFromType(et, name, outDt, outValue) )
		{
			if( !found )
				found = true;
			else
			{
				// Found more than one value in different enum types
				return 2;
			}
		}
	}

	if( found )
		return 1;

	// Didn't find any value
	return 0;
}

#endif // AS_NO_COMPILER

END_AS_NAMESPACE
