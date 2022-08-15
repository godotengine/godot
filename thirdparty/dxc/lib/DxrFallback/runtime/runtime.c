#include <stddef.h>

static const int STACK_SIZE_IN_BYTES = 1024;

typedef float float3 __attribute__((vector_size(3*sizeof(float))));
typedef float float4 __attribute__((vector_size(4*sizeof(float))));
typedef float float12 __attribute__((vector_size(12*sizeof(float))));
typedef float (M3x4)[12];
typedef int   (StackType)[STACK_SIZE_IN_BYTES/sizeof(int)];
typedef unsigned char byte;


typedef struct RuntimeDataStruct
{
  int DispatchRaysIndex[2];
  int DispatchRaysDimensions[2];

  float RayTMin;
  float RayTCurrent;
  unsigned RayFlags;
  float WorldRayOrigin[3];
  float WorldRayDirection[3];
  float ObjectRayOrigin[3];
  float ObjectRayDirection[3];
  M3x4 ObjectToWorld;
  M3x4 WorldToObject;

  unsigned PrimitiveIndex;
  unsigned InstanceIndex;
  unsigned InstanceID;
  unsigned HitKind;
  unsigned ShaderRecordOffset;


  // Pending hit values - accessed in anyHit and intersection shaders before a hit has been committed
  float PendingRayTCurrent;
  unsigned PendingPrimitiveIndex;
  unsigned PendingInstanceIndex;
  unsigned PendingInstanceID;
  unsigned PendingHitKind;
  unsigned PendingShaderRecordOffset; 

  int GroupIndex; 
  int AnyHitResult;
  int AnyHitStateId;  // Originally temporary. We needed to avoid resource usage
                      // in ReportHit() because of linking issues so weset the value here first. 
                      // May be worth retaining to cache the value when fetching the intersection 
                      // stateId (fetch them both at once). 

  int PayloadOffset;            
  int CommittedAttrOffset;      
  int PendingAttrOffset;        
  
  int StackOffset; // offset from the start of the stack
  StackType* Stack;
} RuntimeData;

typedef RuntimeData* RuntimeDataType;

typedef struct TraceRaySpills_ClosestHit
{
  float RayTMin;                 
  float RayTCurrent;             
  unsigned RayFlags;             
  float WorldRayOrigin[3];       
  float WorldRayDirection[3];    
  float ObjectRayOrigin[3];      
  float ObjectRayDirection[3];   

  unsigned PrimitiveIndex;       
  unsigned InstanceIndex;        
  unsigned InstanceID;           
  unsigned HitKind;              
  unsigned ShaderRecordOffset;
} TraceRaySpills_ClosestHit;

typedef struct TraceRaySpills_Miss
{
  float RayTMin;                 
  float RayTCurrent;             
  unsigned RayFlags;             
  float WorldRayOrigin[3];       
  float WorldRayDirection[3];    
            
  unsigned ShaderRecordOffset;
} TraceRaySpills_Miss;


#define REF(x) (runtimeData->x)
#define REF_FLT(x) (runtimeData->x)
#define REF_STACK(offset) ((*runtimeData->Stack)[runtimeData->StackOffset + offset])
#define REF_FLT_OFS(x, offset) (runtimeData->x[offset])

// Return next stateID
int rewrite_dispatch(RuntimeDataType runtimeData, int stateID);
void* rewrite_setLaunchParams(RuntimeDataType runtimeData, unsigned dimx, unsigned dimy);
unsigned rewrite_getStackSize(void);
StackType* rewrite_createStack(void);

void stackInit(RuntimeDataType runtimeData, StackType* theStack, unsigned stackSize)
{
  REF(Stack) = theStack;
  REF(StackOffset) = stackSize/sizeof(int) - 1;
  REF(PayloadOffset)       = 1111; // recognizable bogus values
  REF(CommittedAttrOffset) = 2222;
  REF(PendingAttrOffset)   = 3333;
}

void stackFramePush(RuntimeDataType runtimeData, int size)
{
  REF(StackOffset) -= size;
}

void stackFramePop(RuntimeDataType runtimeData, int size)
{ 
  REF(StackOffset) += size;
}

int stackFrameOffset(RuntimeDataType runtimeData)
{
  return REF(StackOffset);
}

int payloadOffset(RuntimeDataType runtimeData)
{
  return REF(PayloadOffset);
}

int committedAttrOffset(RuntimeDataType runtimeData)
{
  return REF(CommittedAttrOffset);
}

int pendingAttrOffset(RuntimeDataType runtimeData)
{
  return REF(PendingAttrOffset);
}

int* stackIntPtr(RuntimeDataType runtimeData, int baseOffset, int offset)
{
  return &(*runtimeData->Stack)[baseOffset + offset];
}


void traceFramePush(RuntimeDataType runtimeData, int attrSize)
{
  // Save the old payload and attribute offsets
  REF_STACK(-1) = REF(CommittedAttrOffset);
  REF_STACK(-2) = REF(PendingAttrOffset);

  // Set new offsets
  REF(CommittedAttrOffset) = REF(StackOffset) - 2 - attrSize; 
  REF(PendingAttrOffset)   = REF(StackOffset) - 2 - 2 * attrSize; 
}

void traceFramePop(RuntimeDataType runtimeData)
{
  // Restore the old attribute offsets
  REF(CommittedAttrOffset) = REF_STACK(-1); 
  REF(PendingAttrOffset) = REF_STACK(-2);
}

void traceRaySave_ClosestHit(RuntimeDataType runtimeData, TraceRaySpills_ClosestHit* spills)
{
  spills->RayFlags              = REF(RayFlags);
  spills->RayTCurrent           = REF_FLT(RayTCurrent);
  spills->RayTMin               = REF_FLT(RayTMin);
  spills->WorldRayOrigin[0]     = REF_FLT(WorldRayOrigin[0]);
  spills->WorldRayOrigin[1]     = REF_FLT(WorldRayOrigin[1]);
  spills->WorldRayOrigin[2]     = REF_FLT(WorldRayOrigin[2]);
  spills->WorldRayDirection[0]  = REF_FLT(WorldRayDirection[0]);
  spills->WorldRayDirection[1]  = REF_FLT(WorldRayDirection[1]);
  spills->WorldRayDirection[2]  = REF_FLT(WorldRayDirection[2]);
  spills->ObjectRayOrigin[0]    = REF_FLT(ObjectRayOrigin[0]);
  spills->ObjectRayOrigin[1]    = REF_FLT(ObjectRayOrigin[1]);
  spills->ObjectRayOrigin[2]    = REF_FLT(ObjectRayOrigin[2]);
  spills->ObjectRayDirection[0] = REF_FLT(ObjectRayDirection[0]);
  spills->ObjectRayDirection[1] = REF_FLT(ObjectRayDirection[1]);
  spills->ObjectRayDirection[2] = REF_FLT(ObjectRayDirection[2]);

  spills->PrimitiveIndex      = REF(PrimitiveIndex);       
  spills->InstanceIndex       = REF(InstanceIndex);        
  spills->InstanceID          = REF(InstanceID);           
  spills->HitKind             = REF(HitKind);              
  spills->ShaderRecordOffset  = REF(ShaderRecordOffset);  
}

void traceRayRestore_ClosestHit(RuntimeDataType runtimeData, TraceRaySpills_ClosestHit* spills)
{
  REF(RayFlags)                  = spills->RayFlags;               
  REF_FLT(RayTCurrent)           = spills->RayTCurrent;            
  REF_FLT(RayTMin)               = spills->RayTMin;                
  REF_FLT(WorldRayOrigin[0])     = spills->WorldRayOrigin[0];      
  REF_FLT(WorldRayOrigin[1])     = spills->WorldRayOrigin[1];      
  REF_FLT(WorldRayOrigin[2])     = spills->WorldRayOrigin[2];      
  REF_FLT(WorldRayDirection[0])  = spills->WorldRayDirection[0];   
  REF_FLT(WorldRayDirection[1])  = spills->WorldRayDirection[1];   
  REF_FLT(WorldRayDirection[2])  = spills->WorldRayDirection[2];   
  REF_FLT(ObjectRayOrigin[0])    = spills->ObjectRayOrigin[0];     
  REF_FLT(ObjectRayOrigin[1])    = spills->ObjectRayOrigin[1];     
  REF_FLT(ObjectRayOrigin[2])    = spills->ObjectRayOrigin[2];     
  REF_FLT(ObjectRayDirection[0]) = spills->ObjectRayDirection[0];  
  REF_FLT(ObjectRayDirection[1]) = spills->ObjectRayDirection[1];  
  REF_FLT(ObjectRayDirection[2]) = spills->ObjectRayDirection[2];  

  REF(PrimitiveIndex)     = spills->PrimitiveIndex;          
  REF(InstanceIndex)      = spills->InstanceIndex;           
  REF(InstanceID)         = spills->InstanceID;              
  REF(HitKind)            = spills->HitKind;                 
  REF(ShaderRecordOffset) = spills->ShaderRecordOffset;    
}

void traceRaySave_Miss(RuntimeDataType runtimeData, TraceRaySpills_Miss* spills)
{
  spills->RayFlags              = REF(RayFlags);
  spills->RayTCurrent           = REF_FLT(RayTCurrent);
  spills->RayTMin               = REF_FLT(RayTMin);
  spills->WorldRayOrigin[0]     = REF_FLT(WorldRayOrigin[0]);
  spills->WorldRayOrigin[1]     = REF_FLT(WorldRayOrigin[1]);
  spills->WorldRayOrigin[2]     = REF_FLT(WorldRayOrigin[2]);
  spills->WorldRayDirection[0]  = REF_FLT(WorldRayDirection[0]);
  spills->WorldRayDirection[1]  = REF_FLT(WorldRayDirection[1]);
  spills->WorldRayDirection[2]  = REF_FLT(WorldRayDirection[2]);

  spills->ShaderRecordOffset    = REF(ShaderRecordOffset);  
}

void traceRayRestore_Miss(RuntimeDataType runtimeData, TraceRaySpills_Miss* spills)
{
  REF(RayFlags)                  = spills->RayFlags;               
  REF_FLT(RayTCurrent)           = spills->RayTCurrent;            
  REF_FLT(RayTMin)               = spills->RayTMin;                
  REF_FLT(WorldRayOrigin[0])     = spills->WorldRayOrigin[0];      
  REF_FLT(WorldRayOrigin[1])     = spills->WorldRayOrigin[1];      
  REF_FLT(WorldRayOrigin[2])     = spills->WorldRayOrigin[2];      
  REF_FLT(WorldRayDirection[0])  = spills->WorldRayDirection[0];   
  REF_FLT(WorldRayDirection[1])  = spills->WorldRayDirection[1];   
  REF_FLT(WorldRayDirection[2])  = spills->WorldRayDirection[2];   

  REF(ShaderRecordOffset) = spills->ShaderRecordOffset;    
}





//////////////////////////////////////////////////////////////////////////
//
// Intrinsics for the fallback layer
//
//////////////////////////////////////////////////////////////////////////

void fb_Fallback_Scheduler(int initialStateId, unsigned dimx, unsigned dimy)
{
  StackType* theStack = rewrite_createStack();
  RuntimeData theRuntimeData;
  RuntimeDataType runtimeData = &theRuntimeData;

  rewrite_setLaunchParams(runtimeData, dimx, dimy);
  if(REF(DispatchRaysIndex[0]) >= REF(DispatchRaysDimensions[0]) ||
     REF(DispatchRaysIndex[1]) >= REF(DispatchRaysDimensions[1]))
  { 
    return;
  }


  // Set final return stateID into reserved area at stack top
  unsigned stackSize = rewrite_getStackSize();
  stackInit(runtimeData, theStack, stackSize);
  int stackFrameOffs = stackFrameOffset(runtimeData);
  *stackIntPtr(runtimeData, stackFrameOffs, 0) = -1;

  int stateId = initialStateId;
  int count = 0;
  while( stateId >= 0 )
  {
    stateId = rewrite_dispatch(runtimeData, stateId);
  }
}

void fb_Fallback_SetLaunchParams(RuntimeDataType runtimeData, unsigned DTidx, unsigned DTidy, unsigned dimx, unsigned dimy, unsigned groupIndex)
{ 
  REF(DispatchRaysIndex[0]) = DTidx;
  REF(DispatchRaysIndex[1]) = DTidy;
  REF(DispatchRaysDimensions[0]) = dimx;
  REF(DispatchRaysDimensions[1]) = dimy;

  REF(GroupIndex) = groupIndex;
}

int fb_Fallback_TraceRayBegin(RuntimeDataType runtimeData, unsigned rayFlags, float ox, float oy, float oz, float tmin, float dx, float dy, float dz, float tmax, int newPayloadOffset)
{ 
  REF(RayFlags) = rayFlags;
  REF_FLT(WorldRayOrigin[0]) = ox;
  REF_FLT(WorldRayOrigin[1]) = oy;
  REF_FLT(WorldRayOrigin[2]) = oz;
  REF_FLT(WorldRayDirection[0]) = dx;
  REF_FLT(WorldRayDirection[1]) = dy;
  REF_FLT(WorldRayDirection[2]) = dz;
  REF_FLT(RayTCurrent) = tmax;
  REF_FLT(RayTMin) = tmin;

  int oldOffset = REF(PayloadOffset);
  REF(PayloadOffset) = newPayloadOffset;
  return oldOffset;
}

void fb_Fallback_TraceRayEnd(RuntimeDataType runtimeData, int oldPayloadOffset)
{
  REF(PayloadOffset) = oldPayloadOffset;
}

void fb_Fallback_SetPendingTriVals(RuntimeDataType runtimeData, unsigned shaderRecordOffset, unsigned primitiveIndex, unsigned instanceIndex, unsigned instanceID, float t, unsigned hitKind)
{
  REF(PendingShaderRecordOffset) = shaderRecordOffset;
  REF(PendingPrimitiveIndex) = primitiveIndex;
  REF(PendingInstanceIndex) = instanceIndex;
  REF(PendingInstanceID) = instanceID;
  REF_FLT(PendingRayTCurrent) = t;
  REF(PendingHitKind) = hitKind;
}

void fb_Fallback_SetPendingCustomVals(RuntimeDataType runtimeData, unsigned shaderRecordOffset, unsigned primitiveIndex, unsigned instanceIndex, unsigned instanceID)
{
  REF(PendingShaderRecordOffset) = shaderRecordOffset;
  REF(PendingPrimitiveIndex) = primitiveIndex;
  REF(PendingInstanceIndex) = instanceIndex;
  REF(PendingInstanceID) = instanceID;
}

void fb_Fallback_CommitHit(RuntimeDataType runtimeData)
{
  REF_FLT(RayTCurrent)    = REF_FLT(PendingRayTCurrent);
  REF(ShaderRecordOffset) = REF(PendingShaderRecordOffset);
  REF(PrimitiveIndex)     = REF(PendingPrimitiveIndex);
  REF(InstanceIndex)      = REF(PendingInstanceIndex);
  REF(InstanceID)         = REF(PendingInstanceID);
  REF(HitKind)            = REF(PendingHitKind);  

  int PendingAttrOffset = REF(PendingAttrOffset);
  REF(PendingAttrOffset) = REF(CommittedAttrOffset);
  REF(CommittedAttrOffset) = PendingAttrOffset;
}


int fb_Fallback_RuntimeDataLoadInt(RuntimeDataType runtimeData, int offset)
{
  return (*runtimeData->Stack)[offset];
}

void fb_Fallback_RuntimeDataStoreInt(RuntimeDataType runtimeData, int offset, int val)
{
  (*runtimeData->Stack)[offset] = val;
}

unsigned fb_dxop_dispatchRaysIndex(RuntimeDataType runtimeData, byte i)
{  
  return REF(DispatchRaysIndex[i]);
}

unsigned fb_dxop_dispatchRaysDimensions(RuntimeDataType runtimeData, byte i)
{  
  return REF(DispatchRaysDimensions[i]);
}

float fb_dxop_rayTMin(RuntimeDataType runtimeData)
{
  return REF_FLT(RayTMin);
}

float fb_Fallback_RayTMin(RuntimeDataType runtimeData)
{
  return REF_FLT(RayTMin);
}

void fb_Fallback_SetRayTMin(RuntimeDataType runtimeData, float t)
{
  REF_FLT(RayTMin) = t;
}

float fb_dxop_rayTCurrent(RuntimeDataType runtimeData)
{
  return REF_FLT(RayTCurrent);
}

float fb_Fallback_RayTCurrent(RuntimeDataType runtimeData)
{
  return REF_FLT(RayTCurrent);
}

void fb_Fallback_SetRayTCurrent(RuntimeDataType runtimeData, float t)
{
  REF_FLT(RayTCurrent) = t;
}

unsigned fb_dxop_rayFlags(RuntimeDataType runtimeData)
{
  return REF(RayFlags);
}

unsigned fb_Fallback_RayFlags(RuntimeDataType runtimeData)
{
  return REF(RayFlags);
}

void fb_Fallback_SetRayFlags(RuntimeDataType runtimeData, unsigned flags)
{
  REF(RayFlags) = flags;
}

float fb_dxop_worldRayOrigin(RuntimeDataType runtimeData, byte i)
{ 
  return REF_FLT(WorldRayOrigin[i]);
}

float fb_Fallback_WorldRayOrigin(RuntimeDataType runtimeData, byte i)
{ 
  return REF_FLT(WorldRayOrigin[i]);
}

void fb_Fallback_SetWorldRayOrigin(RuntimeDataType runtimeData, float x, float y, float z)
{ 
  REF_FLT(WorldRayOrigin[0]) = x;
  REF_FLT(WorldRayOrigin[1]) = y;
  REF_FLT(WorldRayOrigin[2]) = z;
}

float fb_dxop_worldRayDirection(RuntimeDataType runtimeData, byte i)
{  
  return REF_FLT(WorldRayDirection[i]);
}

float fb_Fallback_WorldRayDirection(RuntimeDataType runtimeData, byte i)
{  
  return REF_FLT(WorldRayDirection[i]);
}

void fb_Fallback_SetWorldRayDirection(RuntimeDataType runtimeData, float x, float y, float z)
{ 
  REF_FLT(WorldRayDirection[0]) = x;
  REF_FLT(WorldRayDirection[1]) = y;
  REF_FLT(WorldRayDirection[2]) = z;
}

float fb_dxop_objectRayOrigin(RuntimeDataType runtimeData, byte i)
{ 
  return REF_FLT(ObjectRayOrigin[i]);
}

float fb_Fallback_ObjectRayOrigin(RuntimeDataType runtimeData, byte i)
{ 
  return REF_FLT(ObjectRayOrigin[i]);
}

void fb_Fallback_SetObjectRayOrigin(RuntimeDataType runtimeData, float x, float y, float z)
{ 
  REF_FLT(ObjectRayOrigin[0]) = x;
  REF_FLT(ObjectRayOrigin[1]) = y;
  REF_FLT(ObjectRayOrigin[2]) = z;
}

float fb_dxop_objectRayDirection(RuntimeDataType runtimeData, byte i)
{  
  return REF_FLT(ObjectRayDirection[i]);
}

float fb_Fallback_ObjectRayDirection(RuntimeDataType runtimeData, byte i)
{  
  return REF_FLT(ObjectRayDirection[i]);
}

void fb_Fallback_SetObjectRayDirection(RuntimeDataType runtimeData, float x, float y, float z)
{ 
  REF_FLT(ObjectRayDirection[0]) = x;
  REF_FLT(ObjectRayDirection[1]) = y;
  REF_FLT(ObjectRayDirection[2]) = z;
}

float fb_dxop_objectToWorld(RuntimeDataType runtimeData, int r, byte c)
{
  int i = r * 4 + c;
  return REF_FLT_OFS(ObjectToWorld, i);
}

void fb_Fallback_SetObjectToWorld(RuntimeDataType runtimeData, float12 M)
{
  REF_FLT_OFS(ObjectToWorld, 0)  = M[0]; 
  REF_FLT_OFS(ObjectToWorld, 1)  = M[1]; 
  REF_FLT_OFS(ObjectToWorld, 2)  = M[2]; 
  REF_FLT_OFS(ObjectToWorld, 3)  = M[3]; 
  REF_FLT_OFS(ObjectToWorld, 4)  = M[4]; 
  REF_FLT_OFS(ObjectToWorld, 5)  = M[5]; 
  REF_FLT_OFS(ObjectToWorld, 6)  = M[6]; 
  REF_FLT_OFS(ObjectToWorld, 7)  = M[7]; 
  REF_FLT_OFS(ObjectToWorld, 8)  = M[8]; 
  REF_FLT_OFS(ObjectToWorld, 9)  = M[9]; 
  REF_FLT_OFS(ObjectToWorld, 10) = M[10];
  REF_FLT_OFS(ObjectToWorld, 11) = M[11];
}

float fb_dxop_worldToObject(RuntimeDataType runtimeData, int r, byte c)
{
  int i = r * 4 + c;
  return REF_FLT_OFS(WorldToObject, i);
}

void fb_Fallback_SetWorldToObject(RuntimeDataType runtimeData, float12 M)
{
  REF_FLT_OFS(WorldToObject, 0)  = M[0]; 
  REF_FLT_OFS(WorldToObject, 1)  = M[1]; 
  REF_FLT_OFS(WorldToObject, 2)  = M[2]; 
  REF_FLT_OFS(WorldToObject, 3)  = M[3]; 
  REF_FLT_OFS(WorldToObject, 4)  = M[4]; 
  REF_FLT_OFS(WorldToObject, 5)  = M[5]; 
  REF_FLT_OFS(WorldToObject, 6)  = M[6]; 
  REF_FLT_OFS(WorldToObject, 7)  = M[7]; 
  REF_FLT_OFS(WorldToObject, 8)  = M[8]; 
  REF_FLT_OFS(WorldToObject, 9)  = M[9]; 
  REF_FLT_OFS(WorldToObject, 10) = M[10];
  REF_FLT_OFS(WorldToObject, 11) = M[11];
}

unsigned fb_dxop_primitiveID(RuntimeDataType runtimeData)
//unsigned fb_dxop_primitiveIndex(RuntimeDataType runtimeData)
{
  return REF(PrimitiveIndex);
}

unsigned fb_Fallback_PrimitiveIndex(RuntimeDataType runtimeData)
{
  return REF(PrimitiveIndex);
}

void fb_Fallback_SetPrimitiveIndex(RuntimeDataType runtimeData, unsigned i)
{
  REF(PrimitiveIndex) = i;
}

unsigned fb_Fallback_ShaderRecordOffset(RuntimeDataType runtimeData)
{
  return REF(ShaderRecordOffset);
}

void fb_Fallback_SetShaderRecordOffset(RuntimeDataType runtimeData, unsigned shaderRecordOffset)
{
  REF(ShaderRecordOffset) = shaderRecordOffset;
}

unsigned fb_dxop_instanceIndex(RuntimeDataType runtimeData)
{
  return REF(InstanceIndex);
}

unsigned fb_Fallback_InstanceIndex(RuntimeDataType runtimeData)
{
  return REF(InstanceIndex);
}

void fb_Fallback_SetInstanceIndex(RuntimeDataType runtimeData, unsigned i)
{
  REF(InstanceIndex) = i;
}

unsigned fb_dxop_instanceID(RuntimeDataType runtimeData)
{
  return REF(InstanceID);
}

unsigned fb_Fallback_InstanceID(RuntimeDataType runtimeData)
{
  return REF(InstanceID);
}

void fb_Fallback_SetInstanceID(RuntimeDataType runtimeData, unsigned i)
{
  REF(InstanceID) = i;
}

unsigned fb_dxop_hitKind(RuntimeDataType runtimeData)
{
  return REF(HitKind);
}

unsigned fb_Fallback_HitKind(RuntimeDataType runtimeData)
{
  return REF(HitKind);
}

void fb_Fallback_SetHitKind(RuntimeDataType runtimeData, unsigned i)
{
  REF(HitKind) = i;
}

float fb_dxop_pending_rayTCurrent(RuntimeDataType runtimeData)
{
  return REF_FLT(PendingRayTCurrent);
}

void fb_Fallback_SetPendingRayTCurrent(RuntimeDataType runtimeData, float t)
{
  REF_FLT(PendingRayTCurrent) = t;
}

unsigned fb_dxop_pending_primitiveID(RuntimeDataType runtimeData)
//unsigned fb_dxop_pending_primitiveIndex(RuntimeDataType runtimeData)
{
  return REF(PendingPrimitiveIndex);
}

unsigned fb_Fallback_PendingShaderRecordOffset(RuntimeDataType runtimeData)
{
  return REF(PendingShaderRecordOffset);
}

unsigned fb_dxop_pending_instanceIndex(RuntimeDataType runtimeData)
{
  return REF(PendingInstanceIndex);
}

unsigned fb_dxop_pending_instanceID(RuntimeDataType runtimeData)
{
  return REF(PendingInstanceID);
}

unsigned fb_dxop_pending_hitKind(RuntimeDataType runtimeData)
{
  return REF(PendingHitKind);
}

void fb_Fallback_SetPendingHitKind(RuntimeDataType runtimeData, unsigned i)
{
  REF(PendingHitKind) = i;
}

unsigned fb_Fallback_GroupIndex(RuntimeDataType runtimeData)
{ 
  return REF(GroupIndex);
}

int fb_Fallback_AnyHitResult(RuntimeDataType runtimeData)
{
  return REF(AnyHitResult);
}

void fb_Fallback_SetAnyHitResult(RuntimeDataType runtimeData, int result)
{
  REF(AnyHitResult) = result;
}

int fb_Fallback_AnyHitStateId(RuntimeDataType runtimeData)
{
  return REF(AnyHitStateId);
}

void fb_Fallback_SetAnyHitStateId(RuntimeDataType runtimeData, int id)
{
  REF(AnyHitStateId) = id;
}
