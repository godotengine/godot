/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "btBatchedConstraints.h"

#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btMinMax.h"
#include "LinearMath/btStackAlloc.h"
#include "LinearMath/btQuickprof.h"

#include <string.h> //for memset

const int kNoMerge = -1;

bool btBatchedConstraints::s_debugDrawBatches = false;


struct btBatchedConstraintInfo
{
    int constraintIndex;
    int numConstraintRows;
    int bodyIds[2];
};


struct btBatchInfo
{
    int numConstraints;
    int mergeIndex;

    btBatchInfo() : numConstraints(0), mergeIndex(kNoMerge) {}
};


bool btBatchedConstraints::validate(btConstraintArray* constraints, const btAlignedObjectArray<btSolverBody>& bodies) const
{
    //
    // validate: for debugging only. Verify coloring of bodies, that no body is touched by more than one batch in any given phase
    //
    int errors = 0;
    const int kUnassignedBatch = -1;

    btAlignedObjectArray<int> bodyBatchId;
    for (int iPhase = 0; iPhase < m_phases.size(); ++iPhase)
    {
        bodyBatchId.resizeNoInitialize(0);
        bodyBatchId.resize( bodies.size(), kUnassignedBatch );
        const Range& phase = m_phases[iPhase];
        for (int iBatch = phase.begin; iBatch < phase.end; ++iBatch)
        {
            const Range& batch = m_batches[iBatch];
            for (int iiCons = batch.begin; iiCons < batch.end; ++iiCons)
            {
                int iCons = m_constraintIndices[iiCons];
                const btSolverConstraint& cons = constraints->at(iCons);
                const btSolverBody& bodyA = bodies[cons.m_solverBodyIdA];
                const btSolverBody& bodyB = bodies[cons.m_solverBodyIdB];
                if (! bodyA.internalGetInvMass().isZero())
                {
                    int thisBodyBatchId = bodyBatchId[cons.m_solverBodyIdA];
                    if (thisBodyBatchId == kUnassignedBatch)
                    {
                        bodyBatchId[cons.m_solverBodyIdA] = iBatch;
                    }
                    else if (thisBodyBatchId != iBatch)
                    {
                        btAssert( !"dynamic body is used in 2 different batches in the same phase" );
                        errors++;
                    }
                }
                if (! bodyB.internalGetInvMass().isZero())
                {
                    int thisBodyBatchId = bodyBatchId[cons.m_solverBodyIdB];
                    if (thisBodyBatchId == kUnassignedBatch)
                    {
                        bodyBatchId[cons.m_solverBodyIdB] = iBatch;
                    }
                    else if (thisBodyBatchId != iBatch)
                    {
                        btAssert( !"dynamic body is used in 2 different batches in the same phase" );
                        errors++;
                    }
                }
            }
        }
    }
    return errors == 0;
}


static void debugDrawSingleBatch( const btBatchedConstraints* bc,
    btConstraintArray* constraints,
    const btAlignedObjectArray<btSolverBody>& bodies,
    int iBatch,
    const btVector3& color,
    const btVector3& offset
    )
{
    if (bc && bc->m_debugDrawer && iBatch < bc->m_batches.size())
    {
        const btBatchedConstraints::Range& b = bc->m_batches[iBatch];
        for (int iiCon = b.begin; iiCon < b.end; ++iiCon)
        {
            int iCon = bc->m_constraintIndices[iiCon];
            const btSolverConstraint& con = constraints->at(iCon);
            int iBody0 = con.m_solverBodyIdA;
            int iBody1 = con.m_solverBodyIdB;
            btVector3 pos0 = bodies[iBody0].getWorldTransform().getOrigin() + offset;
            btVector3 pos1 = bodies[iBody1].getWorldTransform().getOrigin() + offset;
            bc->m_debugDrawer->drawLine(pos0, pos1, color);
        }
    }
}


static void debugDrawPhase( const btBatchedConstraints* bc,
    btConstraintArray* constraints,
    const btAlignedObjectArray<btSolverBody>& bodies,
    int iPhase,
    const btVector3& color0,
    const btVector3& color1,
    const btVector3& offset
    )
{
    BT_PROFILE( "debugDrawPhase" );
    if ( bc && bc->m_debugDrawer && iPhase < bc->m_phases.size() )
    {
        const btBatchedConstraints::Range& phase = bc->m_phases[iPhase];
        for (int iBatch = phase.begin; iBatch < phase.end; ++iBatch)
        {
            float tt = float(iBatch - phase.begin) / float(btMax(1, phase.end - phase.begin - 1));
            btVector3 col = lerp(color0, color1, tt);
            debugDrawSingleBatch(bc, constraints, bodies, iBatch, col, offset);
        }
    }
}


static void debugDrawAllBatches( const btBatchedConstraints* bc,
    btConstraintArray* constraints,
    const btAlignedObjectArray<btSolverBody>& bodies
    )
{
    BT_PROFILE( "debugDrawAllBatches" );
    if ( bc && bc->m_debugDrawer && bc->m_phases.size() > 0 )
    {
        btVector3 bboxMin(BT_LARGE_FLOAT, BT_LARGE_FLOAT, BT_LARGE_FLOAT);
        btVector3 bboxMax = -bboxMin;
        for (int iBody = 0; iBody < bodies.size(); ++iBody)
        {
            const btVector3& pos = bodies[iBody].getWorldTransform().getOrigin();
            bboxMin.setMin(pos);
            bboxMax.setMax(pos);
        }
        btVector3 bboxExtent = bboxMax - bboxMin;
        btVector3 offsetBase = btVector3( 0, bboxExtent.y()*1.1f, 0 );
        btVector3 offsetStep = btVector3( 0, 0, bboxExtent.z()*1.1f );
        int numPhases = bc->m_phases.size();
        for (int iPhase = 0; iPhase < numPhases; ++iPhase)
        {
            float b = float(iPhase)/float(numPhases-1);
            btVector3 color0 = btVector3(1,0,b);
            btVector3 color1 = btVector3(0,1,b);
            btVector3 offset = offsetBase + offsetStep*(float(iPhase) - float(numPhases-1)*0.5);
            debugDrawPhase(bc, constraints, bodies, iPhase, color0, color1, offset);
        }
    }
}


static void initBatchedBodyDynamicFlags(btAlignedObjectArray<bool>* outBodyDynamicFlags, const btAlignedObjectArray<btSolverBody>& bodies)
{
    BT_PROFILE("initBatchedBodyDynamicFlags");
    btAlignedObjectArray<bool>& bodyDynamicFlags = *outBodyDynamicFlags;
    bodyDynamicFlags.resizeNoInitialize(bodies.size());
    for (int i = 0; i < bodies.size(); ++i)
    {
        const btSolverBody& body = bodies[ i ];
        bodyDynamicFlags[i] = ( body.internalGetInvMass().x() > btScalar( 0 ) );
    }
}


static int runLengthEncodeConstraintInfo(btBatchedConstraintInfo* outConInfos, int numConstraints)
{
    BT_PROFILE("runLengthEncodeConstraintInfo");
    // detect and run-length encode constraint rows that repeat the same bodies
    int iDest = 0;
    int iSrc = 0;
    while (iSrc < numConstraints)
    {
        const btBatchedConstraintInfo& srcConInfo = outConInfos[iSrc];
        btBatchedConstraintInfo& conInfo = outConInfos[iDest];
        conInfo.constraintIndex = iSrc;
        conInfo.bodyIds[0] = srcConInfo.bodyIds[0];
        conInfo.bodyIds[1] = srcConInfo.bodyIds[1];
        while (iSrc < numConstraints && outConInfos[iSrc].bodyIds[0] == srcConInfo.bodyIds[0] && outConInfos[iSrc].bodyIds[1] == srcConInfo.bodyIds[1])
        {
            ++iSrc;
        }
        conInfo.numConstraintRows = iSrc - conInfo.constraintIndex;
        ++iDest;
    }
    return iDest;
}


struct ReadSolverConstraintsLoop : public btIParallelForBody
{
    btBatchedConstraintInfo* m_outConInfos;
    btConstraintArray* m_constraints;

    ReadSolverConstraintsLoop( btBatchedConstraintInfo* outConInfos, btConstraintArray* constraints )
    {
        m_outConInfos = outConInfos;
        m_constraints = constraints;
    }
    void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
    {
        for (int i = iBegin; i < iEnd; ++i)
        {
            btBatchedConstraintInfo& conInfo = m_outConInfos[i];
            const btSolverConstraint& con = m_constraints->at( i );
            conInfo.bodyIds[0] = con.m_solverBodyIdA;
            conInfo.bodyIds[1] = con.m_solverBodyIdB;
            conInfo.constraintIndex = i;
            conInfo.numConstraintRows = 1;
        }
    }
};


static int initBatchedConstraintInfo(btBatchedConstraintInfo* outConInfos, btConstraintArray* constraints)
{
    BT_PROFILE("initBatchedConstraintInfo");
    int numConstraints = constraints->size();
    bool inParallel = true;
    if (inParallel)
    {
        ReadSolverConstraintsLoop loop(outConInfos, constraints);
        int grainSize = 1200;
        btParallelFor(0, numConstraints, grainSize, loop);
    }
    else
    {
        for (int i = 0; i < numConstraints; ++i)
        {
            btBatchedConstraintInfo& conInfo = outConInfos[i];
            const btSolverConstraint& con = constraints->at( i );
            conInfo.bodyIds[0] = con.m_solverBodyIdA;
            conInfo.bodyIds[1] = con.m_solverBodyIdB;
            conInfo.constraintIndex = i;
            conInfo.numConstraintRows = 1;
        }
    }
    bool useRunLengthEncoding = true;
    if (useRunLengthEncoding)
    {
        numConstraints = runLengthEncodeConstraintInfo(outConInfos, numConstraints);
    }
    return numConstraints;
}


static void expandConstraintRowsInPlace(int* constraintBatchIds, const btBatchedConstraintInfo* conInfos, int numConstraints, int numConstraintRows)
{
    BT_PROFILE("expandConstraintRowsInPlace");
    if (numConstraintRows > numConstraints)
    {
        // we walk the array in reverse to avoid overwriteing
        for (int iCon = numConstraints - 1; iCon >= 0; --iCon)
        {
            const btBatchedConstraintInfo& conInfo = conInfos[iCon];
            int iBatch = constraintBatchIds[iCon];
            for (int i = conInfo.numConstraintRows - 1; i >= 0; --i)
            {
                int iDest = conInfo.constraintIndex + i;
                btAssert(iDest >= iCon);
                btAssert(iDest >= 0 && iDest < numConstraintRows);
                constraintBatchIds[iDest] = iBatch;
            }
        }
    }
}


static void expandConstraintRows(int* destConstraintBatchIds, const int* srcConstraintBatchIds, const btBatchedConstraintInfo* conInfos, int numConstraints, int numConstraintRows)
{
    BT_PROFILE("expandConstraintRows");
    for ( int iCon = 0; iCon < numConstraints; ++iCon )
    {
        const btBatchedConstraintInfo& conInfo = conInfos[ iCon ];
        int iBatch = srcConstraintBatchIds[ iCon ];
        for ( int i = 0; i < conInfo.numConstraintRows; ++i )
        {
            int iDest = conInfo.constraintIndex + i;
            btAssert( iDest >= iCon );
            btAssert( iDest >= 0 && iDest < numConstraintRows );
            destConstraintBatchIds[ iDest ] = iBatch;
        }
    }
}


struct ExpandConstraintRowsLoop : public btIParallelForBody
{
    int* m_destConstraintBatchIds;
    const int* m_srcConstraintBatchIds;
    const btBatchedConstraintInfo* m_conInfos;
    int m_numConstraintRows;

    ExpandConstraintRowsLoop( int* destConstraintBatchIds, const int* srcConstraintBatchIds, const btBatchedConstraintInfo* conInfos, int numConstraintRows)
    {
        m_destConstraintBatchIds = destConstraintBatchIds;
        m_srcConstraintBatchIds = srcConstraintBatchIds;
        m_conInfos = conInfos;
        m_numConstraintRows = numConstraintRows;
    }
    void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
    {
        expandConstraintRows(m_destConstraintBatchIds, m_srcConstraintBatchIds + iBegin, m_conInfos + iBegin, iEnd - iBegin, m_numConstraintRows);
    }
};


static void expandConstraintRowsMt(int* destConstraintBatchIds, const int* srcConstraintBatchIds, const btBatchedConstraintInfo* conInfos, int numConstraints, int numConstraintRows)
{
    BT_PROFILE("expandConstraintRowsMt");
    ExpandConstraintRowsLoop loop(destConstraintBatchIds, srcConstraintBatchIds, conInfos, numConstraintRows);
    int grainSize = 600;
    btParallelFor(0, numConstraints, grainSize, loop);
}


static void initBatchedConstraintInfoArray(btAlignedObjectArray<btBatchedConstraintInfo>* outConInfos, btConstraintArray* constraints)
{
    BT_PROFILE("initBatchedConstraintInfoArray");
    btAlignedObjectArray<btBatchedConstraintInfo>& conInfos = *outConInfos;
    int numConstraints = constraints->size();
    conInfos.resizeNoInitialize(numConstraints);

    int newSize = initBatchedConstraintInfo(&outConInfos->at(0), constraints);
    conInfos.resizeNoInitialize(newSize);
}


static void mergeSmallBatches(btBatchInfo* batches, int iBeginBatch, int iEndBatch, int minBatchSize, int maxBatchSize)
{
    BT_PROFILE("mergeSmallBatches");
    for ( int iBatch = iEndBatch - 1; iBatch >= iBeginBatch; --iBatch )
    {
        btBatchInfo& batch = batches[ iBatch ];
        if ( batch.mergeIndex == kNoMerge && batch.numConstraints > 0 && batch.numConstraints < minBatchSize )
        {
            for ( int iDestBatch = iBatch - 1; iDestBatch >= iBeginBatch; --iDestBatch )
            {
                btBatchInfo& destBatch = batches[ iDestBatch ];
                if ( destBatch.mergeIndex == kNoMerge && ( destBatch.numConstraints + batch.numConstraints ) < maxBatchSize )
                {
                    destBatch.numConstraints += batch.numConstraints;
                    batch.numConstraints = 0;
                    batch.mergeIndex = iDestBatch;
                    break;
                }
            }
        }
    }
    // flatten mergeIndexes
    // e.g. in case where A was merged into B and then B was merged into C, we need A to point to C instead of B
    // Note: loop goes forward through batches because batches always merge from higher indexes to lower,
    //     so by going from low to high it reduces the amount of trail-following
    for ( int iBatch = iBeginBatch; iBatch < iEndBatch; ++iBatch )
    {
        btBatchInfo& batch = batches[ iBatch ];
        if ( batch.mergeIndex != kNoMerge )
        {
            int iMergeDest = batches[ batch.mergeIndex ].mergeIndex;
            // follow trail of merges to the end
            while ( iMergeDest != kNoMerge )
            {
                int iNext = batches[ iMergeDest ].mergeIndex;
                if ( iNext == kNoMerge )
                {
                    batch.mergeIndex = iMergeDest;
                    break;
                }
                iMergeDest = iNext;
            }
        }
    }
}


static void updateConstraintBatchIdsForMerges(int* constraintBatchIds, int numConstraints, const btBatchInfo* batches, int numBatches)
{
    BT_PROFILE("updateConstraintBatchIdsForMerges");
    // update batchIds to account for merges
    for (int i = 0; i < numConstraints; ++i)
    {
        int iBatch = constraintBatchIds[i];
        btAssert(iBatch < numBatches);
        // if this constraint references a batch that was merged into another batch
        if (batches[iBatch].mergeIndex != kNoMerge)
        {
            // update batchId
            constraintBatchIds[i] = batches[iBatch].mergeIndex;
        }
    }
}


struct UpdateConstraintBatchIdsForMergesLoop : public btIParallelForBody
{
    int* m_constraintBatchIds;
    const btBatchInfo* m_batches;
    int m_numBatches;

    UpdateConstraintBatchIdsForMergesLoop( int* constraintBatchIds, const btBatchInfo* batches, int numBatches )
    {
        m_constraintBatchIds = constraintBatchIds;
        m_batches = batches;
        m_numBatches = numBatches;
    }
    void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
    {
        BT_PROFILE( "UpdateConstraintBatchIdsForMergesLoop" );
        updateConstraintBatchIdsForMerges( m_constraintBatchIds + iBegin, iEnd - iBegin, m_batches, m_numBatches );
    }
};


static void updateConstraintBatchIdsForMergesMt(int* constraintBatchIds, int numConstraints, const btBatchInfo* batches, int numBatches)
{
    BT_PROFILE( "updateConstraintBatchIdsForMergesMt" );
    UpdateConstraintBatchIdsForMergesLoop loop(constraintBatchIds, batches, numBatches);
    int grainSize = 800;
    btParallelFor(0, numConstraints, grainSize, loop);
}


inline bool BatchCompare(const btBatchedConstraints::Range& a, const btBatchedConstraints::Range& b)
{
    int lenA = a.end - a.begin;
    int lenB = b.end - b.begin;
    return lenA > lenB;
}


static void writeOutConstraintIndicesForRangeOfBatches(btBatchedConstraints* bc,
    const int* constraintBatchIds,
    int numConstraints,
    int* constraintIdPerBatch,
    int batchBegin,
    int batchEnd
    )
{
    BT_PROFILE("writeOutConstraintIndicesForRangeOfBatches");
    for ( int iCon = 0; iCon < numConstraints; ++iCon )
    {
        int iBatch = constraintBatchIds[ iCon ];
        if (iBatch >= batchBegin && iBatch < batchEnd)
        {
            int iDestCon = constraintIdPerBatch[ iBatch ];
            constraintIdPerBatch[ iBatch ] = iDestCon + 1;
            bc->m_constraintIndices[ iDestCon ] = iCon;
        }
    }
}


struct WriteOutConstraintIndicesLoop : public btIParallelForBody
{
    btBatchedConstraints* m_batchedConstraints;
    const int* m_constraintBatchIds;
    int m_numConstraints;
    int* m_constraintIdPerBatch;
    int m_maxNumBatchesPerPhase;

    WriteOutConstraintIndicesLoop( btBatchedConstraints* bc, const int* constraintBatchIds, int numConstraints, int* constraintIdPerBatch, int maxNumBatchesPerPhase )
    {
        m_batchedConstraints = bc;
        m_constraintBatchIds = constraintBatchIds;
        m_numConstraints = numConstraints;
        m_constraintIdPerBatch = constraintIdPerBatch;
        m_maxNumBatchesPerPhase = maxNumBatchesPerPhase;
    }
    void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
    {
        BT_PROFILE( "WriteOutConstraintIndicesLoop" );
        int batchBegin = iBegin * m_maxNumBatchesPerPhase;
        int batchEnd = iEnd * m_maxNumBatchesPerPhase;
        writeOutConstraintIndicesForRangeOfBatches(m_batchedConstraints,
            m_constraintBatchIds,
            m_numConstraints,
            m_constraintIdPerBatch,
            batchBegin,
            batchEnd
        );
    }
};


static void writeOutConstraintIndicesMt(btBatchedConstraints* bc,
    const int* constraintBatchIds,
    int numConstraints,
    int* constraintIdPerBatch,
    int maxNumBatchesPerPhase,
    int numPhases
    )
{
    BT_PROFILE("writeOutConstraintIndicesMt");
    bool inParallel = true;
    if (inParallel)
    {
        WriteOutConstraintIndicesLoop loop( bc, constraintBatchIds, numConstraints, constraintIdPerBatch, maxNumBatchesPerPhase );
        btParallelFor( 0, numPhases, 1, loop );
    }
    else
    {
        for ( int iCon = 0; iCon < numConstraints; ++iCon )
        {
            int iBatch = constraintBatchIds[ iCon ];
            int iDestCon = constraintIdPerBatch[ iBatch ];
            constraintIdPerBatch[ iBatch ] = iDestCon + 1;
            bc->m_constraintIndices[ iDestCon ] = iCon;
        }
    }
}


static void writeGrainSizes(btBatchedConstraints* bc)
{
    typedef btBatchedConstraints::Range Range;
    int numPhases = bc->m_phases.size();
    bc->m_phaseGrainSize.resizeNoInitialize(numPhases);
    int numThreads = btGetTaskScheduler()->getNumThreads();
    for (int iPhase = 0; iPhase < numPhases; ++iPhase)
    {
        const Range& phase = bc->m_phases[ iPhase ];
        int numBatches = phase.end - phase.begin;
        float grainSize = floor((0.25f*numBatches / float(numThreads)) + 0.0f);
        bc->m_phaseGrainSize[ iPhase ] = btMax(1, int(grainSize));
    }
}


static void writeOutBatches(btBatchedConstraints* bc,
    const int* constraintBatchIds,
    int numConstraints,
    const btBatchInfo* batches,
    int* batchWork,
    int maxNumBatchesPerPhase,
    int numPhases
)
{
    BT_PROFILE("writeOutBatches");
    typedef btBatchedConstraints::Range Range;
    bc->m_constraintIndices.reserve( numConstraints );
    bc->m_batches.resizeNoInitialize( 0 );
    bc->m_phases.resizeNoInitialize( 0 );

    //int maxNumBatches = numPhases * maxNumBatchesPerPhase;
    {
        int* constraintIdPerBatch = batchWork;  // for each batch, keep an index into the next available slot in the m_constraintIndices array
        int iConstraint = 0;
        for (int iPhase = 0; iPhase < numPhases; ++iPhase)
        {
            int curPhaseBegin = bc->m_batches.size();
            int iBegin = iPhase * maxNumBatchesPerPhase;
            int iEnd = iBegin + maxNumBatchesPerPhase;
            for ( int i = iBegin; i < iEnd; ++i )
            {
                const btBatchInfo& batch = batches[ i ];
                int curBatchBegin = iConstraint;
                constraintIdPerBatch[ i ] = curBatchBegin;  // record the start of each batch in m_constraintIndices array
                int numConstraints = batch.numConstraints;
                iConstraint += numConstraints;
                if ( numConstraints > 0 )
                {
                    bc->m_batches.push_back( Range( curBatchBegin, iConstraint ) );
                }
            }
            // if any batches were emitted this phase,
            if ( bc->m_batches.size() > curPhaseBegin )
            {
                // output phase
                bc->m_phases.push_back( Range( curPhaseBegin, bc->m_batches.size() ) );
            }
        }

        btAssert(iConstraint == numConstraints);
        bc->m_constraintIndices.resizeNoInitialize( numConstraints );
        writeOutConstraintIndicesMt( bc, constraintBatchIds, numConstraints, constraintIdPerBatch, maxNumBatchesPerPhase, numPhases );
    }
    // for each phase
    for (int iPhase = 0; iPhase < bc->m_phases.size(); ++iPhase)
    {
        // sort the batches from largest to smallest (can be helpful to some task schedulers)
        const Range& curBatches = bc->m_phases[iPhase];
        bc->m_batches.quickSortInternal(BatchCompare, curBatches.begin, curBatches.end-1);
    }
    bc->m_phaseOrder.resize(bc->m_phases.size());
    for (int i = 0; i < bc->m_phases.size(); ++i)
    {
        bc->m_phaseOrder[i] = i;
    }
    writeGrainSizes(bc);
}


//
// PreallocatedMemoryHelper -- helper object for allocating a number of chunks of memory in a single contiguous block.
//                             It is generally more efficient to do a single larger allocation than many smaller allocations.
//
// Example Usage:
//
//  btVector3* bodyPositions = NULL;
//  btBatchedConstraintInfo* conInfos = NULL;
//  {
//    PreallocatedMemoryHelper<8> memHelper;
//    memHelper.addChunk( (void**) &bodyPositions, sizeof( btVector3 ) * bodies.size() );
//    memHelper.addChunk( (void**) &conInfos, sizeof( btBatchedConstraintInfo ) * numConstraints );
//    void* memPtr = malloc( memHelper.getSizeToAllocate() );  // allocate the memory
//    memHelper.setChunkPointers( memPtr );  // update pointers to chunks
//  }
template <int N>
class PreallocatedMemoryHelper
{
    struct Chunk
    {
        void** ptr;
        size_t size;
    };
    Chunk m_chunks[N];
    int m_numChunks;
public:
    PreallocatedMemoryHelper() {m_numChunks=0;}
    void addChunk( void** ptr, size_t sz )
    {
        btAssert( m_numChunks < N );
        if ( m_numChunks < N )
        {
            Chunk& chunk = m_chunks[ m_numChunks ];
            chunk.ptr = ptr;
            chunk.size = sz;
            m_numChunks++;
        }
    }
    size_t getSizeToAllocate() const
    {
        size_t totalSize = 0;
        for (int i = 0; i < m_numChunks; ++i)
        {
            totalSize += m_chunks[i].size;
        }
        return totalSize;
    }
    void setChunkPointers(void* mem) const
    {
        size_t totalSize = 0;
        for (int i = 0; i < m_numChunks; ++i)
        {
            const Chunk& chunk = m_chunks[ i ];
            char* chunkPtr = static_cast<char*>(mem) + totalSize;
            *chunk.ptr = chunkPtr;
            totalSize += chunk.size;
        }
    }
};



static btVector3 findMaxDynamicConstraintExtent(
    btVector3* bodyPositions,
    bool* bodyDynamicFlags,
    btBatchedConstraintInfo* conInfos,
    int numConstraints,
    int numBodies
    )
{
    BT_PROFILE("findMaxDynamicConstraintExtent");
    btVector3 consExtent = btVector3(1,1,1) * 0.001;
    for (int iCon = 0; iCon < numConstraints; ++iCon)
    {
        const btBatchedConstraintInfo& con = conInfos[ iCon ];
        int iBody0 = con.bodyIds[0];
        int iBody1 = con.bodyIds[1];
        btAssert(iBody0 >= 0 && iBody0 < numBodies);
        btAssert(iBody1 >= 0 && iBody1 < numBodies);
        // is it a dynamic constraint?
        if (bodyDynamicFlags[iBody0] && bodyDynamicFlags[iBody1])
        {
            btVector3 delta = bodyPositions[iBody1] - bodyPositions[iBody0];
            consExtent.setMax(delta.absolute());
        }
    }
    return consExtent;
}


struct btIntVec3
{
    int m_ints[ 3 ];

    SIMD_FORCE_INLINE const int& operator[](int i) const {return m_ints[i];}
    SIMD_FORCE_INLINE int&       operator[](int i)       {return m_ints[i];}
};


struct AssignConstraintsToGridBatchesParams
{
    bool* bodyDynamicFlags;
    btIntVec3* bodyGridCoords;
    int numBodies;
    btBatchedConstraintInfo* conInfos;
    int* constraintBatchIds;
    btIntVec3 gridChunkDim;
    int maxNumBatchesPerPhase;
    int numPhases;
    int phaseMask;

    AssignConstraintsToGridBatchesParams()
    {
        memset(this, 0, sizeof(*this));
    }
};


static void assignConstraintsToGridBatches(const AssignConstraintsToGridBatchesParams& params, int iConBegin, int iConEnd)
{
    BT_PROFILE("assignConstraintsToGridBatches");
    // (can be done in parallel)
    for ( int iCon = iConBegin; iCon < iConEnd; ++iCon )
    {
        const btBatchedConstraintInfo& con = params.conInfos[ iCon ];
        int iBody0 = con.bodyIds[ 0 ];
        int iBody1 = con.bodyIds[ 1 ];
        int iPhase = iCon; //iBody0; // pseudorandom choice to distribute evenly amongst phases
        iPhase &= params.phaseMask;
        int gridCoord[ 3 ];
        // is it a dynamic constraint?
        if ( params.bodyDynamicFlags[ iBody0 ] && params.bodyDynamicFlags[ iBody1 ] )
        {
            const btIntVec3& body0Coords = params.bodyGridCoords[iBody0];
            const btIntVec3& body1Coords = params.bodyGridCoords[iBody1];
            // for each dimension x,y,z,
            for (int i = 0; i < 3; ++i)
            {
                int coordMin = btMin(body0Coords.m_ints[i], body1Coords.m_ints[i]);
                int coordMax = btMax(body0Coords.m_ints[i], body1Coords.m_ints[i]);
                if (coordMin != coordMax)
                {
                    btAssert( coordMax == coordMin + 1 );
                    if ((coordMin&1) == 0)
                    {
                        iPhase &= ~(1 << i); // force bit off
                    }
                    else
                    {
                        iPhase |= (1 << i); // force bit on
                        iPhase &= params.phaseMask;
                    }
                }
                gridCoord[ i ] = coordMin;
            }
        }
        else
        {
            if ( !params.bodyDynamicFlags[ iBody0 ] )
            {
                iBody0 = con.bodyIds[ 1 ];
            }
            btAssert(params.bodyDynamicFlags[ iBody0 ]);
            const btIntVec3& body0Coords = params.bodyGridCoords[iBody0];
            // for each dimension x,y,z,
            for ( int i = 0; i < 3; ++i )
            {
                gridCoord[ i ] = body0Coords.m_ints[ i ];
            }
        }
        // calculate chunk coordinates
        int chunkCoord[ 3 ];
        btIntVec3 gridChunkDim = params.gridChunkDim;
        // for each dimension x,y,z,
        for ( int i = 0; i < 3; ++i )
        {
            int coordOffset = ( iPhase >> i ) & 1;
            chunkCoord[ i ] = (gridCoord[ i ] - coordOffset)/2;
            btClamp( chunkCoord[ i ], 0, gridChunkDim[ i ] - 1);
            btAssert( chunkCoord[ i ] < gridChunkDim[ i ] );
        }
        int iBatch = iPhase * params.maxNumBatchesPerPhase + chunkCoord[ 0 ] + chunkCoord[ 1 ] * gridChunkDim[ 0 ] + chunkCoord[ 2 ] * gridChunkDim[ 0 ] * gridChunkDim[ 1 ];
        btAssert(iBatch >= 0 && iBatch < params.maxNumBatchesPerPhase*params.numPhases);
        params.constraintBatchIds[ iCon ] = iBatch;
    }
}


struct AssignConstraintsToGridBatchesLoop : public btIParallelForBody
{
    const AssignConstraintsToGridBatchesParams* m_params;

    AssignConstraintsToGridBatchesLoop( const AssignConstraintsToGridBatchesParams& params )
    {
        m_params = &params;
    }
    void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
    {
        assignConstraintsToGridBatches(*m_params, iBegin, iEnd);
    }
};


//
// setupSpatialGridBatchesMt -- generate batches using a uniform 3D grid
//
/*

Bodies are treated as 3D points at their center of mass. We only consider dynamic bodies at this stage,
because only dynamic bodies are mutated when a constraint is solved, thus subject to race conditions.

1. Compute a bounding box around all dynamic bodies
2. Compute the maximum extent of all dynamic constraints. Each dynamic constraint is treated as a line segment, and we need the size of
   box that will fully enclose any single dynamic constraint

3. Establish the cell size of our grid, the cell size in each dimension must be at least as large as the dynamic constraints max-extent,
   so that no dynamic constraint can span more than 2 cells of our grid on any axis of the grid. The cell size should be adjusted
   larger in order to keep the total number of cells from being excessively high

Key idea: Given that each constraint spans 1 or 2 grid cells in each dimension, we can handle all constraints by processing
          in chunks of 2x2x2 cells with 8 different 1-cell offsets ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0)...).
          For each of the 8 offsets, we create a phase, and for each 2x2x2 chunk with dynamic constraints becomes a batch in that phase.

4. Once the grid is established, we can calculate for each constraint which phase and batch it belongs in.

5. Do a merge small batches on the batches of each phase separately, to try to even out the sizes of batches

Optionally, we can "collapse" one dimension of our 3D grid to turn it into a 2D grid, which reduces the number of phases
to 4. With fewer phases, there are more constraints per phase and this makes it easier to create batches of a useful size.
*/
//
static void setupSpatialGridBatchesMt(
    btBatchedConstraints* batchedConstraints,
    btAlignedObjectArray<char>* scratchMemory,
    btConstraintArray* constraints,
    const btAlignedObjectArray<btSolverBody>& bodies,
    int minBatchSize,
    int maxBatchSize,
    bool use2DGrid
)
{
    BT_PROFILE("setupSpatialGridBatchesMt");
    const int numPhases = 8;
    int numConstraints = constraints->size();
    int numConstraintRows = constraints->size();

    const int maxGridChunkCount = 128;
    int allocNumBatchesPerPhase = maxGridChunkCount;
    int minNumBatchesPerPhase = 16;
    int allocNumBatches = allocNumBatchesPerPhase * numPhases;

    btVector3* bodyPositions = NULL;
    bool* bodyDynamicFlags = NULL;
    btIntVec3* bodyGridCoords = NULL;
    btBatchInfo* batches = NULL;
    int* batchWork = NULL;
    btBatchedConstraintInfo* conInfos = NULL;
    int* constraintBatchIds = NULL;
    int* constraintRowBatchIds = NULL;
    {
        PreallocatedMemoryHelper<10> memHelper;
        memHelper.addChunk( (void**) &bodyPositions, sizeof( btVector3 ) * bodies.size() );
        memHelper.addChunk( (void**) &bodyDynamicFlags, sizeof( bool ) * bodies.size() );
        memHelper.addChunk( (void**) &bodyGridCoords, sizeof( btIntVec3 ) * bodies.size() );
        memHelper.addChunk( (void**) &batches, sizeof( btBatchInfo )* allocNumBatches );
        memHelper.addChunk( (void**) &batchWork, sizeof( int )* allocNumBatches );
        memHelper.addChunk( (void**) &conInfos, sizeof( btBatchedConstraintInfo ) * numConstraints );
        memHelper.addChunk( (void**) &constraintBatchIds, sizeof( int ) * numConstraints );
        memHelper.addChunk( (void**) &constraintRowBatchIds, sizeof( int ) * numConstraintRows );
        size_t scratchSize = memHelper.getSizeToAllocate();
        // if we need to reallocate
        if (scratchMemory->capacity() < scratchSize)
        {
            // allocate 6.25% extra to avoid repeated reallocs
            scratchMemory->reserve( scratchSize + scratchSize/16 );
        }
        scratchMemory->resizeNoInitialize( scratchSize );
        char* memPtr = &scratchMemory->at(0);
        memHelper.setChunkPointers( memPtr );
    }

    numConstraints = initBatchedConstraintInfo(conInfos, constraints);

    // compute bounding box around all dynamic bodies
    // (could be done in parallel)
    btVector3 bboxMin(BT_LARGE_FLOAT, BT_LARGE_FLOAT, BT_LARGE_FLOAT);
    btVector3 bboxMax = -bboxMin;
    //int dynamicBodyCount = 0;
    for (int i = 0; i < bodies.size(); ++i)
    {
        const btSolverBody& body = bodies[i];
        btVector3 bodyPos = body.getWorldTransform().getOrigin();
        bool isDynamic = ( body.internalGetInvMass().x() > btScalar( 0 ) );
        bodyPositions[i] = bodyPos;
        bodyDynamicFlags[i] = isDynamic;
        if (isDynamic)
        {
            //dynamicBodyCount++;
            bboxMin.setMin(bodyPos);
            bboxMax.setMax(bodyPos);
        }
    }

    // find max extent of all dynamic constraints
    // (could be done in parallel)
    btVector3 consExtent = findMaxDynamicConstraintExtent(bodyPositions, bodyDynamicFlags, conInfos, numConstraints, bodies.size());

    btVector3 gridExtent = bboxMax - bboxMin;

    btVector3 gridCellSize = consExtent;
    int gridDim[3];
    gridDim[ 0 ] = int( 1.0 + gridExtent.x() / gridCellSize.x() );
    gridDim[ 1 ] = int( 1.0 + gridExtent.y() / gridCellSize.y() );
    gridDim[ 2 ] = int( 1.0 + gridExtent.z() / gridCellSize.z() );

    // if we can collapse an axis, it will cut our number of phases in half which could be more efficient
    int phaseMask = 7;
    bool collapseAxis = use2DGrid;
    if ( collapseAxis )
    {
        // pick the smallest axis to collapse, leaving us with the greatest number of cells in our grid
        int iAxisToCollapse = 0;
        int axisDim = gridDim[iAxisToCollapse];
        //for each dimension
        for ( int i = 0; i < 3; ++i )
        {
            if (gridDim[i] < axisDim)
            {
                iAxisToCollapse = i;
                axisDim = gridDim[i];
            }
        }
        // collapse it
        gridCellSize[iAxisToCollapse] = gridExtent[iAxisToCollapse] * 2.0f;
        phaseMask &= ~(1 << iAxisToCollapse);
    }

    int numGridChunks = 0;
    btIntVec3 gridChunkDim;  // each chunk is 2x2x2 group of cells
    while (true)
    {
        gridDim[0] = int( 1.0 + gridExtent.x() / gridCellSize.x() );
        gridDim[1] = int( 1.0 + gridExtent.y() / gridCellSize.y() );
        gridDim[2] = int( 1.0 + gridExtent.z() / gridCellSize.z() );
        gridChunkDim[ 0 ] = btMax( 1, ( gridDim[ 0 ] + 0 ) / 2 );
        gridChunkDim[ 1 ] = btMax( 1, ( gridDim[ 1 ] + 0 ) / 2 );
        gridChunkDim[ 2 ] = btMax( 1, ( gridDim[ 2 ] + 0 ) / 2 );
        numGridChunks = gridChunkDim[ 0 ] * gridChunkDim[ 1 ] * gridChunkDim[ 2 ];
        float nChunks = float(gridChunkDim[0]) * float(gridChunkDim[1]) * float(gridChunkDim[2]);  // suceptible to integer overflow
        if ( numGridChunks <= maxGridChunkCount && nChunks <= maxGridChunkCount )
        {
            break;
        }
        gridCellSize *= 1.25; // should roughly cut numCells in half
    }
    btAssert(numGridChunks <= maxGridChunkCount );
    int maxNumBatchesPerPhase = numGridChunks;

    // for each dynamic body, compute grid coords
    btVector3 invGridCellSize = btVector3(1,1,1)/gridCellSize;
    // (can be done in parallel)
    for (int iBody = 0; iBody < bodies.size(); ++iBody)
    {
        btIntVec3& coords = bodyGridCoords[iBody];
        if (bodyDynamicFlags[iBody])
        {
            btVector3 v = ( bodyPositions[ iBody ] - bboxMin )*invGridCellSize;
            coords.m_ints[0] = int(v.x());
            coords.m_ints[1] = int(v.y());
            coords.m_ints[2] = int(v.z());
            btAssert(coords.m_ints[0] >= 0 && coords.m_ints[0] < gridDim[0]);
            btAssert(coords.m_ints[1] >= 0 && coords.m_ints[1] < gridDim[1]);
            btAssert(coords.m_ints[2] >= 0 && coords.m_ints[2] < gridDim[2]);
        }
        else
        {
            coords.m_ints[0] = -1;
            coords.m_ints[1] = -1;
            coords.m_ints[2] = -1;
        }
    }

    for (int iPhase = 0; iPhase < numPhases; ++iPhase)
    {
        int batchBegin = iPhase * maxNumBatchesPerPhase;
        int batchEnd = batchBegin + maxNumBatchesPerPhase;
        for ( int iBatch = batchBegin; iBatch < batchEnd; ++iBatch )
        {
            btBatchInfo& batch = batches[ iBatch ];
            batch = btBatchInfo();
        }
    }

    {
        AssignConstraintsToGridBatchesParams params;
        params.bodyDynamicFlags = bodyDynamicFlags;
        params.bodyGridCoords = bodyGridCoords;
        params.numBodies = bodies.size();
        params.conInfos = conInfos;
        params.constraintBatchIds = constraintBatchIds;
        params.gridChunkDim = gridChunkDim;
        params.maxNumBatchesPerPhase = maxNumBatchesPerPhase;
        params.numPhases = numPhases;
        params.phaseMask = phaseMask;
        bool inParallel = true;
        if (inParallel)
        {
            AssignConstraintsToGridBatchesLoop loop(params);
            int grainSize = 250;
            btParallelFor(0, numConstraints, grainSize, loop);
        }
        else
        {
            assignConstraintsToGridBatches( params, 0, numConstraints );
        }
    }
    for ( int iCon = 0; iCon < numConstraints; ++iCon )
    {
        const btBatchedConstraintInfo& con = conInfos[ iCon ];
        int iBatch = constraintBatchIds[ iCon ];
        btBatchInfo& batch = batches[iBatch];
        batch.numConstraints += con.numConstraintRows;
    }

    for (int iPhase = 0; iPhase < numPhases; ++iPhase)
    {
        // if phase is legit,
        if (iPhase == (iPhase&phaseMask))
        {
            int iBeginBatch = iPhase * maxNumBatchesPerPhase;
            int iEndBatch = iBeginBatch + maxNumBatchesPerPhase;
            mergeSmallBatches( batches, iBeginBatch, iEndBatch, minBatchSize, maxBatchSize );
        }
    }
    // all constraints have been assigned a batchId
    updateConstraintBatchIdsForMergesMt(constraintBatchIds, numConstraints, batches, maxNumBatchesPerPhase*numPhases);

    if (numConstraintRows > numConstraints)
    {
        expandConstraintRowsMt(&constraintRowBatchIds[0], &constraintBatchIds[0], &conInfos[0], numConstraints, numConstraintRows);
    }
    else
    {
        constraintRowBatchIds = constraintBatchIds;
    }

    writeOutBatches(batchedConstraints, constraintRowBatchIds, numConstraintRows, batches, batchWork, maxNumBatchesPerPhase, numPhases);
    btAssert(batchedConstraints->validate(constraints, bodies));
}


static void setupSingleBatch(
    btBatchedConstraints* bc,
    int numConstraints
)
{
    BT_PROFILE("setupSingleBatch");
    typedef btBatchedConstraints::Range Range;

    bc->m_constraintIndices.resize( numConstraints );
    for ( int i = 0; i < numConstraints; ++i )
    {
        bc->m_constraintIndices[ i ] = i;
    }

    bc->m_batches.resizeNoInitialize( 0 );
    bc->m_phases.resizeNoInitialize( 0 );
    bc->m_phaseOrder.resizeNoInitialize( 0 );
    bc->m_phaseGrainSize.resizeNoInitialize( 0 );

    if (numConstraints > 0)
    {
        bc->m_batches.push_back( Range( 0, numConstraints ) );
        bc->m_phases.push_back( Range( 0, 1 ) );
        bc->m_phaseOrder.push_back(0);
        bc->m_phaseGrainSize.push_back(1);
    }
}


void btBatchedConstraints::setup(
    btConstraintArray* constraints,
    const btAlignedObjectArray<btSolverBody>& bodies,
    BatchingMethod batchingMethod,
    int minBatchSize,
    int maxBatchSize,
    btAlignedObjectArray<char>* scratchMemory
    )
{
    if (constraints->size() >= minBatchSize*4)
    {
        bool use2DGrid = batchingMethod == BATCHING_METHOD_SPATIAL_GRID_2D;
        setupSpatialGridBatchesMt( this, scratchMemory, constraints, bodies, minBatchSize, maxBatchSize, use2DGrid );
        if (s_debugDrawBatches)
        {
            debugDrawAllBatches( this, constraints, bodies );
        }
    }
    else
    {
        setupSingleBatch( this, constraints->size() );
    }
}


