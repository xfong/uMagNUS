package opencl

import (
	"fmt"
	"log"
	"math"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

/*
Need to update the reduction sum and dot algorithms to balance the distribution of inputs to the work-groups
Less of a problem for max and min because they are direct comparisons

for sum and dot, the magnitude of intermediate values depend on how many input values were summed to obtain
them. Thus, the entire input data should be distributed in a binary tree for the summation, and the work-
groups should "synchronize" at a fixed level of the tree. Small input sizes will need fewer work-groups
that can efficiently calculate for trees having short depths. Larger input sizes will need larger number of
possibly small work-groups.
*/

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	var err error
	util.Argument(in.NComp() == 1)

	out := reduceBuf(0)

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	event := k_reducesum_async(in.DevPtr(0), out, 0,
		in.Len(), reducecfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)

	results := copyback(out)
	return results
}

// Dot product
func Dot(a, b *data.Slice) float32 {
	var err error

	util.Argument(a.NComp() == b.NComp())
	util.Argument(a.Len() == b.Len())
	result := float32(0)
	numComp := a.NComp()
	out := make([]unsafe.Pointer, numComp)
	for c := 0; c < numComp; c++ {
		out[c] = reduceBuf(0)
	}

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	evtList := make([]*cl.Event, numComp)
	for c := 0; c < numComp; c++ {
		event := k_reducedot_async(a.DevPtr(c), b.DevPtr(c), out[c], 0,
			a.Len(), reducecfg, evtWL) // all components add to out
		// set event markers
		evtList[c] = event
		AddEventToSequence(event)
	}

	// set event marker
	UpdateLastEventList(evtList)

	for c := 0; c < a.NComp(); c++ {
		results := copyback(out[c])
		//		for _, v := range results {
		//			result += v
		//		}
		result += results
	}
	return result
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	var err error

	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	event := k_reducemaxabs_async(in.DevPtr(0), out, 0,
		in.Len(), reducecfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)

	results := copyback(out)
	return results
}

// Maximum element-wise difference
func MaxDiff(a, b *data.Slice) []float32 {
	var err error

	util.Argument(a.NComp() == b.NComp())
	util.Argument(a.Len() == b.Len())
	numComp := a.NComp()
	returnVal := make([]float32, numComp)
	out := make([]unsafe.Pointer, numComp)
	for c := 0; c < numComp; c++ {
		out[c] = reduceBuf(0)
	}

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	evtList := make([]*cl.Event, numComp)
	for c := 0; c < numComp; c++ {
		event := k_reducemaxdiff_async(a.DevPtr(c), b.DevPtr(c), out[c], 0,
			a.Len(), reducecfg, evtWL)
		// set event markers
		evtList[c] = event
		AddEventToSequence(event)
	}

	// set event marker
	UpdateLastEventList(evtList)

	for c := 0; c < numComp; c++ {
		returnVal[c] = copyback(out[c])
	}
	return returnVal
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//
//	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	var err error

	util.Argument(v.NComp() == 3)
	out := reduceBuf(0)

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	event := k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2),
		out, 0, v.Len(), reducecfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)

	results := copyback(out)
	return math.Sqrt(float64(results))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	var err error

	util.Argument(x.Len() == y.Len())
	util.Argument(x.NComp() == 3)
	util.Argument(y.NComp() == 3)
	out := reduceBuf(0)

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			fmt.Printf("failed to wait for last marker to finish in sum: %+v \n", err)
		}
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	event := k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)

	results := copyback(out)
	return math.Sqrt(float64(results))
}

var reduceBuffers chan (*cl.MemObject) // pool of 1-float OpenCL buffers for reduce

// return a 1-float and an N-float OPENCL reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	var err error
	var event *cl.Event
	var queue *cl.CommandQueue

	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute (i.e., zero buffer. needed?? why not zero on return of buffer to pool?)
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panic("failed to create command queue in reduceBuf: %+v \n", err)
	}
	if event, err = queue.EnqueueFillBuffer(buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, SIZEOF_FLOAT32, evtWL); err != nil {
		fmt.Printf("reduceBuf failed: %+v \n", err)
		return nil
	}

	if err = queue.Release(); err != nil { // implicit flush
		fmt.Printf("failed to release queue in reduceBuf: %+v \n", err)
	}

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)

	return (unsafe.Pointer)(buf)
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), buf, SIZEOF_FLOAT32)
	reduceBuffers <- (*cl.MemObject)(buf)
	return result
}

// initialize pool of 1-float and N-float OPENCL reduction buffers
// should we use sub-buffers instead? then we can zero all buffers
// using a single command
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan *cl.MemObject, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: []int{1, 1, 1}, Block: []int{1, 1, 1}}
var ReduceWorkitems int
