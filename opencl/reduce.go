package opencl

import (
	"fmt"
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
	util.Argument(in.NComp() == 1)
	var err error
	var result float32

	out := reduceBuf()

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in sum beginning: %+v \n", err)
		}
	}

	k_reducesum_async(in.DevPtr(0), out, 0,
		in.Len(), reducecfg, ClCmdQueue, nil)

	copyback(out, &result)
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in sum: %+v \n", err)
	}

	return result
}

// Dot product
func Dot(a, b *data.Slice) float32 {
	util.Argument(a.NComp() == b.NComp())
	util.Argument(a.Len() == b.Len())
	var err error
	results := make([]float32, a.NComp())
	result := float32(0)
	numComp := a.NComp()

	out := make([]unsafe.Pointer, numComp)
	for c := 0; c < numComp; c++ {
		out[c] = reduceBuf()
	}

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in dot beginning: %+v \n", err)
		}
	}

	for c := 0; c < numComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c), out[c], 0,
			a.Len(), reducecfg, ClCmdQueue, nil) // all components add to out
	}

	for c := 0; c < a.NComp(); c++ {
		copyback(out[c], &results[c])
	}
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in dot: %+v \n", err)
	}

	for _, v := range results {
		result += v
	}
	return result
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	var err error
	var result float32

	out := reduceBuf()

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in maxabs beginning: %+v \n", err)
		}
	}

	k_reducemaxabs_async(in.DevPtr(0), out, 0,
		in.Len(), reducecfg, ClCmdQueue, nil)

	copyback(out, &result)
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in maxabs: %+v \n", err)
	}

	return float32(result)
}

// Maximum element-wise difference
func MaxDiff(a, b *data.Slice) []float32 {
	util.Argument(a.NComp() == b.NComp())
	util.Argument(a.Len() == b.Len())
	var err error
	numComp := a.NComp()
	results := make([]float32, numComp)

	out := make([]unsafe.Pointer, numComp)
	for c := 0; c < numComp; c++ {
		out[c] = reduceBuf()
	}

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in maxdiff beginning: %+v \n", err)
		}
	}

	for c := 0; c < numComp; c++ {
		k_reducemaxdiff_async(a.DevPtr(c), b.DevPtr(c), out[c], 0,
			a.Len(), reducecfg, ClCmdQueue, nil)
	}

	for c := 0; c < numComp; c++ {
		copyback(out[c], &results[c])
	}
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in maxdiff: %+v \n", err)
	}
	return results
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//
//	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	util.Argument(v.NComp() == 3)
	var err error
	var result float32

	out := reduceBuf()

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in maxvecnorm beginning: %+v \n", err)
		}
	}

	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2),
		out, 0, v.Len(), reducecfg, ClCmdQueue, nil)

	copyback(out, &result)
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in maxvecnorm: %+v \n", err)
	}
	return math.Sqrt(float64(result))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	util.Argument(x.NComp() == 3)
	util.Argument(y.NComp() == 3)
	var err error
	var result float32

	out := reduceBuf()

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in maxvecdiff beginning: %+v \n", err)
		}
	}

	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg, ClCmdQueue, nil)

	copyback(out, &result)
	if err = ClCmdQueue.Finish(); err != nil {
		fmt.Printf("failed to wait for queue to finish in maxvecdiff: %+v \n", err)
	}
	return math.Sqrt(float64(result))
}

var reduceBuffers chan (*cl.MemObject) // pool of 1-float OpenCL buffers for reduce

// return a 1-float and an N-float OPENCL reduction buffer from a pool
// initialized to initVal
func reduceBuf() unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	return (unsafe.Pointer)(buf)
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer, result *float32) {
	MemCpyDtoH(unsafe.Pointer(result), buf, SIZEOF_FLOAT32)
	if err := ClCmdQueue.Finish(); err != nil {
		fmt.Printf("error waiting for queue to finish in copyback: %+v \n", err)
	}
	numVal := float32(0)
	_, err := ClCmdQueue.EnqueueFillBuffer((*cl.MemObject)(buf), unsafe.Pointer(&numVal), SIZEOF_FLOAT32, 0, SIZEOF_FLOAT32, nil)
	if err != nil {
		fmt.Printf("enqueuefillbuffer in copyback failed: %+v \n", err)
	}
	reduceBuffers <- (*cl.MemObject)(buf)
}

// initialize pool of 1-float and N-float OPENCL reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan *cl.MemObject, N)

	numVal := float32(0)
	for i := 0; i < N; i++ {
		buf := MemAlloc(SIZEOF_FLOAT32)
		_, err := ClCmdQueue.EnqueueFillBuffer(buf, unsafe.Pointer(&numVal), SIZEOF_FLOAT32, 0, SIZEOF_FLOAT32, nil)
		if err != nil {
			fmt.Printf("enqueuefillbuffer failed in initReduceBuf: %+v \n", err)
		}
		reduceBuffers <- buf
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: []int{1, 1, 1}, Block: []int{1, 1, 1}}
var ReduceWorkitems int
