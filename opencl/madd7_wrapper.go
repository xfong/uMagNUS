package opencl

/*
 THIS FILE IS AUTO-GENERATED BY OCL2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/seeder-research/uMagNUS/cl"
	"github.com/seeder-research/uMagNUS/timer"
	"sync"
	"unsafe"
)

// Stores the arguments for madd7 kernel invocation
type madd7_args_t struct {
	arg_dst  unsafe.Pointer
	arg_src1 unsafe.Pointer
	arg_fac1 float32
	arg_src2 unsafe.Pointer
	arg_fac2 float32
	arg_src3 unsafe.Pointer
	arg_fac3 float32
	arg_src4 unsafe.Pointer
	arg_fac4 float32
	arg_src5 unsafe.Pointer
	arg_fac5 float32
	arg_src6 unsafe.Pointer
	arg_fac6 float32
	arg_src7 unsafe.Pointer
	arg_fac7 float32
	arg_N    int
	argptr   [16]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for madd7 kernel invocation
var madd7_args madd7_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	madd7_args.argptr[0] = unsafe.Pointer(&madd7_args.arg_dst)
	madd7_args.argptr[1] = unsafe.Pointer(&madd7_args.arg_src1)
	madd7_args.argptr[2] = unsafe.Pointer(&madd7_args.arg_fac1)
	madd7_args.argptr[3] = unsafe.Pointer(&madd7_args.arg_src2)
	madd7_args.argptr[4] = unsafe.Pointer(&madd7_args.arg_fac2)
	madd7_args.argptr[5] = unsafe.Pointer(&madd7_args.arg_src3)
	madd7_args.argptr[6] = unsafe.Pointer(&madd7_args.arg_fac3)
	madd7_args.argptr[7] = unsafe.Pointer(&madd7_args.arg_src4)
	madd7_args.argptr[8] = unsafe.Pointer(&madd7_args.arg_fac4)
	madd7_args.argptr[9] = unsafe.Pointer(&madd7_args.arg_src5)
	madd7_args.argptr[10] = unsafe.Pointer(&madd7_args.arg_fac5)
	madd7_args.argptr[11] = unsafe.Pointer(&madd7_args.arg_src6)
	madd7_args.argptr[12] = unsafe.Pointer(&madd7_args.arg_fac6)
	madd7_args.argptr[13] = unsafe.Pointer(&madd7_args.arg_src7)
	madd7_args.argptr[14] = unsafe.Pointer(&madd7_args.arg_fac7)
	madd7_args.argptr[15] = unsafe.Pointer(&madd7_args.arg_N)
}

// Wrapper for madd7 OpenCL kernel, asynchronous.
func k_madd7_async(dst unsafe.Pointer, src1 unsafe.Pointer, fac1 float32, src2 unsafe.Pointer, fac2 float32, src3 unsafe.Pointer, fac3 float32, src4 unsafe.Pointer, fac4 float32, src5 unsafe.Pointer, fac5 float32, src6 unsafe.Pointer, fac6 float32, src7 unsafe.Pointer, fac7 float32, N int, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("madd7")
	}

	madd7_args.Lock()
	defer madd7_args.Unlock()

	madd7_args.arg_dst = dst
	madd7_args.arg_src1 = src1
	madd7_args.arg_fac1 = fac1
	madd7_args.arg_src2 = src2
	madd7_args.arg_fac2 = fac2
	madd7_args.arg_src3 = src3
	madd7_args.arg_fac3 = fac3
	madd7_args.arg_src4 = src4
	madd7_args.arg_fac4 = fac4
	madd7_args.arg_src5 = src5
	madd7_args.arg_fac5 = fac5
	madd7_args.arg_src6 = src6
	madd7_args.arg_fac6 = fac6
	madd7_args.arg_src7 = src7
	madd7_args.arg_fac7 = fac7
	madd7_args.arg_N = N

	SetKernelArgWrapper("madd7", 0, dst)
	SetKernelArgWrapper("madd7", 1, src1)
	SetKernelArgWrapper("madd7", 2, fac1)
	SetKernelArgWrapper("madd7", 3, src2)
	SetKernelArgWrapper("madd7", 4, fac2)
	SetKernelArgWrapper("madd7", 5, src3)
	SetKernelArgWrapper("madd7", 6, fac3)
	SetKernelArgWrapper("madd7", 7, src4)
	SetKernelArgWrapper("madd7", 8, fac4)
	SetKernelArgWrapper("madd7", 9, src5)
	SetKernelArgWrapper("madd7", 10, fac5)
	SetKernelArgWrapper("madd7", 11, src6)
	SetKernelArgWrapper("madd7", 12, fac6)
	SetKernelArgWrapper("madd7", 13, src7)
	SetKernelArgWrapper("madd7", 14, fac7)
	SetKernelArgWrapper("madd7", 15, N)

	//	args := madd7_args.argptr[:]
	event := LaunchKernel("madd7", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("madd7")
	}

	return event
}
