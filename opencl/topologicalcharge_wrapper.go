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

// Stores the arguments for settopologicalcharge kernel invocation
type settopologicalcharge_args_t struct {
	arg_s     unsafe.Pointer
	arg_mx    unsafe.Pointer
	arg_my    unsafe.Pointer
	arg_mz    unsafe.Pointer
	arg_icxcy float32
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	arg_PBC   uint8
	argptr    [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for settopologicalcharge kernel invocation
var settopologicalcharge_args settopologicalcharge_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	settopologicalcharge_args.argptr[0] = unsafe.Pointer(&settopologicalcharge_args.arg_s)
	settopologicalcharge_args.argptr[1] = unsafe.Pointer(&settopologicalcharge_args.arg_mx)
	settopologicalcharge_args.argptr[2] = unsafe.Pointer(&settopologicalcharge_args.arg_my)
	settopologicalcharge_args.argptr[3] = unsafe.Pointer(&settopologicalcharge_args.arg_mz)
	settopologicalcharge_args.argptr[4] = unsafe.Pointer(&settopologicalcharge_args.arg_icxcy)
	settopologicalcharge_args.argptr[5] = unsafe.Pointer(&settopologicalcharge_args.arg_Nx)
	settopologicalcharge_args.argptr[6] = unsafe.Pointer(&settopologicalcharge_args.arg_Ny)
	settopologicalcharge_args.argptr[7] = unsafe.Pointer(&settopologicalcharge_args.arg_Nz)
	settopologicalcharge_args.argptr[8] = unsafe.Pointer(&settopologicalcharge_args.arg_PBC)
}

// Wrapper for settopologicalcharge OpenCL kernel, asynchronous.
func k_settopologicalcharge_async(s unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, icxcy float32, Nx int, Ny int, Nz int, PBC uint8, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("settopologicalcharge")
	}

	settopologicalcharge_args.Lock()
	defer settopologicalcharge_args.Unlock()

	settopologicalcharge_args.arg_s = s
	settopologicalcharge_args.arg_mx = mx
	settopologicalcharge_args.arg_my = my
	settopologicalcharge_args.arg_mz = mz
	settopologicalcharge_args.arg_icxcy = icxcy
	settopologicalcharge_args.arg_Nx = Nx
	settopologicalcharge_args.arg_Ny = Ny
	settopologicalcharge_args.arg_Nz = Nz
	settopologicalcharge_args.arg_PBC = PBC

	SetKernelArgWrapper("settopologicalcharge", 0, s)
	SetKernelArgWrapper("settopologicalcharge", 1, mx)
	SetKernelArgWrapper("settopologicalcharge", 2, my)
	SetKernelArgWrapper("settopologicalcharge", 3, mz)
	SetKernelArgWrapper("settopologicalcharge", 4, icxcy)
	SetKernelArgWrapper("settopologicalcharge", 5, Nx)
	SetKernelArgWrapper("settopologicalcharge", 6, Ny)
	SetKernelArgWrapper("settopologicalcharge", 7, Nz)
	SetKernelArgWrapper("settopologicalcharge", 8, PBC)

	//	args := settopologicalcharge_args.argptr[:]
	event := LaunchKernel("settopologicalcharge", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("settopologicalcharge")
	}

	return event
}
