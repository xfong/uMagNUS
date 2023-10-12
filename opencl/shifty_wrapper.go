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

// Stores the arguments for shifty kernel invocation
type shifty_args_t struct {
	arg_dst    unsafe.Pointer
	arg_src    unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_shy    int
	arg_clampL float32
	arg_clampR float32
	argptr     [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for shifty kernel invocation
var shifty_args shifty_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	shifty_args.argptr[0] = unsafe.Pointer(&shifty_args.arg_dst)
	shifty_args.argptr[1] = unsafe.Pointer(&shifty_args.arg_src)
	shifty_args.argptr[2] = unsafe.Pointer(&shifty_args.arg_Nx)
	shifty_args.argptr[3] = unsafe.Pointer(&shifty_args.arg_Ny)
	shifty_args.argptr[4] = unsafe.Pointer(&shifty_args.arg_Nz)
	shifty_args.argptr[5] = unsafe.Pointer(&shifty_args.arg_shy)
	shifty_args.argptr[6] = unsafe.Pointer(&shifty_args.arg_clampL)
	shifty_args.argptr[7] = unsafe.Pointer(&shifty_args.arg_clampR)
}

// Wrapper for shifty OpenCL kernel, asynchronous.
func k_shifty_async(dst unsafe.Pointer, src unsafe.Pointer, Nx int, Ny int, Nz int, shy int, clampL float32, clampR float32, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("shifty")
	}

	shifty_args.Lock()
	defer shifty_args.Unlock()

	shifty_args.arg_dst = dst
	shifty_args.arg_src = src
	shifty_args.arg_Nx = Nx
	shifty_args.arg_Ny = Ny
	shifty_args.arg_Nz = Nz
	shifty_args.arg_shy = shy
	shifty_args.arg_clampL = clampL
	shifty_args.arg_clampR = clampR

	SetKernelArgWrapper("shifty", 0, dst)
	SetKernelArgWrapper("shifty", 1, src)
	SetKernelArgWrapper("shifty", 2, Nx)
	SetKernelArgWrapper("shifty", 3, Ny)
	SetKernelArgWrapper("shifty", 4, Nz)
	SetKernelArgWrapper("shifty", 5, shy)
	SetKernelArgWrapper("shifty", 6, clampL)
	SetKernelArgWrapper("shifty", 7, clampR)

	//	args := shifty_args.argptr[:]
	event := LaunchKernel("shifty", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("shifty")
	}

	return event
}
