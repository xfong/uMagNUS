package oclRAND

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

// Stores the arguments for mrg32k3a kernel invocation
type mrg32k3a_args_t struct {
	arg_seed_table unsafe.Pointer
	arg_randoms    unsafe.Pointer
	arg_N          uint32
	argptr         [3]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for mrg32k3a kernel invocation
var mrg32k3a_args mrg32k3a_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	mrg32k3a_args.argptr[0] = unsafe.Pointer(&mrg32k3a_args.arg_seed_table)
	mrg32k3a_args.argptr[1] = unsafe.Pointer(&mrg32k3a_args.arg_randoms)
	mrg32k3a_args.argptr[2] = unsafe.Pointer(&mrg32k3a_args.arg_N)
}

// Wrapper for mrg32k3a OpenCL kernel, asynchronous.
func k_mrg32k3a_async(seed_table unsafe.Pointer, randoms unsafe.Pointer, N uint32, cfg *config, events []*cl.Event) *cl.Event {
	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("mrg32k3a")
	}

	mrg32k3a_args.Lock()
	defer mrg32k3a_args.Unlock()

	mrg32k3a_args.arg_seed_table = seed_table
	mrg32k3a_args.arg_randoms = randoms
	mrg32k3a_args.arg_N = N

	SetKernelArgWrapper("mrg32k3a", 0, seed_table)
	SetKernelArgWrapper("mrg32k3a", 1, randoms)
	SetKernelArgWrapper("mrg32k3a", 2, N)

	//	args := mrg32k3a_args.argptr[:]
	event := LaunchKernel("mrg32k3a", cfg.Grid, cfg.Block, events)

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Stop("mrg32k3a")
	}

	return event
}
