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

// Stores the arguments for adduniaxialanisotropy2 kernel invocation
type adduniaxialanisotropy2_args_t struct {
	arg_Bx     unsafe.Pointer
	arg_By     unsafe.Pointer
	arg_Bz     unsafe.Pointer
	arg_mx     unsafe.Pointer
	arg_my     unsafe.Pointer
	arg_mz     unsafe.Pointer
	arg_Ms_    unsafe.Pointer
	arg_Ms_mul float32
	arg_K1_    unsafe.Pointer
	arg_K1_mul float32
	arg_K2_    unsafe.Pointer
	arg_K2_mul float32
	arg_ux_    unsafe.Pointer
	arg_ux_mul float32
	arg_uy_    unsafe.Pointer
	arg_uy_mul float32
	arg_uz_    unsafe.Pointer
	arg_uz_mul float32
	arg_N      int
	argptr     [19]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for adduniaxialanisotropy2 kernel invocation
var adduniaxialanisotropy2_args adduniaxialanisotropy2_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	adduniaxialanisotropy2_args.argptr[0] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_Bx)
	adduniaxialanisotropy2_args.argptr[1] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_By)
	adduniaxialanisotropy2_args.argptr[2] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_Bz)
	adduniaxialanisotropy2_args.argptr[3] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_mx)
	adduniaxialanisotropy2_args.argptr[4] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_my)
	adduniaxialanisotropy2_args.argptr[5] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_mz)
	adduniaxialanisotropy2_args.argptr[6] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_Ms_)
	adduniaxialanisotropy2_args.argptr[7] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_Ms_mul)
	adduniaxialanisotropy2_args.argptr[8] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_K1_)
	adduniaxialanisotropy2_args.argptr[9] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_K1_mul)
	adduniaxialanisotropy2_args.argptr[10] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_K2_)
	adduniaxialanisotropy2_args.argptr[11] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_K2_mul)
	adduniaxialanisotropy2_args.argptr[12] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_ux_)
	adduniaxialanisotropy2_args.argptr[13] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_ux_mul)
	adduniaxialanisotropy2_args.argptr[14] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_uy_)
	adduniaxialanisotropy2_args.argptr[15] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_uy_mul)
	adduniaxialanisotropy2_args.argptr[16] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_uz_)
	adduniaxialanisotropy2_args.argptr[17] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_uz_mul)
	adduniaxialanisotropy2_args.argptr[18] = unsafe.Pointer(&adduniaxialanisotropy2_args.arg_N)
}

// Wrapper for adduniaxialanisotropy2 OpenCL kernel, asynchronous.
func k_adduniaxialanisotropy2_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, K1_ unsafe.Pointer, K1_mul float32, K2_ unsafe.Pointer, K2_mul float32, ux_ unsafe.Pointer, ux_mul float32, uy_ unsafe.Pointer, uy_mul float32, uz_ unsafe.Pointer, uz_mul float32, N int, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("adduniaxialanisotropy2")
	}

	adduniaxialanisotropy2_args.Lock()
	defer adduniaxialanisotropy2_args.Unlock()

	adduniaxialanisotropy2_args.arg_Bx = Bx
	adduniaxialanisotropy2_args.arg_By = By
	adduniaxialanisotropy2_args.arg_Bz = Bz
	adduniaxialanisotropy2_args.arg_mx = mx
	adduniaxialanisotropy2_args.arg_my = my
	adduniaxialanisotropy2_args.arg_mz = mz
	adduniaxialanisotropy2_args.arg_Ms_ = Ms_
	adduniaxialanisotropy2_args.arg_Ms_mul = Ms_mul
	adduniaxialanisotropy2_args.arg_K1_ = K1_
	adduniaxialanisotropy2_args.arg_K1_mul = K1_mul
	adduniaxialanisotropy2_args.arg_K2_ = K2_
	adduniaxialanisotropy2_args.arg_K2_mul = K2_mul
	adduniaxialanisotropy2_args.arg_ux_ = ux_
	adduniaxialanisotropy2_args.arg_ux_mul = ux_mul
	adduniaxialanisotropy2_args.arg_uy_ = uy_
	adduniaxialanisotropy2_args.arg_uy_mul = uy_mul
	adduniaxialanisotropy2_args.arg_uz_ = uz_
	adduniaxialanisotropy2_args.arg_uz_mul = uz_mul
	adduniaxialanisotropy2_args.arg_N = N

	SetKernelArgWrapper("adduniaxialanisotropy2", 0, Bx)
	SetKernelArgWrapper("adduniaxialanisotropy2", 1, By)
	SetKernelArgWrapper("adduniaxialanisotropy2", 2, Bz)
	SetKernelArgWrapper("adduniaxialanisotropy2", 3, mx)
	SetKernelArgWrapper("adduniaxialanisotropy2", 4, my)
	SetKernelArgWrapper("adduniaxialanisotropy2", 5, mz)
	SetKernelArgWrapper("adduniaxialanisotropy2", 6, Ms_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 7, Ms_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 8, K1_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 9, K1_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 10, K2_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 11, K2_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 12, ux_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 13, ux_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 14, uy_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 15, uy_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 16, uz_)
	SetKernelArgWrapper("adduniaxialanisotropy2", 17, uz_mul)
	SetKernelArgWrapper("adduniaxialanisotropy2", 18, N)

	//	args := adduniaxialanisotropy2_args.argptr[:]
	event := LaunchKernel("adduniaxialanisotropy2", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("adduniaxialanisotropy2")
	}

	return event
}
