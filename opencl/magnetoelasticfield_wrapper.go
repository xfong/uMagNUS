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

// Stores the arguments for addmagnetoelasticfield kernel invocation
type addmagnetoelasticfield_args_t struct {
	arg_Bx      unsafe.Pointer
	arg_By      unsafe.Pointer
	arg_Bz      unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_exx_    unsafe.Pointer
	arg_exx_mul float32
	arg_eyy_    unsafe.Pointer
	arg_eyy_mul float32
	arg_ezz_    unsafe.Pointer
	arg_ezz_mul float32
	arg_exy_    unsafe.Pointer
	arg_exy_mul float32
	arg_exz_    unsafe.Pointer
	arg_exz_mul float32
	arg_eyz_    unsafe.Pointer
	arg_eyz_mul float32
	arg_B1_     unsafe.Pointer
	arg_B1_mul  float32
	arg_B2_     unsafe.Pointer
	arg_B2_mul  float32
	arg_Ms_     unsafe.Pointer
	arg_Ms_mul  float32
	arg_N       int
	argptr      [25]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addmagnetoelasticfield kernel invocation
var addmagnetoelasticfield_args addmagnetoelasticfield_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	addmagnetoelasticfield_args.argptr[0] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Bx)
	addmagnetoelasticfield_args.argptr[1] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_By)
	addmagnetoelasticfield_args.argptr[2] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Bz)
	addmagnetoelasticfield_args.argptr[3] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_mx)
	addmagnetoelasticfield_args.argptr[4] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_my)
	addmagnetoelasticfield_args.argptr[5] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_mz)
	addmagnetoelasticfield_args.argptr[6] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exx_)
	addmagnetoelasticfield_args.argptr[7] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exx_mul)
	addmagnetoelasticfield_args.argptr[8] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyy_)
	addmagnetoelasticfield_args.argptr[9] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyy_mul)
	addmagnetoelasticfield_args.argptr[10] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_ezz_)
	addmagnetoelasticfield_args.argptr[11] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_ezz_mul)
	addmagnetoelasticfield_args.argptr[12] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exy_)
	addmagnetoelasticfield_args.argptr[13] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exy_mul)
	addmagnetoelasticfield_args.argptr[14] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exz_)
	addmagnetoelasticfield_args.argptr[15] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exz_mul)
	addmagnetoelasticfield_args.argptr[16] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyz_)
	addmagnetoelasticfield_args.argptr[17] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyz_mul)
	addmagnetoelasticfield_args.argptr[18] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B1_)
	addmagnetoelasticfield_args.argptr[19] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B1_mul)
	addmagnetoelasticfield_args.argptr[20] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B2_)
	addmagnetoelasticfield_args.argptr[21] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B2_mul)
	addmagnetoelasticfield_args.argptr[22] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Ms_)
	addmagnetoelasticfield_args.argptr[23] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Ms_mul)
	addmagnetoelasticfield_args.argptr[24] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_N)
}

// Wrapper for addmagnetoelasticfield OpenCL kernel, asynchronous.
func k_addmagnetoelasticfield_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, exx_ unsafe.Pointer, exx_mul float32, eyy_ unsafe.Pointer, eyy_mul float32, ezz_ unsafe.Pointer, ezz_mul float32, exy_ unsafe.Pointer, exy_mul float32, exz_ unsafe.Pointer, exz_mul float32, eyz_ unsafe.Pointer, eyz_mul float32, B1_ unsafe.Pointer, B1_mul float32, B2_ unsafe.Pointer, B2_mul float32, Ms_ unsafe.Pointer, Ms_mul float32, N int, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("addmagnetoelasticfield")
	}

	addmagnetoelasticfield_args.Lock()
	defer addmagnetoelasticfield_args.Unlock()

	addmagnetoelasticfield_args.arg_Bx = Bx
	addmagnetoelasticfield_args.arg_By = By
	addmagnetoelasticfield_args.arg_Bz = Bz
	addmagnetoelasticfield_args.arg_mx = mx
	addmagnetoelasticfield_args.arg_my = my
	addmagnetoelasticfield_args.arg_mz = mz
	addmagnetoelasticfield_args.arg_exx_ = exx_
	addmagnetoelasticfield_args.arg_exx_mul = exx_mul
	addmagnetoelasticfield_args.arg_eyy_ = eyy_
	addmagnetoelasticfield_args.arg_eyy_mul = eyy_mul
	addmagnetoelasticfield_args.arg_ezz_ = ezz_
	addmagnetoelasticfield_args.arg_ezz_mul = ezz_mul
	addmagnetoelasticfield_args.arg_exy_ = exy_
	addmagnetoelasticfield_args.arg_exy_mul = exy_mul
	addmagnetoelasticfield_args.arg_exz_ = exz_
	addmagnetoelasticfield_args.arg_exz_mul = exz_mul
	addmagnetoelasticfield_args.arg_eyz_ = eyz_
	addmagnetoelasticfield_args.arg_eyz_mul = eyz_mul
	addmagnetoelasticfield_args.arg_B1_ = B1_
	addmagnetoelasticfield_args.arg_B1_mul = B1_mul
	addmagnetoelasticfield_args.arg_B2_ = B2_
	addmagnetoelasticfield_args.arg_B2_mul = B2_mul
	addmagnetoelasticfield_args.arg_Ms_ = Ms_
	addmagnetoelasticfield_args.arg_Ms_mul = Ms_mul
	addmagnetoelasticfield_args.arg_N = N

	SetKernelArgWrapper("addmagnetoelasticfield", 0, Bx)
	SetKernelArgWrapper("addmagnetoelasticfield", 1, By)
	SetKernelArgWrapper("addmagnetoelasticfield", 2, Bz)
	SetKernelArgWrapper("addmagnetoelasticfield", 3, mx)
	SetKernelArgWrapper("addmagnetoelasticfield", 4, my)
	SetKernelArgWrapper("addmagnetoelasticfield", 5, mz)
	SetKernelArgWrapper("addmagnetoelasticfield", 6, exx_)
	SetKernelArgWrapper("addmagnetoelasticfield", 7, exx_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 8, eyy_)
	SetKernelArgWrapper("addmagnetoelasticfield", 9, eyy_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 10, ezz_)
	SetKernelArgWrapper("addmagnetoelasticfield", 11, ezz_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 12, exy_)
	SetKernelArgWrapper("addmagnetoelasticfield", 13, exy_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 14, exz_)
	SetKernelArgWrapper("addmagnetoelasticfield", 15, exz_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 16, eyz_)
	SetKernelArgWrapper("addmagnetoelasticfield", 17, eyz_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 18, B1_)
	SetKernelArgWrapper("addmagnetoelasticfield", 19, B1_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 20, B2_)
	SetKernelArgWrapper("addmagnetoelasticfield", 21, B2_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 22, Ms_)
	SetKernelArgWrapper("addmagnetoelasticfield", 23, Ms_mul)
	SetKernelArgWrapper("addmagnetoelasticfield", 24, N)

	//	args := addmagnetoelasticfield_args.argptr[:]
	event := LaunchKernel("addmagnetoelasticfield", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("addmagnetoelasticfield")
	}

	return event
}
