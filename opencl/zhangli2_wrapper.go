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

// Stores the arguments for addzhanglitorque2 kernel invocation
type addzhanglitorque2_args_t struct {
	arg_tx        unsafe.Pointer
	arg_ty        unsafe.Pointer
	arg_tz        unsafe.Pointer
	arg_mx        unsafe.Pointer
	arg_my        unsafe.Pointer
	arg_mz        unsafe.Pointer
	arg_Ms_       unsafe.Pointer
	arg_Ms_mul    float32
	arg_jx_       unsafe.Pointer
	arg_jx_mul    float32
	arg_jy_       unsafe.Pointer
	arg_jy_mul    float32
	arg_jz_       unsafe.Pointer
	arg_jz_mul    float32
	arg_alpha_    unsafe.Pointer
	arg_alpha_mul float32
	arg_xi_       unsafe.Pointer
	arg_xi_mul    float32
	arg_pol_      unsafe.Pointer
	arg_pol_mul   float32
	arg_cx        float32
	arg_cy        float32
	arg_cz        float32
	arg_Nx        int
	arg_Ny        int
	arg_Nz        int
	arg_PBC       uint8
	argptr        [27]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addzhanglitorque2 kernel invocation
var addzhanglitorque2_args addzhanglitorque2_args_t

func init() {
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	addzhanglitorque2_args.argptr[0] = unsafe.Pointer(&addzhanglitorque2_args.arg_tx)
	addzhanglitorque2_args.argptr[1] = unsafe.Pointer(&addzhanglitorque2_args.arg_ty)
	addzhanglitorque2_args.argptr[2] = unsafe.Pointer(&addzhanglitorque2_args.arg_tz)
	addzhanglitorque2_args.argptr[3] = unsafe.Pointer(&addzhanglitorque2_args.arg_mx)
	addzhanglitorque2_args.argptr[4] = unsafe.Pointer(&addzhanglitorque2_args.arg_my)
	addzhanglitorque2_args.argptr[5] = unsafe.Pointer(&addzhanglitorque2_args.arg_mz)
	addzhanglitorque2_args.argptr[6] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ms_)
	addzhanglitorque2_args.argptr[7] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ms_mul)
	addzhanglitorque2_args.argptr[8] = unsafe.Pointer(&addzhanglitorque2_args.arg_jx_)
	addzhanglitorque2_args.argptr[9] = unsafe.Pointer(&addzhanglitorque2_args.arg_jx_mul)
	addzhanglitorque2_args.argptr[10] = unsafe.Pointer(&addzhanglitorque2_args.arg_jy_)
	addzhanglitorque2_args.argptr[11] = unsafe.Pointer(&addzhanglitorque2_args.arg_jy_mul)
	addzhanglitorque2_args.argptr[12] = unsafe.Pointer(&addzhanglitorque2_args.arg_jz_)
	addzhanglitorque2_args.argptr[13] = unsafe.Pointer(&addzhanglitorque2_args.arg_jz_mul)
	addzhanglitorque2_args.argptr[14] = unsafe.Pointer(&addzhanglitorque2_args.arg_alpha_)
	addzhanglitorque2_args.argptr[15] = unsafe.Pointer(&addzhanglitorque2_args.arg_alpha_mul)
	addzhanglitorque2_args.argptr[16] = unsafe.Pointer(&addzhanglitorque2_args.arg_xi_)
	addzhanglitorque2_args.argptr[17] = unsafe.Pointer(&addzhanglitorque2_args.arg_xi_mul)
	addzhanglitorque2_args.argptr[18] = unsafe.Pointer(&addzhanglitorque2_args.arg_pol_)
	addzhanglitorque2_args.argptr[19] = unsafe.Pointer(&addzhanglitorque2_args.arg_pol_mul)
	addzhanglitorque2_args.argptr[20] = unsafe.Pointer(&addzhanglitorque2_args.arg_cx)
	addzhanglitorque2_args.argptr[21] = unsafe.Pointer(&addzhanglitorque2_args.arg_cy)
	addzhanglitorque2_args.argptr[22] = unsafe.Pointer(&addzhanglitorque2_args.arg_cz)
	addzhanglitorque2_args.argptr[23] = unsafe.Pointer(&addzhanglitorque2_args.arg_Nx)
	addzhanglitorque2_args.argptr[24] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ny)
	addzhanglitorque2_args.argptr[25] = unsafe.Pointer(&addzhanglitorque2_args.arg_Nz)
	addzhanglitorque2_args.argptr[26] = unsafe.Pointer(&addzhanglitorque2_args.arg_PBC)
}

// Wrapper for addzhanglitorque2 OpenCL kernel, asynchronous.
func k_addzhanglitorque2_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, jx_ unsafe.Pointer, jx_mul float32, jy_ unsafe.Pointer, jy_mul float32, jz_ unsafe.Pointer, jz_mul float32, alpha_ unsafe.Pointer, alpha_mul float32, xi_ unsafe.Pointer, xi_mul float32, pol_ unsafe.Pointer, pol_mul float32, cx float32, cy float32, cz float32, Nx int, Ny int, Nz int, PBC uint8, cfg *config, events []*cl.Event, launchQueue *cl.CommandQueue) *cl.Event {
	if Synchronous { // debug
		launchQueue.Finish()
		timer.Start("addzhanglitorque2")
	}

	addzhanglitorque2_args.Lock()
	defer addzhanglitorque2_args.Unlock()

	addzhanglitorque2_args.arg_tx = tx
	addzhanglitorque2_args.arg_ty = ty
	addzhanglitorque2_args.arg_tz = tz
	addzhanglitorque2_args.arg_mx = mx
	addzhanglitorque2_args.arg_my = my
	addzhanglitorque2_args.arg_mz = mz
	addzhanglitorque2_args.arg_Ms_ = Ms_
	addzhanglitorque2_args.arg_Ms_mul = Ms_mul
	addzhanglitorque2_args.arg_jx_ = jx_
	addzhanglitorque2_args.arg_jx_mul = jx_mul
	addzhanglitorque2_args.arg_jy_ = jy_
	addzhanglitorque2_args.arg_jy_mul = jy_mul
	addzhanglitorque2_args.arg_jz_ = jz_
	addzhanglitorque2_args.arg_jz_mul = jz_mul
	addzhanglitorque2_args.arg_alpha_ = alpha_
	addzhanglitorque2_args.arg_alpha_mul = alpha_mul
	addzhanglitorque2_args.arg_xi_ = xi_
	addzhanglitorque2_args.arg_xi_mul = xi_mul
	addzhanglitorque2_args.arg_pol_ = pol_
	addzhanglitorque2_args.arg_pol_mul = pol_mul
	addzhanglitorque2_args.arg_cx = cx
	addzhanglitorque2_args.arg_cy = cy
	addzhanglitorque2_args.arg_cz = cz
	addzhanglitorque2_args.arg_Nx = Nx
	addzhanglitorque2_args.arg_Ny = Ny
	addzhanglitorque2_args.arg_Nz = Nz
	addzhanglitorque2_args.arg_PBC = PBC

	SetKernelArgWrapper("addzhanglitorque2", 0, tx)
	SetKernelArgWrapper("addzhanglitorque2", 1, ty)
	SetKernelArgWrapper("addzhanglitorque2", 2, tz)
	SetKernelArgWrapper("addzhanglitorque2", 3, mx)
	SetKernelArgWrapper("addzhanglitorque2", 4, my)
	SetKernelArgWrapper("addzhanglitorque2", 5, mz)
	SetKernelArgWrapper("addzhanglitorque2", 6, Ms_)
	SetKernelArgWrapper("addzhanglitorque2", 7, Ms_mul)
	SetKernelArgWrapper("addzhanglitorque2", 8, jx_)
	SetKernelArgWrapper("addzhanglitorque2", 9, jx_mul)
	SetKernelArgWrapper("addzhanglitorque2", 10, jy_)
	SetKernelArgWrapper("addzhanglitorque2", 11, jy_mul)
	SetKernelArgWrapper("addzhanglitorque2", 12, jz_)
	SetKernelArgWrapper("addzhanglitorque2", 13, jz_mul)
	SetKernelArgWrapper("addzhanglitorque2", 14, alpha_)
	SetKernelArgWrapper("addzhanglitorque2", 15, alpha_mul)
	SetKernelArgWrapper("addzhanglitorque2", 16, xi_)
	SetKernelArgWrapper("addzhanglitorque2", 17, xi_mul)
	SetKernelArgWrapper("addzhanglitorque2", 18, pol_)
	SetKernelArgWrapper("addzhanglitorque2", 19, pol_mul)
	SetKernelArgWrapper("addzhanglitorque2", 20, cx)
	SetKernelArgWrapper("addzhanglitorque2", 21, cy)
	SetKernelArgWrapper("addzhanglitorque2", 22, cz)
	SetKernelArgWrapper("addzhanglitorque2", 23, Nx)
	SetKernelArgWrapper("addzhanglitorque2", 24, Ny)
	SetKernelArgWrapper("addzhanglitorque2", 25, Nz)
	SetKernelArgWrapper("addzhanglitorque2", 26, PBC)

	//	args := addzhanglitorque2_args.argptr[:]
	event := LaunchKernel("addzhanglitorque2", cfg.Grid, cfg.Block, launchQueue, events)

	if Synchronous { // debug
		launchQueue.Finish()
		timer.Stop("addzhanglitorque2")
	}

	return event
}
