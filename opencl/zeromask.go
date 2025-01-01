package opencl

import (
	"log"
	"unsafe"

	data "github.com/seeder-research/uMagNUS/data"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in zeromask: %+v \n", err)
		}
	}

	for c := 0; c < dst.NComp(); c++ {
		k_zeromask_async(dst.DevPtr(c), unsafe.Pointer(mask),
			regions.Ptr, N,
			cfg, ClCmdQueue, nil)
	}

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in zeromask end: %+v \n", err)
		}
	}
}
