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

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in zeromask: %+v \n", err)
		}
	}

	for c := 0; c < dst.NComp(); c++ {
		ClLastEvent = k_zeromask_async(dst.DevPtr(c), unsafe.Pointer(mask),
			regions.Ptr, N,
			cfg, ClCmdQueue[0], tmpEvents)
		if err = ClCmdQueue[0].Flush(); err != nil {
			log.Printf("failed to flush queue in zeromask: %+v \n", err)
		}
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in zeromask end: %+v \n", err)
		}
	}
}
