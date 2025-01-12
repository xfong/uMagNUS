package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// divide: dst[i] = a[i] / b[i]
// divide by zero automagically returns 0.0
func Divide(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in divide: %+v \n", err)
		}
	}

	// TODO: create multiple queues(??)
	EmptyLastEvent()
	for c := 0; c < nComp; c++ {
		tmpEvent := k_divide_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg,
			ClCmdQueue[0], tmpEvents)
		if err = ClCmdQueue[0].Flush(); err != nil {
			log.Printf("failed to flush queue in divide: %+v \n", err)
		}
		ClLastEvent = append(ClLastEvent, tmpEvent[0])
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in divide end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}
