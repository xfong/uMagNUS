package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// dst = sqrt(dot(a, a)),
func VecNorm(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in vecnorm: %+v \n", err)
		}
	}

	ClLastEvent = k_vecnorm_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		N, cfg, ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in vecnorm: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in vecnorm end: %+v \n", err)
		}
	}
}
