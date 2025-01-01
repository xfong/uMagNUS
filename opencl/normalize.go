package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// Normalize vec to unit length, unless length or vol are zero.
func Normalize(vec, vol *data.Slice) {
	util.Argument(vol == nil || vol.NComp() == 1)
	N := vec.Len()
	cfg := make1DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in normalize: %+v \n", err)
		}
	}

	k_normalize2_async(vec.DevPtr(X), vec.DevPtr(Y), vec.DevPtr(Z),
		vol.DevPtr(0), N, cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in normalize end: %+v \n", err)
		}
	}
}
