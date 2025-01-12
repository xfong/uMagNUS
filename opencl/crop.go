package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// Crop stores in dst a rectangle cropped from src at given offset position.
// dst size may be smaller than src.
func Crop(dst, src *data.Slice, offX, offY, offZ int) {
	D := dst.Size()
	S := src.Size()
	util.Argument(dst.NComp() == src.NComp())
	util.Argument(D[X]+offX <= S[X] && D[Y]+offY <= S[Y] && D[Z]+offZ <= S[Z])

	cfg := make3DConf(D)

	var err error
	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in crop: %+v \n", err)
		}
	}

	// TODO: can overlap kernel execution using multiple queues (??)
	EmptyLastEvent()
	// assume each component of dst and src buffers are unique so there are no
	// dependencies between them
	for c := 0; c < dst.NComp(); c++ {
		tmpEvent := k_crop_async(dst.DevPtr(c), D[X], D[Y], D[Z],
			src.DevPtr(c), S[X], S[Y], S[Z],
			offX, offY, offZ, cfg, ClCmdQueue[0], tmpEvents)
		if err = ClCmdQueue[0].Flush(); err != nil {
			log.Printf("failed to flush queue in crop: %+v \n", err)
		}
		ClLastEvent = append(ClLastEvent, tmpEvent[0])
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in crop end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}
