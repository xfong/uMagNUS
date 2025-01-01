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
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in crop: %+v \n", err)
		}
	}

	for c := 0; c < dst.NComp(); c++ {
		k_crop_async(dst.DevPtr(c), D[X], D[Y], D[Z],
			src.DevPtr(c), S[X], S[Y], S[Z],
			offX, offY, offZ, cfg, ClCmdQueue, nil)
	}

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in crop end: %+v \n", err)
		}
	}
}
