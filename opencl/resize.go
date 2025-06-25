package opencl

import (
	"log"
	"fmt"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// Select and resize one layer for interactive output
func Resize(dst, src *data.Slice, layer int) {
	dstsize := dst.Size()
	srcsize := src.Size()
	util.Assert(dstsize[Z] == 1)
	util.Assert(dst.NComp() == 1 && src.NComp() == 1)

	scalex := srcsize[X] / dstsize[X]
	scaley := srcsize[Y] / dstsize[Y]
	util.Assert(scalex > 0 && scaley > 0)

	cfg := make3DConf(dstsize)

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	event := k_resize_async(dst.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z],
		src.DevPtr(0), srcsize[X], srcsize[Y], srcsize[Z], layer, scalex, scaley, cfg,
		evtWL)

	// set event markers
	log.Println("in resize")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous {
		if err := WaitLatestCmd(); err != nil {
			fmt.Printf("wait for last event failed in resize: %+v \n", err)
		}
	}
}
