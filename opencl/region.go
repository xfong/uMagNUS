package opencl

import (
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// dst += LUT[region], for vectors. Used to add terms to excitation.
func RegionAddV(dst *data.Slice, lut LUTPtrs, regions *Bytes) {
	util.Argument(dst.NComp() == 3)
	N := dst.Len()
	cfg := make1DConf(N)

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	event := k_regionaddv_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		lut[X], lut[Y], lut[Z], regions.Ptr, N, cfg, evtWL)

	// set event markers
	log.Println("in egionaddv")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)
}

// dst += LUT[region], for scalar. Used to add terms to scalar excitation.
func RegionAddS(dst *data.Slice, lut LUTPtr, regions *Bytes) {
	util.Argument(dst.NComp() == 1)
	N := dst.Len()
	cfg := make1DConf(N)

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	event := k_regionadds_async(dst.DevPtr(0), unsafe.Pointer(lut), regions.Ptr, N, cfg,
		evtWL)

	// set event markers
	log.Println("in regionadds")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)
}

// decode the regions+LUT pair into an uncompressed array
func RegionDecode(dst *data.Slice, lut LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	event := k_regiondecode_async(dst.DevPtr(0), unsafe.Pointer(lut), regions.Ptr, N, cfg,
		evtWL)

	// set event markers
	log.Println("in regiondecode")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)
}

// select the part of src within the specified region, set 0's everywhere else.
func RegionSelect(dst, src *data.Slice, regions *Bytes, region byte) {
	util.Argument(dst.NComp() == src.NComp())
	N := dst.Len()
	cfg := make1DConf(N)

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	evtList := make([]*cl.Event, dst.NComp())
	for c := 0; c < dst.NComp(); c++ {
		event := k_regionselect_async(dst.DevPtr(c), src.DevPtr(c), regions.Ptr, region, N, cfg,
			evtWL)
		// set event markers
		evtList[c] = event
		log.Println("in regionselect")
		InsertEventToCmdSeqTail(event)
	}

	// set event markers
	UpdateLatestCmdList(evtList)
}
