package opencl

import (
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	evtList := make([]*cl.Event, dst.NComp())
	for c := 0; c < dst.NComp(); c++ {
		event := k_zeromask_async(dst.DevPtr(c), unsafe.Pointer(mask),
			regions.Ptr, N,
			cfg, evtWL)
		// set event markers
		evtList[c] = event
		AddEventToSequence(event)
	}

	// set event markers
	UpdateLastEventList(evtList)
}
