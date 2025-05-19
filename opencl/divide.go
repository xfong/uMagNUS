package opencl

import (
	cl "github.com/seeder-research/uMagNUS/cl"
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

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// execute
	evtList := make([]*cl.Event, 3)
	for c := 0; c < nComp; c++ {
		event := k_divide_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg,
			evtWL)
		// set event markers
		InsertEventToCmdSeqTail(event)
		evtList[c] = event
	}

	// set event markers
	UpdateLatestCmdList(evtList)
}
