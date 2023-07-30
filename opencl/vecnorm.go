package opencl

import (
	"fmt"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// dst = sqrt(dot(a, a)),
func VecNorm(dst *data.Slice, a *data.Slice, q *cl.CommandQueue, ewl []*cl.Event) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)

	event := k_vecnorm_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		N, cfg, ewl, q)

	dst.SetEvent(0, event)

	glist := []GSlice{a}
	InsertEventIntoGSlices(event, glist)

	if Synchronous || Debug {
		if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
			fmt.Printf("WaitForEvents failed in vecnorm: %+v \n", err)
		}
		WaitAndUpdateDataSliceEvents(event, glist, false)
	}

	return
}
