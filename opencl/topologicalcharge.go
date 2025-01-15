package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// Set s to the toplogogical charge density s = m · (∂m/∂x ❌ ∂m/∂y)
// see topologicalcharge.cl
func SetTopologicalCharge(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	icxcy := float32(1.0 / (cellsize[X] * cellsize[Y]))

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in settopologicalcharge: %+v \n", err)
		}
	}

	ClLastEvent = k_settopologicalcharge_async(s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		icxcy, N[X], N[Y], N[Z], mesh.PBC_code(), cfg,
		ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in settopologicalcharge: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in settopologicalcharge end: %+v \n", err)
		}
	}
}
