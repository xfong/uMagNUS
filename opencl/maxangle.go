package opencl

import (
	"log"
	"unsafe"

	data "github.com/seeder-research/uMagNUS/data"
)

// SetMaxAngle sets dst to the maximum angle of each cells magnetization with all of its neighbors,
// provided the exchange stiffness with that neighbor is nonzero.
func SetMaxAngle(dst, m *data.Slice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in setmaxangle: %+v \n", err)
		}
	}

	ClLastEvent = k_setmaxangle_async(dst.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex_red), regions.Ptr,
		N[X], N[Y], N[Z], pbc, cfg, ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in setmaxangle: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in setmaxangle end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}
