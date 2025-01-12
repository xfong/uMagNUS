package opencl

// Region paired spin torque calculations

import (
	"log"
	"math"

	data "github.com/seeder-research/uMagNUS/data"
)

func AddRegionSpinTorque(torque, m *data.Slice, Msat MSlice, regions *Bytes, regionA, regionB uint8, sX, sY, sZ int, J, alpha, pfix, pfree, λfix, λfree, ε_prime float32, mesh *data.Mesh) {
	c := mesh.CellSize()
	dX := float64(sX) * c[X]
	dY := float64(sY) * c[Y]
	dZ := float64(sZ) * c[Z]

	distsq := dX*dX + dY*dY + dZ*dZ
	cellwgt := math.Abs(dX*c[X]) + math.Abs(dY*c[Y]) + math.Abs(dZ*c[Z])
	if cellwgt > 0.0 {
		cellwgt = math.Sqrt(distsq) / cellwgt
	}

	N := mesh.Size()
	cfg := make3DConf(N)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionspintorque: %+v \n", err)
		}
	}

	ClLastEvent = k_addtworegionoommfslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, regionA, regionB,
		sX, sY, sZ, N[X], N[Y], N[Z],
		J, alpha, pfix, pfree, λfix, λfree, ε_prime, float32(cellwgt),
		cfg, ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in addregionspintorque: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionspintorque end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}
