package opencl

// Region paired exchange interaction

import (
	"log"
	"math"

	data "github.com/seeder-research/uMagNUS/data"
)

// Add exchange field to Beff.
//
//	m: normalized magnetization
//	B: effective field in Tesla
func AddRegionExchangeField(B, m *data.Slice, Msat MSlice, regions *Bytes, regionA, regionB uint8, sX, sY, sZ int, sig, sig2 float32, mesh *data.Mesh) {
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

	sig_eff := sig * float32(cellwgt)
	sig2_eff := sig2 * float32(cellwgt)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionexchangefield: %+v \n", err)
		}
	}

	ClLastEvent = k_tworegionexchange_field_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, regionA, regionB,
		sX, sY, sZ, sig_eff, sig2_eff, N[X], N[Y], N[Z], cfg,
		ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("error flushing queue in addregionexchangefield: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionexchangefield end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}

func AddRegionExchangeEdens(Edens, m *data.Slice, Msat MSlice, regions *Bytes, regionA, regionB uint8, sX, sY, sZ int, sig, sig2 float32, mesh *data.Mesh) {
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

	sig_eff := sig * float32(cellwgt)
	sig2_eff := sig2 * float32(cellwgt)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionexchangeedens: %+v \n", err)
		}
	}

	ClLastEvent = k_tworegionexchange_edens_async(Edens.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, regionA, regionB,
		sX, sY, sZ, sig_eff, sig2_eff, N[X], N[Y], N[Z], cfg,
		ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("error flushing queue in addregionexchangeedens: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addregionexchangeedens end: %+v \n", err)
		}
		EmptyLastEvent()
	}
}
