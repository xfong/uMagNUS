package opencl

import (
	"log"
	"unsafe"

	data "github.com/seeder-research/uMagNUS/data"
)

// Add Slonczewski ST torque to torque (Tesla).
func AddOommfSlonczewskiTorque(torque, m *data.Slice, Msat, J, fixedP, alpha, pfix, pfree, λfix, λfree, ε_prime MSlice, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	flt := float32(mesh.WorldSize()[Z])

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addoommfslonczewskitorque: %+v \n", err)
		}
	}

	ClLastEvent = k_addoommfslonczewskitorque_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(Z), J.Mul(Z),
		fixedP.DevPtr(X), fixedP.Mul(X),
		fixedP.DevPtr(Y), fixedP.Mul(Y),
		fixedP.DevPtr(Z), fixedP.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		pfix.DevPtr(0), pfix.Mul(0),
		pfree.DevPtr(0), pfree.Mul(0),
		λfix.DevPtr(0), λfix.Mul(0),
		λfree.DevPtr(0), λfree.Mul(0),
		ε_prime.DevPtr(0), ε_prime.Mul(0),
		unsafe.Pointer(uintptr(0)), flt,
		N, cfg, ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in addoommfslonczewskitorque: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in addoommfslonczewskitorque end: %+v \n", err)
		}
	}
}
