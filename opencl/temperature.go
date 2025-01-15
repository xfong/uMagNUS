package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperature(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Msat, Temp, Alpha MSlice) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	var err error

	tmpEvents := LastEventToWaitList()

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in settemperature2: %+v \n", err)
		}
	}

	ClLastEvent = k_settemperature2_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Msat.DevPtr(0), Msat.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		N, cfg,
		ClCmdQueue[0], tmpEvents)

	if err = ClCmdQueue[0].Flush(); err != nil {
		log.Printf("failed to flush queue in settemperature2: %+v \n", err)
	}

	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Printf("failed to wait for queue to finish in settemperature2 end: %+v \n", err)
		}
	}
}
