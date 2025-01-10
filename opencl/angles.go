package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

func SetPhi(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	if Synchronous {
		if err = cl.WaitForEvents(tmpEvents); err != nil {
			log.Printf("failed to wait for queue to finish in setphi: %+v \n", err)
		}
	}

	ClLastEvent = k_setPhi_async(s.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y),
		N[X], N[Y], N[Z],
		cfg, ClCmdQueue, tmpEvents)

	if err = ClCmdQueue.Flush(); err != nil {
		log.Printf("failed to flush queue in setphi: %+v \n", err)
	}

	if Synchronous {
		if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			log.Printf("failed to wait for queue to finish in setphi end: %+v \n", err)
		}
	}
}

func SetTheta(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	if Synchronous {
		if err = cl.WaitForEvents(tmpEvents); err != nil {
			log.Printf("failed to wait for queue to finish in settheta: %+v \n", err)
		}
	}

	ClLastEvent = k_setTheta_async(s.DevPtr(0), m.DevPtr(Z),
		N[X], N[Y], N[Z],
		cfg, ClCmdQueue, tmpEvents)

	if Synchronous {
		if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			log.Printf("failed to wait for queue to finish in settheta end: %+v \n", err)
		}
	}
}
