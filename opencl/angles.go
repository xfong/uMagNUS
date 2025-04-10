package opencl

import (
	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

func SetPhi(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute and add event to sequence
	event := k_setPhi_async(s.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y),
		N[X], N[Y], N[Z],
		cfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)
}

func SetTheta(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute and add event to sequence
	event := k_setTheta_async(s.DevPtr(0), m.DevPtr(Z),
		N[X], N[Y], N[Z],
		cfg, evtWL)

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)
}
