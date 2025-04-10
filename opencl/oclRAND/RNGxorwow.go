package oclRAND

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	timer "github.com/seeder-research/uMagNUS/timer"
)

func (p *XORWOW_status_array_ptr) Init(seed uint64, events []*cl.Event) *cl.Event {
	var err error
	var queue *cl.CommandQueue
	var jump_mat *cl.MemObject
	var event *cl.Event

	context := p.GetContext()
	totalCount := p.GetStatusSize()

	// Set up jump matrices on GPU...
	//    ....Creating device buffer to hold matrices
	jump_mat, err = context.CreateBufferUnsafe(cl.MemReadWrite, int(unsafe.Sizeof(h_xorwow_sequence_jump_matrices[0][0]))*int(XORWOW_SIZE*XORWOW_JUMP_MATRICES), nil)
	defer jump_mat.Release()
	if err != nil {
		log.Fatalln("Unable to create buffer for XORWOW jump matrices array!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("failed to wait for last marker in beginning of xorwow.init: %+v \n", err)
		}
		timer.Start("xorwow_init")
	}

	//    ....Copying jump matrices from host side to device side
	jump_events := make([]*cl.Event, int(XORWOW_JUMP_MATRICES))
	for idx := 0; idx < int(XORWOW_JUMP_MATRICES); idx++ {
		// create command queue and execute
		if queue, err = CreateCommandQueue(); err != nil {
			log.Panicf("failed to create command queue in xorwow.init: %+v \n", err)
		}
		if jump_events[idx], err = queue.EnqueueWriteBuffer(jump_mat, false, idx*int(XORWOW_SIZE)*int(unsafe.Sizeof(h_xorwow_sequence_jump_matrices[0][0])), int(unsafe.Sizeof(h_xorwow_sequence_jump_matrices[0][0]))*int(XORWOW_SIZE), unsafe.Pointer(&h_xorwow_sequence_jump_matrices[idx][0]), nil); err != nil {
			log.Fatalln("Unable to write jump matrices to device: ", err)
		}

		queue.Release() // implicit flush
	}

	// Seed the RNG
	var seed_events []*cl.Event
	seed_events = jump_events
	if events != nil {
		seed_events = append(events)
	}
	event = k_xorwow_seed_async(unsafe.Pointer(p.Status_buf), unsafe.Pointer(jump_mat), seed, &config{[]int{totalCount}, []int{p.GetGroupSize()}}, seed_events)

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for last marker in xorwow.init: %+v \n", err)
		}
		timer.Stop("xorwow_init")
	}

	p.Ini = true

	return event
}

func (p *XORWOW_status_array_ptr) GenerateUniform(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {
	var err error
	var event *cl.Event

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("failed to wait for last marker in beginning of xorwow.generateuniform: %+v \n", err)
		}
		timer.Start("xorwow_uniform")
	}

	event = k_xorwow_uniform_async(unsafe.Pointer(p.Status_buf), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, events)

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for last marker in xorwow.generateuniform: %+v \n", err)
		}
		timer.Stop("xorwow_uniform")
	}

	return event
}

func (p *XORWOW_status_array_ptr) GenerateNormal(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {
	var err error
	var event *cl.Event

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("failed to wait for last marker in beginning of xorwow.generatenormal: %+v \n", err)
		}
		timer.Start("xorwow_normal")
	}

	// execute
	event = k_xorwow_normal_async(unsafe.Pointer(p.Status_buf), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, events)

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for last marker in xorwow.generatenormal: %+v \n", err)
		}
		timer.Stop("xorwow_normal")
	}

	return event
}
