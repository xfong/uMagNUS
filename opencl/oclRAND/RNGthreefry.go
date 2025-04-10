package oclRAND

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	timer "github.com/seeder-research/uMagNUS/timer"
	"math/rand"
)

func (p *THREEFRY_status_array_ptr) Init(seed uint64, events []*cl.Event) *cl.Event {
	var err error
	var seed_buf *cl.MemObject
	var seed_event *cl.Event
	var event *cl.Event
	var queue *cl.CommandQueue

	// Generate random seed array to seed the PRNG
	rand.Seed((int64)(seed))
	totalCount := p.GetStatusSize()
	seed_arr := make([]uint32, totalCount)
	for idx := 0; idx < totalCount; idx++ {
		tmpNum := rand.Uint32()
		for tmpNum == 0 {
			tmpNum = rand.Uint32()
		}
		seed_arr[idx] = tmpNum
	}

	// copy random seed array to GPU
	context := p.GetContext()
	seed_buf, err = context.CreateBufferUnsafe(cl.MemReadWrite, int(unsafe.Sizeof(seed_arr[0]))*totalCount, nil)
	defer seed_buf.Release()
	if err != nil {
		log.Fatalln("failed to create buffer for threefry seed array!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("failed to wait for last marker in beginning of threefry.init: %+v \n", err)
		}
		timer.Start("threefry_init")
	}

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in threefry.init: %+v \n", err)
	}
	if seed_event, err = queue.EnqueueWriteBuffer(seed_buf, false, 0, int(unsafe.Sizeof(seed_arr[0]))*totalCount, unsafe.Pointer(&seed_arr[0]), events); err != nil {
		log.Fatalln("failed to write seed buffer to device in threefry.init: %+v \n", err)
	}

	queue.Release() // implicit flush

	// seed the RNG
	event = k_threefry_seed_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), unsafe.Pointer(seed_buf),
		&config{[]int{totalCount}, []int{p.GetGroupSize()}}, []*cl.Event{seed_event})

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for last marker in threefry.init: %+v \n", err)
		}
		timer.Stop("threefry_init")
	}

	p.Ini = true

	return event
}

func (p *THREEFRY_status_array_ptr) GenerateUniform(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {
	var err error

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("failed to wait for last marker in beginning of threefry.generateuniform: %+v \n", err)
		}
		timer.Start("threefry_uniform")
	}

	// execute
	event := k_threefry_uniform_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, events)

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for last marker at end of threefry.generateuniform: %+v \n", err)
		}
		timer.Stop("threefry_uniform")
	}

	return event
}

func (p *THREEFRY_status_array_ptr) GenerateNormal(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {
	var err error

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("wait for events in threefry.generatenormal failed: %+v \n", err)
		}
		timer.Start("threefry_normal")
	}

	// execute
	event := k_threefry_normal_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, events)

	if Synchronous { // debug
		if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for event in threefry.generatenormal: %+v \n", err)
		}
		timer.Stop("threefry_normal")
	}

	return event
}
