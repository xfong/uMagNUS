package oclRAND

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	timer "github.com/seeder-research/uMagNUS/timer"
	"math/rand"
)

func (p *THREEFRY_status_array_ptr) Init(seed uint64, queue *cl.CommandQueue, events []*cl.Event) *cl.Event {
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
	seed_buf, err := context.CreateBufferUnsafe(cl.MemReadWrite, int(unsafe.Sizeof(seed_arr[0]))*totalCount, nil)
	defer seed_buf.Release()
	if err != nil {
		log.Fatalln("failed to create buffer for threefry seed array!")
	}
	var seed_event *cl.Event
	seed_event, err = queue.EnqueueWriteBuffer(seed_buf, false, 0, int(unsafe.Sizeof(seed_arr[0]))*totalCount, unsafe.Pointer(&seed_arr[0]), events)
	if err != nil {
		log.Fatalln("failed to write seed buffer to device in threefry.init: %+v \n", err)
	}
	if err = queue.Flush(); err != nil {
		log.Printf("flush queue in threefry.init failed: %+v \n", err)
	}

	// seed the RNG
	event := k_threefry_seed_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), unsafe.Pointer(seed_buf),
		&config{[]int{totalCount}, []int{p.GetGroupSize()}}, queue, []*cl.Event{seed_event})

	p.Ini = true
	if err = queue.Flush(); err != nil {
		log.Printf("second flush queue in threefry.init failed: %+v \n", err)
	}

	return event
}

func (p *THREEFRY_status_array_ptr) GenerateUniform(d_data unsafe.Pointer, data_size int, queue *cl.CommandQueue, events []*cl.Event) *cl.Event {
	var err error

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = queue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in beginning of threefry.generateuniform: %+v \n", err)
		}
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("wait for events in threefry.generateuniform failed: %+v \n", err)
		}
		timer.Start("threefry_uniform")
	}

	// execute
	event := k_threefry_uniform_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, queue, events)

	if err = queue.Flush(); err != nil {
		log.Printf("flush queue in threefry.generateuniform failed: %+v \n", err)
	}

	if Synchronous { // debug
		if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for event at end of threefry.generateuniform: %+v \n", err)
		}
		timer.Stop("threefry_uniform")
	}

	return event
}

func (p *THREEFRY_status_array_ptr) GenerateNormal(d_data unsafe.Pointer, data_size int, queue *cl.CommandQueue, events []*cl.Event) *cl.Event {
	var err error

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		if err = queue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in beginning of threefry.generatenormal: %+v \n", err)
		}
		if err = cl.WaitForEvents(events); err != nil {
			log.Printf("wait for events in threefry.generatenormal failed: %+v \n", err)
		}
		timer.Start("threefry_normal")
	}

	// execute
	event := k_threefry_normal_async(unsafe.Pointer(p.Status_key), unsafe.Pointer(p.Status_counter),
		unsafe.Pointer(p.Status_result), unsafe.Pointer(p.Status_tracker), d_data, data_size,
		&config{[]int{p.GetStatusSize()}, []int{p.GetGroupSize()}}, queue, events)

	if err = queue.Flush(); err != nil {
		log.Printf("flush queue in threefry.generatenormal failed: %+v \n", err)
	}

	if Synchronous { // debug
		if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
			log.Printf("failed to wait for event in threefry.generatenormal: %+v \n", err)
		}
		timer.Stop("threefry_normal")
	}

	return event
}
