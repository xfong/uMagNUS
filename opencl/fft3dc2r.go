package opencl

import (
	"log"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	timer "github.com/seeder-research/uMagNUS/timer"
)

// 3D single-precision real-to-complex FFT plan.
type fft3DC2RPlan struct {
	fftplan
	size [3]int
}

// 3D single-precision real-to-complex FFT plan.
func newFFT3DC2R(Nx, Ny, Nz int) fft3DC2RPlan {
	var err error
	var queue *cl.CommandQueue
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in newFFT3DC2R: %+v \n", err)
	}
	handle := cl.NewVkFFTPlan(ClCtx, queue)
	handle.VkFFTSetFFTPlanSize([]int{Nx, Ny, Nz})

	return fft3DC2RPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) error {
	var err error
	var event *cl.Event
	var queue *cl.CommandQueue

	if Synchronous {
		if err = WaitLastMarker(); err != nil {
			log.Printf("failed to wait for last event in beginning of fft3dc2r.execasync: %+v \n", err)
		}
		timer.Start("bwfft")
	}

	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		log.Panicf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len())
	}
	okdstlen := p.OutputLenFloats()
	if dst.Len() != okdstlen {
		log.Panicf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len())
	}
	tmpPtr := src.DevPtr(0)
	srcMemObj := *(*cl.MemObject)(tmpPtr)
	tmpPtr = dst.DevPtr(0)
	dstMemObj := *(*cl.MemObject)(tmpPtr)

	// sequence command according to queue
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in fft3dc2r.execasync: %+v \n", err)
	}
	if event, err = queue.EnqueueMarkerWithWaitList(ClLastEvent); err != nil {
		log.Panicf("failed to enqueue barrier in fft3dc2r.execasync: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in fft3dc2r.execasync: %+v \n", err)
	}

	// set event markers
	AddEventToSequence(event)
	UpdateLastEventSingle(event)
	p.handle.SetQueueEvent(event)

	// execute
	if err = p.handle.EnqueueBackwardTransform([]*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}); err != nil {
		log.Printf("Failed to enqueue bwFFT: %+v \n", err)
	}

	// set event marker
	event = p.handle.GetQueueEvent()
	AddEventToSequence(event)
	UpdateLastEventSingle(event)
	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			log.Panicf("failed to wait for last event in fft3dc2r.execasync: %+v \n", err)
		}
		timer.Stop("bwfft")
	}

	return err
}

// 3D size of the input array.
func (p *fft3DC2RPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return 2 * (p.size[X]/2 + 1), p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DC2RPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DC2RPlan) InputLenFloats() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DC2RPlan) OutputLenFloats() int {
	return prod3(p.OutputSizeFloats())
}

// Return command queue associated with the plan
func (p *fft3DC2RPlan) GetCommandQueue() *cl.CommandQueue {
	return p.handle.GetCommandQueue()
}
