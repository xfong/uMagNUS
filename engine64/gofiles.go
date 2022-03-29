package engine64

// support for running Go files as if they were mx3 files.

import (
	"flag"
	"log"
	"os"
	"path"

	opencl "github.com/seeder-research/uMagNUS/opencl64"
	util "github.com/seeder-research/uMagNUS/util"
)

var (
	// These flags are shared between cmd/uMagNUS and Go input files.
	Flag_cachedir    = flag.String("cache", os.TempDir(), "Kernel cache directory (empty disables caching)")
	Flag_gpulist     = flag.String("gpu", "", "Comma separated list to specify GPUs to use")
	Flag_host        = flag.Bool("host", false, "Disable GPU acceleration")
	Flag_interactive = flag.Bool("i", false, "Open interactive browser session")
	Flag_od          = flag.String("o", "", "Override output directory")
	Flag_port        = flag.String("http", ":35367", "Port to serve web gui")
	Flag_selftest    = flag.Bool("paranoid", false, "Enable convolution self-test for cuFFT sanity.")
	Flag_silent      = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	Flag_sync        = flag.Bool("sync", false, "Synchronize all OpenCL calls (debug)")
	Flag_forceclean  = flag.Bool("f", false, "Force start, clean existing output directory")
	Flag_failfast    = flag.Bool("failfast", false, "If one simulation fails, stop entire batch immediately")
	Flag_test        = flag.Bool("test", false, "OpenCL test (internal)")
	Flag_version     = flag.Bool("v", true, "Print version")
	Flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	Flag_gpu         = int(-5) // To be set externally
)

// Usage: in every Go input file, write:
//
// 	func main(){
// 		defer InitAndClose()()
// 		// ...
// 	}
//
// This initialises the GPU, output directory, etc,
// and makes sure pending output will get flushed.
func InitAndClose() func() {

	flag.Parse()

	if *Flag_host {
		if Flag_gpu < 0 {
			opencl.Init(Flag_gpu)
		} else {
			log.Fatalln("Cannot disable GPU acceleration while requesting GPU \n")
		}
	} else {
		if Flag_gpu < 0 {
			opencl.Init(0)
		} else {
			opencl.Init(Flag_gpu)
		}
	}
	opencl.Synchronous = *Flag_sync

	od := *Flag_od
	if od == "" {
		od = path.Base(os.Args[0]) + ".out"
	}
	inFile := util.NoExt(od)
	InitIO(inFile, od, *Flag_forceclean)

	GoServe(*Flag_port)

	return func() {
		Close()
	}
}
