package engine

import (
	"io"

	httpfs "github.com/seeder-research/uMagNUS/httpfs"
	util "github.com/seeder-research/uMagNUS/util"
)

const separationline = `
---------------------------------------------------------------------------
`
const bibheader = `
This bibtex file is automatically generated by uMagNUS.

The following list are references relevant for your simulation. If you use
the results of these simulations in any work or publication, we kindly ask
you to cite them.`

var (
	bibfile io.WriteCloser
	library map[string]*bibEntry
)

func init() {
	buildLibrary()
}

func initBib() { // inited in engine.InitIO
	if bibfile != nil {
		panic("bib already initialized")
	}
	var err error
	bibfile, err = httpfs.Create(OD() + "references.bib")
	if err != nil {
		panic(err)
	}
	util.FatalErr(err)
	fprintln(bibfile, bibheader)
	fprintln(bibfile, separationline)
	Refer("vansteenkiste2014") // Make sure that uMagNUS is always referenced
}

type bibEntry struct {
	reason   string
	bibtex   string
	shortref string
	used     bool
}

func Refer(tag string) {
	bibentry, inLibrary := library[tag]
	if bibentry.used || !inLibrary {
		return
	}
	bibentry.used = true
	if bibfile != nil {
		fprintln(bibfile, bibentry.reason)
		fprintln(bibfile, bibentry.bibtex)
		fprintln(bibfile, separationline)
	}
}

func areRefsUsed() bool {
	for _, bibentry := range library {
		if bibentry.used {
			return true
		}
	}
	return false
}

func LogUsedRefs() {
	if !areRefsUsed() {
		return
	}
	LogOut("********************************************************************//")
	LogOut("Please cite the following references, relevant for your simulation. //")
	LogOut("See bibtex file in output folder for justification.                 //")
	LogOut("********************************************************************//")
	for _, bibentry := range library {
		if bibentry.used {
			LogOut("   * " + bibentry.shortref)
		}
	}
}

func buildLibrary() {

	library = make(map[string]*bibEntry)

	library["vansteenkiste2014"] = &bibEntry{
		reason:   "Main paper about Mumax3",
		shortref: "Vansteenkiste et al., AIP Adv. 4, 107133 (2014).",
		bibtex: `
@article{Vansteenkiste2014,
    author  = {Vansteenkiste, Arne and
               Leliaert, Jonathan and
               Dvornik, Mykola and
               Helsen, Mathias and
               Garcia-Sanchez, Felipe and
               {Van Waeyenberge}, Bartel},
    title   = {{The design and verification of Mumax3}},
    journal = {AIP Advances},
    number  = {10},
    pages   = {107133},
    volume  = {4},
    year    = {2014},
    doi     = {10.1063/1.4899186},
    url     = {http://doi.org/10.1063/1.4899186}
}`}

	library["exl2014"] = &bibEntry{
		reason:   "Mumax3 uses Exl's minimizer",
		shortref: "Exl et al., J. Appl. Phys. 115, 17D118 (2014).",
		bibtex: `
@article{Exl2014,
    author  = {Exl, Lukas and
               Bance, Simon and
               Reichel, Franz and
               Schrefl, Thomas and
               {Peter Stimming}, Hans and
               Mauser, Norbert J.},
    title   = {{LaBonte's method revisited: An effective steepest
                descent method for micromagnetic energy minimization}},
    journal = {Journal of Applied Physics},
    number  = {17},
    pages   = {17D118},
    volume  = {115},
    year    = {2014},
    doi     = {10.1063/1.4862839},
    url     = {http://doi.org/10.1063/1.4862839}
}`}

	library["Lel2014"] = &bibEntry{
		reason:   "Mumax3 used function ext_makegrains",
		shortref: "Leliaert et al., J. Appl. Phys. 115, 233903 (2014)",
		bibtex: `
@article{Lel2014,
    author  = {Leliaert, Jonathan and
               Van de Wiele, Ben and
               Vansteenkiste, Arne and
               Laurson, Lasse and
               Durin, Gianfranco and
               Dupr{\'e}, Luc and
               Van Waeyenberge, Bartel},
    title   = {{Current-driven domain wall mobility in polycrystalline permalloy nanowires: A numerical study}},
    journal = {Journal of Applied Physics},
    volume  = {115},
    number  = {23},
    pages   = {233903},
    year    = {2014},
    doi     = {10.1063/1.4883297},
    url     = {http://dx.doi.org/10.1063/1.4883297}
}`}

	library["mulkers2017"] = &bibEntry{
		reason:   "Simulated system has interfacially induced DMI",
		shortref: "Mulkers et al., Phys. Rev. B 95, 144401 (2017).",
		bibtex: `
@article{Mulkers2017,
    author  = {Mulkers, Jeroen and
               Van Waeyenberge, Bartel and
	       Milo{\v{s}}evi{\'{c}}, Milorad V.},
    title   = {{Effects of spatially-engineered Dzyaloshinskii-Moriya
                interaction in ferromagnetic films}},
    journal = {Physical Review B},
    number  = {14},
    pages   = {144401},
    volume  = {95},
    year    = {2017},
    doi     = {10.1103/PhysRevB.95.144401},
    url     = {doi.org/10.1103/PhysRevB.95.144401},
}`}

	library["leliaert2017"] = &bibEntry{
		reason:   "Simulated nonzero temperatures with adaptive time steps",
		shortref: "Leliaert et al., AIP Adv. 7, 125010 (2017).",
		bibtex: `
@article{Leliaert2017,
    author  = {Leliaert, Jonathan and
               Mulkers, Jeroen and
	       De Clercq, Jonas and
	       Coene, Annelies and
               Dvornik, Mykola and
               Van Waeyenberge, Bartel},
    title   = {{Adaptively time stepping the stochastic Landau-Lifshitz-Gilbert equation at nonzero temperature: implementation and validation in MuMax$^3$}},
    journal = {AIP Advances},
    number  = {12},
    pages   = {125010},
    volume  = {7},
    year    = {2017},
    doi     = {doi.org/10.1063/1.5003957},
    url     = {http://aip.scitation.org/doi/10.1063/1.5003957},
}`}

	library["Berg1981"] = &bibEntry{
		reason:   "Computed the topological charge using the formula of Berg and Lüscher",
		shortref: "Berg et al., Nucl. Phys. B 190, 41224 (1981)",
		bibtex: `
@article{Berg1981,
    author  = {Berg, Bernd A
               Lüscher, Martin},
    title   = {{Definition and statistical distributions of a topological number in the lattice O(3) $\sigma$-model}},
    journal = {Nuclear Physics B},
    pages   = {412-424},
    volume  = {190},
    year    = {1981},
    doi     = {doi.org/10.1016/0550-3213(81)90568-X},
    url     = {https://doi.org/10.1016/0550-3213(81)90568-X},
}`}

}
