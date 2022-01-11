from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import math

import openmc

### Materials ###
air = openmc.Material(1, name="air")
air.add_element('C', 0.000150)
air.add_element('N', 0.784431)
air.add_element('O', 0.210748)
air.add_element('Ar', 0.004671)
air.set_density('g/cm3', 0.001205)

iron = openmc.Material(2, name="iron")
iron.add_element('Fe', 1.0)
iron.set_density('g/cm3', 7.874000)

lead = openmc.Material(3, name="lead")
lead.add_element('Pb', 1.0)
lead.set_density('g/cm3', 11.350000)

concrete = openmc.Material(4, name="concrete")
concrete.add_element('H', 0.084739)
concrete.add_element('O', 0.604079)
concrete.add_element('Na', 0.012523)
concrete.add_element('Al', 0.024842)
concrete.add_element('Si', 0.241921)
concrete.add_element('Ca', 0.027244)
concrete.add_element('Fe', 0.004652)
concrete.set_density('g/cm3', 2.250000)

materials = openmc.Materials([iron, air, lead, concrete])
materials.export_to_xml('/home/MCcos/materials.xml')

### Geometry ###
w = 33.5 ### width of shield in one direction from edge of the source
l = 40.2 ### shield thickness
wbox = 68
lbox = 41

radius = openmc.XCylinder(r=w, boundary_type='transmission')
hot_end = openmc.XPlane(x0=0, boundary_type='transmission')
ambient_end = openmc.XPlane(x0=l, boundary_type='transmission')

box_down = openmc.YPlane(y0=-wbox/2, boundary_type='vacuum')
box_up = openmc.YPlane(y0=wbox/2, boundary_type='vacuum')
box_right = openmc.XPlane(x0=lbox, boundary_type='vacuum')
box_left = openmc.XPlane(x0=-1, boundary_type='vacuum')
box_high = openmc.ZPlane(z0=wbox/2, boundary_type='vacuum')
box_low = openmc.ZPlane(z0=-wbox/2, boundary_type='vacuum')

box = -box_up & +box_down & +box_left & -box_right & -box_high & +box_low

shield_region = -radius & +hot_end & -ambient_end
air_region = box & ~shield_region

shield = openmc.Cell(name="shield")
shield.fill = concrete
shield.region = shield_region

box_air = openmc.Cell(name="box air")
box_air.fill = air
box_air.region = air_region

root_universe = openmc.Universe(cells=(shield, box_air))

geometry = openmc.Geometry(root_universe)
geometry.export_to_xml('/home/wiktor/Pulpit/MCcos/geometry.xml')

### Settings ###
phi = openmc.stats.Uniform(0, 2*math.pi)
cos_dist = openmc.stats.Tabular([0, 1], [0, 1])
angular = openmc.stats.PolarAzimuthal(phi=phi, mu=cos_dist, reference_uvw=(1.0, 0.0, 0.0))
iso = openmc.stats.Isotropic()

source_point = openmc.stats.Point(xyz=(0, 0, 0))
Cs137 = openmc.stats.Discrete([0.661e6], [1.0])
Co60 = openmc.stats.Discrete([1.173e6, 1.332e6], [0.5, 0.5])
source = openmc.Source(space=source_point, angle=angular, energy=Cs137)
source.particle = 'photon'

settings = openmc.Settings()
settings.source = source
settings.batches = 10000
settings.particles = 120000
settings.run_mode = 'fixed source'
settings.photon_transport = True

settings.export_to_xml('/home/MCcos/settings.xml')

### Tallies ###

tallies = openmc.Tallies()

energy_bins, dose_coeffs = openmc.data.dose_coefficients(particle="photon", geometry="ISO")

energy_function_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)

mesh = openmc.RegularMesh()
mesh.dimension = [10*lbox, 10*wbox, 1]
mesh.lower_left = [0, -wbox/2, -0.5]
mesh.upper_right = [lbox, wbox/2, 0.5]

mesh_filter = openmc.MeshFilter(mesh)

particle_filter = openmc.ParticleFilter('photon')

dose_tally_photon = openmc.Tally(name="photon_energy_dose_mesh")
dose_tally_photon.scores = ["flux"]

dose_tally_photon.filters = [mesh_filter, particle_filter, energy_function_filter]

tallies.append(dose_tally_photon)

tallies.export_to_xml('/home/MCcos/tallies.xml')

### Run ###

openmc.run()

### Data processing ###

sp = openmc.StatePoint('statepoint.10000.h5')
tally = sp.get_tally(scores=['flux'])
flux = tally.get_slice(scores=['flux'])
flux.mean.shape = (10*wbox, 10*lbox)
flux.std_dev.shape = (10*wbox, 10*lbox)

l_index = int(10*l)
hmin = int(10*(wbox/2-w))
hmax = int(10*(wbox/2+w))

ar_m = []
ar_dev = []

for i in range (0, l_index):
    m = flux.mean[hmin][i]
    d = flux.std_dev[hmin][i]
    ar_m.append(m)
    ar_dev.append(d)

for i in range (hmin, hmax+1):
    m = flux.mean[i][l_index]
    d = flux.std_dev[i][l_index]
    ar_m.append(m)
    ar_dev.append(d)

for i in range (l_index-1, -1, -1):
    m = flux.mean[hmax][i]
    d = flux.std_dev[hmax][i]
    ar_m.append(m)
    ar_dev.append(d)

mean = np.array(ar_m)
dev = np.array(ar_dev)

uni_m = mean[l_index:len(ar_m)-l_index].max()
uni_dev = dev[np.where(mean == uni_m)]
norm_mean = ar_m / uni_m
norm_dev = (dev + uni_dev * mean / uni_m) / uni_m
x = np.linspace(0, len(norm_mean), num=len(norm_mean))

left_max = norm_mean[0:l_index].max()
right_max = norm_mean[len(norm_mean)-l_index:len(norm_mean)].max()
left_err = norm_dev[np.where(norm_mean == left_max)]
right_err = norm_dev[np.where(norm_mean == right_max)]
print(left_max, left_err)
print(right_max, right_err)

plt.scatter(x, norm_mean)
plt.errorbar(x, norm_mean, yerr=norm_dev, ls='none', elinewidth=1, markeredgewidth=1)
plt.xlim(left=0, right=len(norm_mean))
plt.ylim(bottom=0)
plt.savefig('137Cs_conc_100.png')

