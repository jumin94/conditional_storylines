#Sobre cada modelo - empiezo con uno - CanESM5 - calculo el ENSO, analizo la serie temporal. Hago composites de los eventos. Separo los Ninios y analizo los falvors de los ninios. 

#Hago composites de las ssts y de zg 200hPa y 500hPa además de ua y de SLP para caracterizar el SAM en esa estacion. Analizo DJF por el momento. 
import numpy as np
import xarray as xr
import os
import funciones

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
path_dato = '/home/julia.mindlin/Tesis/Capitulo3/scripts/EOF_SST_evaluation/datos' 
path_fig = '/home/julia.mindlin/Tesis/Capitulo3/scripts/conditional_storylines/figures'
path_CMIP6 = 

MODEL = 'CanESM5'
SST_FILE = 'SST_indices_CanESM5_1950-2099.nc'
sst = xr.open_dataset(path_dato+'/'+SST_FILE)
models = [
            'CanESM5'
            ]
ruta = '/datos/julia.mindlin/CMIP6_remap'
var = ['/tos','/zg','slp']
scenarios = ['historical','ssp585']
dato_sst = cargo_todo_crudos_remap(scenarios,models,ruta,var[0])
dato_zg = cargo_todo_crudos_remap(scenarios,models,ruta,var[1])

#search for EN years
nino34 = sst.nino34.rolling(time=3, center=True).mean(Keep_attrs=True).dropna("time")
index_ENSO = np.zeros_like(nino34.values)
index_ENSO[nino34.values > .5*np.std(nino34.values)]=1 # El nino
index_ENSO[nino34.values < -.5*np.std(nino34.values)]=-1 # La nina
phase = xr.DataArray(index_ENSO,dims = ['phase'],name='time_phase',coords=[phase])


#search for flavors
C_index = sst.C_index; E_index = sst.E_index
C_enso = (np.abs(C_index) >=  np.abs(E_index)) 
nino = nino34 > 0.5*np.std(nino34)
nina = nino34 < -0.5*np.std(nino34)
flavors = np.zeros_like(nino34.values)
flavor[np.logical_and(nino , C_enso)] = 2 #Central Niño
flavor[np.logical_and(nina, C_enso)] = -2 #Central Niña
flavor[np.logical_and(nino , np.logical_not(C_enso))] = 1 #Eastern Niño
flavors[np.logical_and(nina , np.logical_not(C_enso))] = -1 #Eastern Niña
flavor = xr.DataArray(flavor,dims = ['flavor'],name='time_flavor',coords=[flavor])


#Junto los experimentos historical y ssp585
full_period = xr.merge([dato_sst[scenarios[0]][MODEL][0].sel(time=slice('1950','2014')).tos,dato_sst[scenarios[1]][MODEL][0].sel(time=slice('2015','2099')).tos])
#Calculo composites
enso_phase = xr.DataArray(data=full_period.values,dims=["phase","lat","lon"],coords=dict(phase = phase,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)
enso_flavor = xr.DataArray(data=full_period.values,dims=["flavor","lat","lon"],coords=dict(flavor = flavor,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)

EN = enso_phase.where(enso_phase.phase == 1,drop=True).mean(dim='phase')
LN = enso_phase.where(enso_phase.phase == -1,drop=True).mean(dim='phase')
EE = enso_flavor.where(enso_flavor.flavor == 1,drop=True).mean(dim='flavor')
LE = enso_flavor.where(enso_flavor.flavor == -1,drop=True).mean(dim='flavor')
EC = enso_flavor.where(enso_flavor.flavor == 2,drop=True).mean(dim='flavor')
LC = enso_flavor.where(enso_flavor.flavor == -2,drop=True).mean(dim='flavor')
Neutral = enso_flavor.where(enso_flavor.flavor == 0,drop=True).mean(dim='flavor')
