#Sobre cada modelo - empiezo con uno - CanESM5 - calculo el ENSO, analizo la serie temporal. Hago composites de los eventos. Separo los Ninios y analizo los falvors de los ninios. 

#Hago composites de las ssts y de zg 200hPa y 500hPa además de ua y de SLP para caracterizar el SAM en esa estacion. Analizo DJF por el momento. 
import numpy as np
import xarray as xr
import os
import funciones
import pandas as pd
import clases

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
dato_sst = funciones.cargo_todo_crudos_remap(scenarios,models,ruta,var[0])
dato_zg = cargo_todo_crudos_remap(scenarios,models,ruta,var[1])

#search for EN years
nino34 = sst.nino34.rolling(time=3, center=True).mean().dropna("time")
nino34 = funciones.seasonal_data(nino34,"DJF")
phase = np.zeros_like(nino34.values)
phase[nino34.values > .5*np.std(nino34.values)]=1 # El nino
phase[nino34.values < -.5*np.std(nino34.values)]=-1 # La nina
phase = xr.DataArray(phase,dims = ['phase'],name='time_phase',coords=[phase])

#search for flavors
C_index = sst.C_index; E_index = sst.E_index
C_enso = (np.abs(C_index) >=  np.abs(E_index)) 
nino = nino34 > 0.5*np.std(nino34)
nina = nino34 < -0.5*np.std(nino34)
flavors = np.zeros_like(nino34.values)
flavors[np.logical_and(nino , C_enso)] = 2 #Central Niño
flavors[np.logical_and(nina, C_enso)] = -2 #Central Niña
flavors[np.logical_and(nino , np.logical_not(C_enso))] = 1 #Eastern Niño
flavors[np.logical_and(nina , np.logical_not(C_enso))] = -1 #Eastern Niña
flavors = xr.DataArray(flavors,dims = ['flavors'],name='time_flavor',coords=[flavors])


#Junto los experimentos historical y ssp585 y veo como son los eventos ninio y ninia
full_period = xr.merge([dato_sst[scenarios[0]][MODEL][0].sel(time=slice('1950','2014')).tos,dato_sst[scenarios[1]][MODEL][0].sel(time=slice('2015','2099')).tos])
full_period = funciones.anomalias(full_period,full_period.isel(time=slice(0,12*50)))
lat = full_period.lat; lon = full_period.lon
#Calculo composites
enso_phase = xr.DataArray(data=full_period.tos.values[1:-1,:,:],dims=["phase","lat","lon"],coords=dict(phase = phase,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)
enso_flavors = xr.DataArray(data=full_period.tos.values[1:-1,:,:],dims=["flavors","lat","lon"],coords=dict(flavors = flavors,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)

EN = enso_phase.where(enso_phase.phase == 1,drop=True).mean(dim='phase')
LN = enso_phase.where(enso_phase.phase == -1,drop=True).mean(dim='phase')
EE = enso_flavors.where(enso_flavors.flavors == 1,drop=True).mean(dim='flavors')
LE = enso_flavors.where(enso_flavors.flavors == -1,drop=True).mean(dim='flavors')
EC = enso_flavors.where(enso_flavors.flavors == 2,drop=True).mean(dim='flavors')
LC = enso_flavors.where(enso_flavors.flavors == -2,drop=True).mean(dim='flavors')
Neutral = enso_flavors.where(enso_flavors.flavors == 0,drop=True).mean(dim='flavors')



fig = plt.figure(figsize=(16, 6),dpi=300,constrained_layout=True)
plt.title('CanESM5')
levels = np.arange(-2,2,0.25)
ax1 = plt.subplot(2,2,1)
im1=ax1.contourf(lon,lat, EC.values, levels=levels,cmap='RdBu_r', extend='both')
ax1.set_title('a) El Niño Central',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
#Add gridlines

ax2 = plt.subplot(2,2,2)
im2=ax2.contourf(lon,lat, EE.values, levels=levels,cmap='RdBu_r', extend='both')
ax2.set_title('b) El Niño Eastern',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

ax3 = plt.subplot(2,2,3)
im3=ax3.contourf(lon,lat,LC.values, levels=levels,cmap='RdBu_r', extend='both')
ax3.set_title('c) La Niña Central',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

ax4 = plt.subplot(2,2,4)
im4=ax4.contourf(lon,lat, LE.values, levels=levels,cmap='RdBu_r', extend='both')
ax4.set_title('d) La Niña Eastern',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

plt1_ax = plt.gca()
left, bottom, width, height = plt1_ax.get_position().bounds
colorbar_axes = fig.add_axes([left + 0.46, bottom,0.02, height*2])
cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
cbar.set_label(r'sea surface temperature (K)',fontsize=14) #rotation= radianes
cbar.ax.tick_params(axis='both',labelsize=14)
fig.tight_layout()
fig.savefig(path_fig+'/composites_prueba.png')


#Junto los experimentos historical y ssp585, remuevo el calentamiento global con una regresion y veo como son los eventos ninio y ninia
full_period = xr.merge([dato_sst[scenarios[0]][MODEL][0].sel(time=slice('1950','2014')).tos,dato_sst[scenarios[1]][MODEL][0].sel(time=slice('2015','2099')).tos])
full_period = funciones.anomalias(full_period,full_period.isel(time=slice(0,12*50)))
gw = full_period.mean(dim='lon').mean(dim='lat')
reg = clases.regression()
regressors = pd.DataFrame({'gw':gw.tos})
reg.regressors = regressors
aux = full_period.tos.rename({'time':'year'})
out_reg1 = reg.perform_regression(aux)
ssts_wo_gw = full_period - out_reg1['gw']['coef']*gw.tos
lat = full_period.lat; lon = full_period.lon
#Calculo composites
enso_phase = xr.DataArray(data=ssts_wo_gw.tos.values[1:-1,:,:],dims=["phase","lat","lon"],coords=dict(phase = phase,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)
enso_flavors = xr.DataArray(data=ssts_wo_gw.tos.values[1:-1,:,:],dims=["flavors","lat","lon"],coords=dict(flavors = flavors,lat=lat,lon=lon),attrs=dict(description="SST anomaly",units="degC",),)

EN = enso_phase.where(enso_phase.phase == 1,drop=True).mean(dim='phase')
LN = enso_phase.where(enso_phase.phase == -1,drop=True).mean(dim='phase')
EE = enso_flavors.where(enso_flavors.flavors == 1,drop=True).mean(dim='flavors')
LE = enso_flavors.where(enso_flavors.flavors == -1,drop=True).mean(dim='flavors')
EC = enso_flavors.where(enso_flavors.flavors == 2,drop=True).mean(dim='flavors')
LC = enso_flavors.where(enso_flavors.flavors == -2,drop=True).mean(dim='flavors')
Neutral = enso_flavors.where(enso_flavors.flavors == 0,drop=True).mean(dim='flavors')



fig = plt.figure(figsize=(16, 6),dpi=300,constrained_layout=True)
plt.title('CanESM5')
levels = np.arange(-2,2,0.25)
ax1 = plt.subplot(2,2,1)
im1=ax1.contourf(lon,lat, EC.values, levels=levels,cmap='RdBu_r', extend='both')
ax1.set_title('a) El Niño Central',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
#Add gridlines

ax2 = plt.subplot(2,2,2)
im2=ax2.contourf(lon,lat, EE.values, levels=levels,cmap='RdBu_r', extend='both')
ax2.set_title('b) El Niño Eastern',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

ax3 = plt.subplot(2,2,3)
im3=ax3.contourf(lon,lat,LC.values, levels=levels,cmap='RdBu_r', extend='both')
ax3.set_title('c) La Niña Central',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

ax4 = plt.subplot(2,2,4)
im4=ax4.contourf(lon,lat, LE.values, levels=levels,cmap='RdBu_r', extend='both')
ax4.set_title('d) La Niña Eastern',fontsize=18, loc='left')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

plt1_ax = plt.gca()
left, bottom, width, height = plt1_ax.get_position().bounds
colorbar_axes = fig.add_axes([left + 0.46, bottom,0.02, height*2])
cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
cbar.set_label(r'sea surface temperature (K)',fontsize=14) #rotation= radianes
cbar.ax.tick_params(axis='both',labelsize=14)
fig.tight_layout()
fig.savefig(path_fig+'/composites_prueba.png')
