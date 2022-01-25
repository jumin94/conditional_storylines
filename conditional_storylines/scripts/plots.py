#Imports
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import os
import glob
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil
import logging
import os, fnmatch
from scipy import signal 
import statsmodels.api as sm
from scipy import stats

def add_box(box,texto,color='black',text_color='black'):
    x_start, x_end = box[0], box[1]
    y_start, y_end = box[2], box[3]
    margin = 0.0007
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start) * margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start) * margin_fractions
    box_x_points = x_lower + (x_upper - x_lower) * np.array([0, 1, 1, 0, 0])
    box_y_points = y_lower + (y_upper - y_lower) * np.array([0, 0, 1, 1, 0])
    plt.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1, color=color, linestyle='-')
    #plt.text(x_start + (x_end - x_start)*0.4, y_start + (y_end - y_start)*0.4, texto,transform=ccrs.PlateCarree( ),color=text_color)
    
def fig_sst_multiple(dato,dato_r2,t,series,titulo,titulo_serie,levels = [np.arange(-1,1.1,.1)]):
    n = len(dato)
    lon = dato[0].lon; lat = dato[0].lat
    fig = plt.figure(figsize=(n*5, n*3),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(180)
    for ii in range(n):
        #print('mapa',int(ii*2+1))
        level=levels[ii]
        #print(ii+1,np.max(dato[ii].values),np.max(dato[ii].values)/10)
        ax1 = plt.subplot(n,2,int(ii*2+1),projection=proj)
        im1=ax1.contourf(lon, lat,dato[ii].values,level,transform=data_crs,cmap='RdBu_r',extend='both')
        clevels = np.arange(-1,1,0.2)
        r2 =  ax1.contour(lon, lat,dato_r2[ii].values,level,colors='k',linewidths=0.5,transform=data_crs)
        ax1.clabel(r2, fontsize=10, inline=True)
        ax1.set_title(titulo[ii],fontsize=10)
        ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
        ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
        ax1.set_extent([-60, 120, -40, 40], ccrs.PlateCarree(central_longitude=180))
        #Add gridlines
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.3, color='gray', alpha=0.01, linestyle='-')
        #Add niño boxes
        # Niño 4
        box = [160,210,-5,5]
        add_box(box,'Niño 4','green','green')
        box = [190,240,-5,5]
        add_box(box,'Niño 3.4','red','red')
        box = [210,270,-5,5]
        add_box(box,'Niño 3')
        box = [270,280,-10,0]
        add_box(box,'Niño 1+2')
        box = [140,170,-5,5]
        add_box(box,'West Pacific')
        
        if ii == 1:
            plt1_ax = plt.gca()
            left, bottom, width, height = plt1_ax.get_position().bounds
            colorbar_axes = fig.add_axes([left + 0.6, bottom,0.02, height])
            cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
            cbar.set_label(r'K',fontsize=10) #rotation= radianes
            cbar.ax.tick_params(axis='both',labelsize=5)
            
            
        [slope, interc, r_va, p_val, z] = stats.linregress(np.arange(0, len(series[ii])),
                                                   series[ii])
        #plot time serie
        ax1 = plt.subplot(n,2,int(ii*2)+2)
        #print('serie',int(ii*2)+2)
        try:
          ax1.plot(t[ii],series[ii], 'r', linewidth=1.5,label='slope: '+str(round(slope,3)))
        except:
           ax1.plot(series[ii], 'r', linewidth=1.5,label='slope: '+str(round(slope,3)))
        plt.axhline(y=0,xmin=0,linestyle='--',linewidth=0.8,color='k',alpha=0.5)
        ax1.set_ylim((np.min(series[ii])-1, np.max(series[ii])+1))
        ax1.legend()
        x = np.arange(0, len(series[ii]))
        try:
          ax1.plot(t[ii],interc+x*slope,'--k')
        except:
          ax1.plot(interc+x*slope,'--k')
        ax1.set_title(titulo_serie[ii])
        fig.tight_layout()

    return fig


def fig_sst_multiple2(dato,dato_r2,t,series,titulo,titulo_serie,levels = [np.arange(-1,1.1,.1)]):
    n = len(dato)
    lon = dato[0].lon; lat = dato[0].lat
    fig = plt.figure(figsize=(n*6, n*3),dpi=200,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(180)
    for ii in range(n):
        #print('mapa',int(ii*2+1))
        level=levels[ii]
        #print(ii+1,np.max(dato[ii].values),np.max(dato[ii].values)/10)
        ax1 = plt.subplot(n,2,int(ii*2+1),projection=proj)
        im1=ax1.contourf(lon, lat,dato[ii].values,level,transform=data_crs,cmap='RdBu_r',extend='both')
        clevels = np.arange(-1,1,0.2)
        r2 =  ax1.contour(lon, lat,dato_r2[ii].values,level,colors='k',linewidths=0.5,transform=data_crs)
        ax1.clabel(r2, fontsize=10, inline=True)
        ax1.set_title(titulo[ii],fontsize=10)
        ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
        ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
        #ax1.set_extent([-60, 120, -40, 40], ccrs.PlateCarree(central_longitude=180))
        ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
        #Add gridlines
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.3, color='gray', alpha=0.01, linestyle='-')
        #Add niño boxes
        # Niño 4
        box = [160,210,-5,5]
        add_box(box,'Niño 4','green','green')
        box = [190,240,-5,5]
        add_box(box,'Niño 3.4','red','red')
        box = [210,270,-5,5]
        add_box(box,'Niño 3')
        box = [270,280,-10,0]
        add_box(box,'Niño 1+2')
        box = [140,170,-5,5]
        add_box(box,'West Pacific')
        
        if ii == 1:
            plt1_ax = plt.gca()
            left, bottom, width, height = plt1_ax.get_position().bounds
            colorbar_axes = fig.add_axes([left + 0.6, bottom,0.02, height])
            cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
            cbar.set_label(r'K',fontsize=10) #rotation= radianes
            cbar.ax.tick_params(axis='both',labelsize=5)
            
            
        [slope, interc, r_va, p_val, z] = stats.linregress(np.arange(0, len(series[ii])),
                                                   series[ii])
        #plot time serie
        ax1 = plt.subplot(n,2,int(ii*2)+2)
        #print('serie',int(ii*2)+2)
        try:
          ax1.plot(t[ii],series[ii], 'r', linewidth=1.5,label='slope: '+str(round(slope,3)))
        except:
           ax1.plot(series[ii], 'r', linewidth=1.5,label='slope: '+str(round(slope,3)))
        plt.axhline(y=0,xmin=0,linestyle='--',linewidth=0.8,color='k',alpha=0.5)
        ax1.set_ylim((np.min(series[ii])-1, np.max(series[ii])+1))
        ax1.legend()
        x = np.arange(0, len(series[ii]))
        try:
          ax1.plot(t[ii],interc+x*slope,'--k')
        except:
          ax1.plot(interc+x*slope,'--k')
        ax1.set_title(titulo_serie[ii])
        fig.tight_layout()

    return fig



def plot_sensitivity_zg_carree(fields,fields_pval,title):
 

    #path_era = '/datos/ERA5/mon'
    #zg_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
    #zg_ERA = u_ERA.u.sel(lev=850).mean(dim='time')
        
    backgroud = fields[0]; backgroud_pval = fields_pval[0]
    anomaly = fields[1];anomaly_pval = fields_pval[0]
    lat = background.lat
    lon = np.arange(0,357.188,2.81)
    bgd_1, lon_c = add_cyclic_point(background[0],lon)
    anom_1, lon_c = add_cyclic_point(anomaly[0],lon)
    bgd_2, lon_c = add_cyclic_point(background[1],lon)
    anom_2, lon_c = add_cyclic_point(anomaly[1],lon)
    bgd_3, lon_c = add_cyclic_point(background[2],lon)
    anom_3, lon_c = add_cyclic_point(anomaly[2],lon)
    bgd_p_1, lon_c = add_cyclic_point(background_pval[0],lon)
    anom_p_1,lon_c = add_cyclic_point(anomaly_pval[0],lon)
    bgd_p_2, lon_c = add_cyclic_point(background_pval[1],lon)
    anom_p_2,lon_c = add_cyclic_point(anomaly_pval[1],lon)
    bgd_p_3, lon_c = add_cyclic_point(background_pval[2],lon)
    anom_p_3,lon_c = add_cyclic_point(anomaly_pval[2],lon) 

    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.PlateCarree(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,2,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-2.4,2.8,0.4)
    im1=ax1.contourf(lon_c, lat, bgd_1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax1.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p.min(),0.05,bgd_p.max()]
    #ax1.contourf(lon_c, lat, bgd_p_1,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Background 1950 - 1999',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,2,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    im2=ax2.contourf(lon_c, lat, bgd_2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax2.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p_2.min(),0.05,bgd_p_2.max()]
    #ax2.contourf(lon_c, lat,bgd_p_2,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Background 2000 - 2049',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,2,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, bgd_3,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax3.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p_3.min(),0.05,bgd_p_3.max()]
    #ax3.contourf(lon_c, lat,bgd_p_3,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Background 2050 - 2099',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,2,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, anom_1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax4.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_1.min(),0.05,anom_p_1.max()]
    #ax4.contourf(lon_c, lat,anom_p_1,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Zonal amonaly 1950 - 1999',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax5 = plt.subplot(3,2,5,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-40,48,8)
    im5=ax5.contourf(lon_c, lat, anom_2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax5.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_2.min(),0.05,anom_p_2.max()]
    #ax5.contourf(lon_c, lat, anom_p_2,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Zonal anomaly 2000 - 2049',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,2,6,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs) 
    #clevels = np.arange(-40,48,8)
    im6=ax6.contourf(lon_c, lat, anom_3,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax6.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_3.min(),0.05,anom_p_3.max()]
    #ax5.contourf(lon_c, lat, anom_p_3,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Zonal anomaly 2050 - 2099',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.1, right=1.3, top=0.4)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.35, bottom_4 -0.15, 0.01, height_4*1.4])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.35, bottom_5-0.15 , 0.01, height_5*1.4])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.35, bottom_1-0.4, 0.01, height_1*1.4])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.35, bottom_2-0.4, 0.01, height_2*1.4])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.35, bottom_3-0.4, 0.01, height_3*1.4])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.35, bottom_6-0.15, 0.01, height_6*1.4])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    #plt.savefig(path_fig+'/zgDJF_ENSO_circulation.png',bbox_inches='tight')
    #plt.clf
    plt.title(title)

    return fig


def plot_change(fields,fields_pval,title):
 

    #path_era = '/datos/ERA5/mon'
    #zg_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
    #zg_ERA = u_ERA.u.sel(lev=850).mean(dim='time')
        
    backgroud = fields[0]; backgroud_pval = fields_pval[0]
    anomaly = fields[1];anomaly_pval = fields_pval[0]
    lat = background.lat
    lon = np.arange(0,357.188,2.81)
    bgd_1, lon_c = add_cyclic_point(background[0],lon)
    anom_1, lon_c = add_cyclic_point(anomaly[0],lon)
    bgd_2, lon_c = add_cyclic_point(background[1],lon)
    anom_2, lon_c = add_cyclic_point(anomaly[1],lon)
    bgd_3, lon_c = add_cyclic_point(background[2],lon)
    anom_3, lon_c = add_cyclic_point(anomaly[2],lon)
    bgd_p_1, lon_c = add_cyclic_point(background_pval[0],lon)
    anom_p_1,lon_c = add_cyclic_point(anomaly_pval[0],lon)
    bgd_p_2, lon_c = add_cyclic_point(background_pval[1],lon)
    anom_p_2,lon_c = add_cyclic_point(anomaly_pval[1],lon)
    bgd_p_3, lon_c = add_cyclic_point(background_pval[2],lon)
    anom_p_3,lon_c = add_cyclic_point(anomaly_pval[2],lon) 

    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.PlateCarree(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,2,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-2.4,2.8,0.4)
    im1=ax1.contourf(lon_c, lat, bgd_1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax1.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p.min(),0.05,bgd_p.max()]
    #ax1.contourf(lon_c, lat, bgd_p_1,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Background 1950 - 1999',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,2,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    im2=ax2.contourf(lon_c, lat, bgd_2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax2.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p_2.min(),0.05,bgd_p_2.max()]
    #ax2.contourf(lon_c, lat,bgd_p_2,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Background 2000 - 2049',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,2,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, bgd_3,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax3.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [bgd_p_3.min(),0.05,bgd_p_3.max()]
    #ax3.contourf(lon_c, lat,bgd_p_3,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Background 2050 - 2099',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,2,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, anom_1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax4.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_1.min(),0.05,anom_p_1.max()]
    #ax4.contourf(lon_c, lat,anom_p_1,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Zonal amonaly 1950 - 1999',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax5 = plt.subplot(3,2,5,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-40,48,8)
    im5=ax5.contourf(lon_c, lat, anom_2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax5.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_2.min(),0.05,anom_p_2.max()]
    #ax5.contourf(lon_c, lat, anom_p_2,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Zonal anomaly 2000 - 2049',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,2,6,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs) 
    #clevels = np.arange(-40,48,8)
    im6=ax6.contourf(lon_c, lat, anom_3,clevels,transform=data_crs,cmap='PuOr',extend='both')
    #cnt=ax6.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [anom_p_3.min(),0.05,anom_p_3.max()]
    #ax5.contourf(lon_c, lat, anom_p_3,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Zonal anomaly 2050 - 2099',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.1, right=1.3, top=0.4)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.35, bottom_4 -0.15, 0.01, height_4*1.4])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.35, bottom_5-0.15 , 0.01, height_5*1.4])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.35, bottom_1-0.4, 0.01, height_1*1.4])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.35, bottom_2-0.4, 0.01, height_2*1.4])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.35, bottom_3-0.4, 0.01, height_3*1.4])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('m',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.35, bottom_6-0.15, 0.01, height_6*1.4])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    #plt.savefig(path_fig+'/zgDJF_ENSO_circulation.png',bbox_inches='tight')
    #plt.clf
    plt.title(title)

    return fig


