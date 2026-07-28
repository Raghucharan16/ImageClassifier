"""Measure per-stage and total enhancement time on real scanned pages."""
import os, sys, time
import numpy as np
import enhance as E

BASES = [r'C:\Users\venka\Downloads\Sample2\Sample2',
         r'C:\Users\venka\Downloads\Sample REG Batch\Sample REG Batch']
N = int(sys.argv[1]) if len(sys.argv)>1 else 12

rows=[]
for base in BASES:
    tifs = sorted(f for f in os.listdir(base) if f.lower().endswith('.tif'))[:N]
    stage=dict(load=0,holes=0,borders=0,skew=0,rot=0,speck=0,crop=0,save=0)
    total=0
    for f in tifs:
        p=os.path.join(base,f)
        t=time.time(); g=E.load_bitonal(p); stage['load']+=time.time()-t
        t=time.time(); g,_=E.remove_feed_holes(g); stage['holes']+=time.time()-t
        t=time.time(); g=E.remove_black_borders(g); stage['borders']+=time.time()-t
        t=time.time(); a=E.find_skew(g); stage['skew']+=time.time()-t
        t=time.time(); g=E.rotate_keep_page(g,a); stage['rot']+=time.time()-t
        t=time.time(); g=E.despeckle(g); stage['speck']+=time.time()-t
        t=time.time(); g=E.crop_to_content(g); stage['crop']+=time.time()-t
        t=time.time(); E.save_group4(g, os.path.join(os.environ.get('TEMP','.'),'_b.tif')); stage['save']+=time.time()-t
    n=len(tifs)
    tot=sum(stage.values())/n
    print(f'--- {os.path.basename(base)} ({n} pages) ---')
    for k,v in stage.items():
        print(f'  {k:<9}{v/n*1000:>8.1f} ms')
    print(f'  {"TOTAL":<9}{tot*1000:>8.1f} ms/page   ({1/tot:.1f} pages/sec single-threaded)')
