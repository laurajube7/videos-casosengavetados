import sys, json
from playwright.sync_api import sync_playwright
times=[float(x) for x in sys.argv[1:]]
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
    pg=b.new_page(viewport={'width':720,'height':1280})
    errs=[]
    pg.on('pageerror',lambda e:errs.append(str(e)))
    pg.on('console',lambda m:errs.append(m.text) if m.type=='error' else None)
    pg.goto('file://WORK/site/index.html')
    pg.wait_for_function('window.__ready===true',timeout=15000)
    for t in times:
        pg.evaluate(f'seek({t})')
        pg.screenshot(path=f'qa/f_{t:06.2f}.png')
    print('errors',errs[:6])
    b.close()
